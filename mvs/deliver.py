"""Stage 9: audio-only delivery. Preview files, chunked file rendering (with optional RSS), and Icecast streaming."""

from __future__ import annotations

import html
import json
import logging
import math
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from xml.sax.saxutils import escape

from . import audio
from .config import Speaker, require_disclosure
from .generator import SentenceGenerator
from .ingest import require_ffmpeg
from .program import Program
from .sources import Manifest
from .tts import disclaimer_audio

log = logging.getLogger("mvs.deliver")
ARTIST = "Markov chain remix of public speech clips (not real statements)"
CODECS = {
    "mp3": (["-c:a", "libmp3lame"], "mp3", "audio/mpeg"),
    "opus": (["-c:a", "libopus"], "ogg", "audio/ogg"),
}


def disclosure_tags(spk: Speaker, part: int | None = None) -> dict[str, str]:
    o = spk["output"]
    title = spk.stream_title + (f", part {part}" if part is not None else "")
    return {"title": title, "artist": ARTIST, "album": spk.stream_title,
            "comment": f"{spk.disclaimer_text} {spk.stream_description}", "genre": o["stream_genre"]}


def _metadata_args(tags: dict[str, str]) -> list[str]:
    args = []
    for k, v in tags.items():
        args += ["-metadata", f"{k}={v}"]
    return args


class FileSink:
    """Writes PCM to a WAV (RIFF INFO tags) or, through ffmpeg, to MP3/Opus (ID3 / Vorbis comment tags)."""

    def __init__(self, spk: Speaker, path: Path, tags: dict[str, str]):
        self.sr = spk["audio"]["sample_rate"]
        self.path = path
        self.samples = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        fmt = path.suffix.lstrip(".").lower()
        if fmt == "wav":
            self.wav, self.proc = audio.WavWriter(path, self.sr, tags), None
        elif fmt in CODECS:
            codec, container, _ = CODECS[fmt]
            cmd = [require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "s16le", "-ar", str(self.sr), "-ac", "1", "-i", "-",
                   *codec, "-b:a", f"{spk['output']['bitrate_kbps']}k", *_metadata_args(tags)]
            if fmt == "mp3":
                cmd += ["-id3v2_version", "3", "-write_id3v1", "1"]
            self.wav, self.proc = None, subprocess.Popen(cmd + ["-f", container, str(path)], stdin=subprocess.PIPE)
        else:
            raise ValueError(f"unsupported output format '{fmt}'")

    def write(self, x) -> None:
        self.samples += len(x)
        if self.wav:
            self.wav.write(x)
        else:
            self.proc.stdin.write(audio.to_pcm16(x))

    def close(self) -> None:
        if self.wav:
            self.wav.close()
        else:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {self.path}")


def _program(spk: Speaker, seed: int | None = None) -> Program:
    require_disclosure(spk)
    return Program(spk, SentenceGenerator(spk, seed=seed), disclaimer_audio(spk))


def preview(spk: Speaker, n: int, out: Path, seed: int | None = None, timeline: list | None = None) -> list[str]:
    """Disclaimer plus n sentences to a local file. Returns the sentence texts.

    If timeline is given, it is filled with {kind, text, start, end} (seconds) for every segment.
    """
    program = _program(spk, seed)
    sink = FileSink(spk, out, disclosure_tags(spk))
    texts = []
    try:
        for seg in program.session():
            if timeline is not None:
                start = sink.samples / sink.sr
                timeline.append({"kind": seg.kind, "text": seg.text, "start": round(start, 3),
                                 "end": round(start + len(seg.audio) / sink.sr, 3)})
            sink.write(seg.audio)
            if seg.kind == "sentence":
                texts.append(seg.text)
                if len(texts) >= n:
                    break
    finally:
        sink.close()
        program.close()
    log.info("preview: %d sentences, %.1f s -> %s", n, sink.samples / sink.sr, out)
    return texts


def render(spk: Speaker, hours: float, out_dir: Path | None = None, rss_base_url: str | None = None,
           seed: int | None = None) -> list[Path]:
    """Files mode: consecutive files of file_chunk_minutes, each opening with the disclaimer."""
    if spk["output"]["mode"] not in ("files", "both"):
        raise RuntimeError(f"output.mode is '{spk['output']['mode']}'; set it to 'files' or 'both' to render")
    o = spk["output"]
    out_dir = out_dir or spk.out_dir
    chunk = int(o["file_chunk_minutes"] * 60 * spk["audio"]["sample_rate"])
    count = max(1, math.ceil(hours * 60 / o["file_chunk_minutes"]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    program = _program(spk, seed)
    paths = []
    try:
        for part in range(1, count + 1):
            path = out_dir / f"{spk.slug}-{stamp}-{part:03d}.{o['format']}"
            sink = FileSink(spk, path, disclosure_tags(spk, part))
            for seg in program.session():
                sink.write(seg.audio)
                if sink.samples >= chunk:
                    break
            sink.close()
            paths.append(path)
            log.info("rendered %s (%.1f min)", path.name, sink.samples / sink.sr / 60)
    finally:
        program.close()
    write_landing_page(spk, out_dir)
    if rss_base_url:
        write_rss(spk, out_dir, rss_base_url)
    return paths


def write_rss(spk: Speaker, out_dir: Path, base_url: str) -> Path:
    o = spk["output"]
    mime = {"mp3": "audio/mpeg", "opus": "audio/ogg", "wav": "audio/wav"}[o["format"]]
    files = sorted(out_dir.glob(f"{spk.slug}-*.{o['format']}"), reverse=True)
    base = base_url.rstrip("/")
    items = []
    for f in files:
        pub = format_datetime(datetime.fromtimestamp(f.stat().st_mtime, timezone.utc))
        items.append(
            f"<item><title>{escape(spk.stream_title)}: {escape(f.stem)}</title>"
            f"<description>{escape(spk.disclaimer_text + ' ' + spk.stream_description)}</description>"
            f"<enclosure url=\"{escape(base + '/' + f.name)}\" length=\"{f.stat().st_size}\" type=\"{mime}\"/>"
            f"<guid isPermaLink=\"false\">{escape(f.name)}</guid><pubDate>{pub}</pubDate></item>"
        )
    feed = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<rss version="2.0"><channel>'
        f"<title>{escape(spk.stream_title)}</title><link>{escape(base + '/')}</link>"
        f"<description>{escape(spk.disclaimer_text + ' ' + spk.stream_description)}</description>"
        + "".join(items) + "</channel></rss>\n"
    )
    path = out_dir / "feed.xml"
    path.write_text(feed)
    return path


def write_landing_page(spk: Speaker, out_dir: Path) -> Path:
    """A plain page that carries the disclosure and the source list."""
    rows = "".join(
        f"<tr><td>{html.escape(s['title'])}</td><td><a href=\"{html.escape(s['url'])}\">"
        f"{html.escape(s['url'])}</a></td><td>{html.escape(str(s['date']))}</td>"
        f"<td>{html.escape(s['license'])}</td></tr>"
        for s in Manifest(spk).sources
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(spk.stream_title)}</title>
<style>body{{font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:#1a1a1a;background:#fff;max-width:46rem;margin:0 auto;padding:2rem 1rem;line-height:1.5}}
.notice{{border:1px solid rgba(26,26,26,.3);padding:1rem;margin:1.5rem 0}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}
td,th{{border-bottom:1px solid rgba(26,26,26,.1);padding:.4rem;text-align:left;vertical-align:top;word-break:break-word}}</style></head>
<body><h1>{html.escape(spk.stream_title)}</h1>
<p class="notice"><strong>Not real statements.</strong> {html.escape(spk.disclaimer_text)} {html.escape(spk.stream_description)}</p>
<h2>Sources</h2><table><tr><th>Title</th><th>URL</th><th>Date</th><th>License / terms</th></tr>{rows}</table>
</body></html>
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "index.html"
    path.write_text(page)
    return path


def _mask(url: str) -> str:
    parts = urlsplit(url)
    if parts.password:
        netloc = f"{parts.username}:***@{parts.hostname}" + (f":{parts.port}" if parts.port else "")
        return urlunsplit(parts._replace(netloc=netloc))
    return url


class _Health:
    def __init__(self, port: int):
        self.last_write = 0.0
        self.sessions = 0
        self.started = time.time()
        state = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                age = time.time() - state.last_write if state.last_write else None
                ok = age is not None and age < 5
                body = json.dumps({"ok": ok, "last_write_age_s": age, "sessions": state.sessions,
                                   "uptime_s": round(time.time() - state.started)}).encode()
                self.send_response(200 if ok else 503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def stream(spk: Speaker, icecast_url: str, landing_dir: Path | None = None) -> None:
    """Stream to Icecast forever, reconnecting with backoff. Every (re)connection opens with the disclaimer."""
    if spk["output"]["mode"] not in ("stream", "both"):
        raise RuntimeError(f"output.mode is '{spk['output']['mode']}'; set it to 'stream' or 'both' to stream")
    o, sr = spk["output"], spk["audio"]["sample_rate"]
    fmt = o["format"] if o["format"] in CODECS else "mp3"
    codec, container, content_type = CODECS[fmt]
    cmd = [require_ffmpeg(), "-hide_banner", "-loglevel", "warning", "-re",
           "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "-",
           *codec, "-b:a", f"{o['bitrate_kbps']}k", "-content_type", content_type,
           "-ice_name", spk.stream_title, "-ice_description", f"{spk.disclaimer_text} {spk.stream_description}",
           "-ice_genre", o["stream_genre"], "-ice_public", "0",
           *_metadata_args(disclosure_tags(spk)), "-f", container, icecast_url]
    if landing_dir:
        write_landing_page(spk, landing_dir)
    program = _program(spk)
    health = _Health(o["health_port"])
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    log.info("streaming to %s (health on 127.0.0.1:%d)", _mask(icecast_url), o["health_port"])
    backoff = 1.0
    step = sr // 4  # write in 250 ms chunks; ffmpeg -re paces the pipe in real time
    try:
        while not stop.is_set():
            health.sessions += 1
            began = time.time()
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            try:
                for seg in program.session():
                    if seg.kind == "sentence":
                        log.debug("say: %s", seg.text)
                    for i in range(0, len(seg.audio), step):
                        proc.stdin.write(audio.to_pcm16(seg.audio[i:i + step]))
                        health.last_write = time.time()
                        if stop.is_set():
                            break
                    if stop.is_set() or proc.poll() is not None:
                        break
            except (BrokenPipeError, OSError) as e:
                log.warning("stream connection lost: %s", e)
            finally:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            if stop.is_set():
                break
            backoff = 1.0 if time.time() - began > 60 else min(backoff * 2, 60.0)
            log.warning("ffmpeg exited (code %s); reconnecting in %.0f s", proc.returncode, backoff)
            stop.wait(backoff)
    finally:
        program.close()
        health.server.shutdown()
        log.info("stream stopped")
