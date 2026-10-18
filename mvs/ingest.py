"""Stage 1: copy or download source audio, convert to mono WAV, record it in the source manifest."""

from __future__ import annotations

import logging
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

from . import audio
from .config import Speaker
from .sources import AUDIO_EXTS, Manifest, missing_fields, now_iso, read_sidecar, source_id, write_sidecar

log = logging.getLogger("mvs.ingest")
ALIGN_SR = 16000


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg is not installed (macOS: brew install ffmpeg; Debian: apt install ffmpeg)")
    return exe


def convert(src: Path, dst: Path, sample_rate: int) -> None:
    """Convert any audio file to mono 16-bit PCM WAV at sample_rate."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav")
    if have_ffmpeg():
        subprocess.run([require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                        "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(tmp)], check=True)
    elif src.suffix.lower() == ".wav":
        x, sr = audio.read_wav(src)
        audio.write_wav(tmp, audio.resample(x, sr, sample_rate), sample_rate)
    else:
        require_ffmpeg()
    tmp.replace(dst)


def download(url: str, dest_dir: Path, filename: str | None = None) -> Path:
    """Download a directly linked, publicly accessible audio file. No login, paywall or scraping support."""
    name = filename or Path(urllib.parse.urlparse(url).path).name
    if Path(name).suffix.lower() not in AUDIO_EXTS:
        raise ValueError(f"'{name}' is not a recognised audio file; pass --filename with an audio extension")
    dest = dest_dir / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "mvs-ingest/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)
    return dest


def add_remote_source(spk: Speaker, url: str, meta: dict, filename: str | None = None) -> Path:
    path = download(url, spk.sources_dir, filename)
    sidecar = read_sidecar(spk)
    sidecar[path.name] = {**sidecar.get(path.name, {}), **{k: v for k, v in meta.items() if v is not None}, "url": url}
    write_sidecar(spk, sidecar)
    log.info("downloaded %s", path.name)
    return path


def ingest(spk: Speaker, force: bool = False) -> list[str]:
    """Convert every described audio file in sources_dir. Returns the files skipped for missing metadata."""
    spk.sources_dir.mkdir(parents=True, exist_ok=True)
    sidecar = read_sidecar(spk)
    manifest = Manifest(spk)
    files = sorted(p for p in spk.sources_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("."))
    if not files:
        log.warning("no audio files in %s", spk.sources_dir)
    skipped, seen = [], {}
    for f in files:
        meta = sidecar.get(f.name)
        if meta is None or missing_fields(meta):
            need = missing_fields(meta or {})
            log.error("%s: not ingested, %s missing from %s", f.name, ", ".join(need), spk.sidecar_path.name)
            skipped.append(f.name)
            continue
        sid = source_id(f.name)
        if sid in seen:
            raise ValueError(f"{f.name} and {seen[sid]} map to the same source id '{sid}'; rename one")
        seen[sid] = f.name
        wav16, wavhq = spk.wav16k_dir / f"{sid}.wav", spk.wav_dir / f"{sid}.wav"
        existing = manifest.get(sid)
        fresh = all(p.exists() and p.stat().st_mtime >= f.stat().st_mtime for p in (wav16, wavhq))
        if not force and existing and fresh:
            log.info("%s: already ingested", f.name)
        else:
            log.info("%s: converting", f.name)
            convert(f, wav16, ALIGN_SR)  # for transcription and alignment
            convert(f, wavhq, spk["audio"]["sample_rate"])  # for cutting clips
        x, sr = audio.read_wav(wav16)
        manifest.upsert({
            "id": sid,
            "file": f.name,
            "title": meta["title"],
            "url": meta["url"],
            "date": str(meta["date"]),
            "license": meta["license"],
            "duration_s": round(len(x) / sr, 2),
            "added": (existing or {}).get("added") or now_iso(),
        })
    manifest.save()
    log.info("%d sources in %s", len(manifest.sources), spk.sources_md)
    return skipped
