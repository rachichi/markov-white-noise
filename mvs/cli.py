"""mvs command line."""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
from pathlib import Path

from .config import ConfigError, Speaker, load_speaker


def _setup_logging(spk: Speaker | None, command: str, verbose: bool) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if spk is not None:
        spk.logs_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.handlers.RotatingFileHandler(
            spk.logs_dir / f"{command}.log", maxBytes=5_000_000, backupCount=5))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mvs", description="Markov Voice Stream: clearly labelled Markov chain "
                                "remixes of real speech clips, audio only.")
    p.add_argument("--root", default=os.getcwd(), help="project root containing speakers/ (default: cwd)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest", help="stage 1: convert source audio and record it in SOURCES.md")
    s.add_argument("speaker")
    s.add_argument("--url", help="download one directly linked public audio file first")
    s.add_argument("--filename", help="file name for --url downloads")
    s.add_argument("--title")
    s.add_argument("--date")
    s.add_argument("--license", help="license or terms of use")
    s.add_argument("--force", action="store_true")

    s = sub.add_parser("transcribe", help="stages 2-3: WhisperX transcription and word alignment")
    s.add_argument("speaker")
    s.add_argument("--force", action="store_true")

    for name, help_ in (("vocab", "stage 4: choose vocabulary and report coverage gaps"),
                        ("clips", "stage 5: cut, rank and normalize clips"),
                        ("chain", "stage 6: build the transition matrix")):
        sub.add_parser(name, help=help_).add_argument("speaker")

    s = sub.add_parser("text", help="print generated sentences (text only)")
    s.add_argument("speaker")
    s.add_argument("-n", type=int, default=10)
    s.add_argument("--seed", type=int)

    s = sub.add_parser("preview", help="render the disclaimer plus n sentences to a local file")
    s.add_argument("speaker")
    s.add_argument("-n", type=int, default=5)
    s.add_argument("-o", "--out", help="output path (.wav, .mp3 or .opus); default <work_dir>/out/preview.wav")
    s.add_argument("--seed", type=int)

    s = sub.add_parser("export-web", help="chain, word clips and transition matrix for the web page")
    s.add_argument("speaker")
    s.add_argument("--top", type=int, default=0, help="limit the matrix to the N most frequent entries (0 = all)")
    s.add_argument("--out", help="output folder (default public/speakers/<slug>)")

    s = sub.add_parser("stream", help="stream to Icecast forever")
    s.add_argument("speaker")
    s.add_argument("--icecast", default=os.environ.get("MVS_ICECAST_URL"),
                   help="icecast://source:PASSWORD@host:8000/mount (or set MVS_ICECAST_URL)")
    s.add_argument("--landing", help="also write the disclosure landing page to this folder")

    s = sub.add_parser("render", help="files mode: render consecutive audio files")
    s.add_argument("speaker")
    s.add_argument("--hours", type=float, required=True)
    s.add_argument("--out", help="output folder (default output.out_dir)")
    s.add_argument("--rss-base-url", help="also write feed.xml with enclosures under this URL")
    s.add_argument("--seed", type=int)

    s = sub.add_parser("purge", help="remove a source and every clip derived from it, then rebuild")
    s.add_argument("--source", required=True, help="source id or file name (see SOURCES.md)")
    s.add_argument("--speaker", help="limit to one speaker")
    s.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.root).resolve()
    try:
        spk = load_speaker(args.speaker, root) if getattr(args, "speaker", None) else None
        _setup_logging(spk, args.command, args.verbose)
        return _run(args, spk, root)
    except (ConfigError, RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


def _run(args, spk: Speaker | None, root: Path) -> int:
    cmd = args.command
    if cmd == "ingest":
        from .ingest import add_remote_source, ingest

        if args.url:
            add_remote_source(spk, args.url, {"title": args.title, "date": args.date, "license": args.license},
                                         args.filename)
        skipped = ingest(spk, force=args.force)
        return 1 if skipped else 0
    if cmd == "transcribe":
        from .transcribe import isolate, transcribe

        transcribe(spk, force=args.force)
        isolate(spk, force=args.force)
        return 0
    if cmd == "vocab":
        from .vocab import build_vocab

        build_vocab(spk)
        print(spk.vocab_report.read_text().split("\n## ")[0])
        return 0
    if cmd == "clips":
        from .clips import build_clips

        build_clips(spk)
        print(spk.clips_report.read_text().split("\n## ")[0])
        return 0
    if cmd == "chain":
        from .chain import build_chain

        build_chain(spk)
        return 0
    if cmd == "text":
        from .generator import SentenceGenerator

        gen = SentenceGenerator(spk, seed=args.seed)
        for _ in range(args.n):
            print(gen.next_sentence().text)
        return 0
    if cmd == "preview":
        from .deliver import preview

        out = Path(args.out) if args.out else spk.out_dir / "preview.wav"
        for text in preview(spk, args.n, out, seed=args.seed):
            print(text)
        print(f"-> {out}")
        return 0
    if cmd == "export-web":
        from .web import export_web

        out = Path(args.out) if args.out else root / "public" / "speakers" / spk.slug
        for path in export_web(spk, out, args.top):
            print(path)
        return 0
    if cmd == "render":
        from .deliver import render

        for path in render(spk, args.hours, Path(args.out) if args.out else None, args.rss_base_url, args.seed):
            print(path)
        return 0
    if cmd == "stream":
        from .deliver import stream

        if not args.icecast:
            raise ConfigError("pass --icecast or set MVS_ICECAST_URL")
        stream(spk, args.icecast, Path(args.landing) if args.landing else None)
        return 0
    if cmd == "purge":
        from .purge import find_source, purge

        hits = [(s, src) for s, src in find_source(root, args.source) if not args.speaker or s.slug == args.speaker]
        if not hits:
            raise ValueError(f"no source matching '{args.source}' in any speaker manifest")
        for s, src in hits:
            print(f"{s.slug}: {src['id']} ({src['file']}) {src['title']}")
        if not args.yes and input("Delete these sources and every clip derived from them? [y/N] ").lower() != "y":
            print("aborted")
            return 1
        for s, src in hits:
            for path in purge(s, src):
                print(f"deleted {path}")
        return 0
    raise AssertionError(cmd)


if __name__ == "__main__":
    sys.exit(main())
