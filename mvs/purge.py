"""Remove a source file and everything derived from it, then rebuild clips and the chain."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .chain import build_chain
from .clips import build_clips
from .config import Speaker, list_speaker_configs, load_speaker
from .sources import Manifest, now_iso, read_sidecar, write_sidecar

log = logging.getLogger("mvs.purge")


def find_source(root: Path, query: str) -> list[tuple[Speaker, dict]]:
    hits = []
    for cfg in list_speaker_configs(root):
        spk = load_speaker(str(cfg), root)
        src = Manifest(spk).find(query)
        if src:
            hits.append((spk, src))
    return hits


def purge(spk: Speaker, src: dict) -> list[Path]:
    """Delete the original file, its conversions, transcripts and clips; rebuild clips and chain. Returns deleted paths."""
    sid = src["id"]
    deleted = []
    for p in (spk.sources_dir / src["file"], spk.wav16k_dir / f"{sid}.wav", spk.wav_dir / f"{sid}.wav",
              spk.aligned_dir / f"{sid}.json", spk.isolated_dir / f"{sid}.json"):
        if p.exists():
            p.unlink()
            deleted.append(p)

    sidecar = read_sidecar(spk)
    if sidecar.pop(src["file"], None) is not None:
        write_sidecar(spk, sidecar)
    manifest = Manifest(spk)
    manifest.remove(sid)
    manifest.save()

    if spk.clips_json.exists():
        old = json.loads(spk.clips_json.read_text())
        n = sum(r["source_id"] == sid for recs in old["entries"].values() for r in recs)
        log.info("%s: %d clips came from this source; rebuilding clips", sid, n)
        build_clips(spk)  # rewrites clips/ from the remaining sources only
    if spk.vocab_path.exists():
        build_chain(spk)

    with open(spk.work_dir / "purge_log.txt", "a") as f:
        f.write(f"{now_iso()}\tpurged {sid}\t{src['file']}\t{src.get('url', '')}\n")
    rendered = list(spk.out_dir.glob("*")) if spk.out_dir.exists() else []
    if rendered:
        log.warning("rendered output in %s may contain clips from %s; delete or re-render it", spk.out_dir, sid)
    return deleted
