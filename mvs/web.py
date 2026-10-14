"""Export files for the web page, which generates sentences itself: the Markov chain, every playable word clip
packed into one audio sprite, the blocklist terms the chain could produce, and the transition matrix."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from . import audio
from .blocklist import Blocklist
from .chain import END, START, load_chain
from .clips import load_clips
from .config import Speaker, require_disclosure
from .deliver import disclosure_tags
from .generator import SentenceGenerator
from .tts import disclaimer_audio
from .vocab import load_vocab

log = logging.getLogger("mvs.web")
STALE = ("preview.wav", "preview.json", "stream.json", "disclaimer.mp3", "sentences")


def transition_matrix(spk: Speaker, top: int = 0) -> dict:
    """First-order P(next | word) between the most frequent playable entries, as sparse rows.

    Probabilities are over every next token, so a row can sum to less than 1 when `top` hides some columns.
    """
    chain = load_chain(spk)
    playable = {t for t, recs in load_clips(spk)["entries"].items() if recs}
    entries = sorted((e for e in load_vocab(spk)["entries"] if e["text"] in playable), key=lambda e: -e["count"])
    if top:
        entries = entries[:top]
    labels = [e["text"] for e in entries]
    columns = labels + [END]
    index = {t: j for j, t in enumerate(columns)}
    rows = []
    for label in [START] + labels:
        options = chain.options((label,))
        total = sum(options.values())
        cells = sorted([index[t], round(c / total, 4)] for t, c in options.items() if t in index) if total else []
        rows.append({"label": label, "total": total, "cells": cells})
    return {
        "order": chain.order,
        "labels": labels,
        "kinds": [e["kind"] for e in entries],
        "counts": [e["count"] for e in entries],
        "columns": columns,
        "rows": rows,
        "start": START,
        "end": END,
    }


def reachable_blocklist(blocklist: Blocklist, tokens: set[str]) -> dict:
    """Only the terms the chain could actually produce, so the page doesn't ship the whole list."""
    words = {w for t in tokens for w in t.split()}
    terms = sorted({t for ts in blocklist.terms.values() for t in ts if set(t) <= words})
    prefixes = sorted({(head, pre) for head, pre in blocklist.prefixes
                       if set(head) <= words and any(w.startswith(pre) for w in words)})
    return {"terms": [list(t) for t in terms], "prefixes": [[list(h), p] for h, p in prefixes],
            "patterns": [p.pattern for p in blocklist.patterns]}


def generator_data(spk: Speaker, sprite_path: Path) -> dict:
    """Everything the page needs to run SentenceGenerator itself. Clip audio goes into one WAV sprite."""
    require_disclosure(spk)
    gen = SentenceGenerator(spk)
    playable = set(gen.clips)
    keep = playable | {START}
    table = {}
    for ctx, options in gen.chain.table.items():
        nxt = {t: c for t, c in options.items() if t in playable or t == END}
        if nxt and set(ctx) <= keep:
            table["\t".join(ctx)] = nxt

    sr, cc, a = spk["audio"]["sample_rate"], spk["chain"], spk["audio"]
    writer = audio.WavWriter(sprite_path, sr, disclosure_tags(spk))
    try:
        def put(x) -> list[int]:
            offset = writer.data_bytes // 2
            writer.write(x)
            return [offset, len(x)]

        disclaimer = put(disclaimer_audio(spk))
        clips = {}
        for token in sorted(playable):
            clips[token] = []
            for rec in gen.clips[token]:
                x, clip_sr = audio.read_wav(spk.work_dir / rec["path"])
                clips[token].append([*put(audio.resample(x, clip_sr, sr)), rec["prev"], rec["next"],
                                     bool(rec["final"]), round(rec["clip_score"], 3)])
    finally:
        writer.close()

    return {
        "name": spk.name,
        "title": spk.stream_title,
        "description": spk.stream_description,
        "disclaimer_text": spk.disclaimer_text,
        "disclaimer_every": spk["output"]["disclaimer_every_minutes"] * 60,
        "sample_rate": sr,
        "sprite": sprite_path.name,
        "disclaimer": disclaimer,
        "order": gen.order,
        "start": START,
        "end": END,
        "backoff_min_options": cc["backoff_min_options"],
        "min_sentence_words": cc["min_sentence_words"],
        "max_sentence_words": cc["max_sentence_words"],
        "recent_sentences": cc["recent_sentences"],
        "recent_clips": cc["recent_clips"],
        "pause_ms": cc["end_token_pause_ms"],
        "crossfade_ms": a["crossfade_ms"],
        "peak_dbfs": a["peak_dbfs"],
        "table": table,
        "clips": clips,
        "blocklist": reachable_blocklist(gen.blocklist, playable),
    }


def export_web(spk: Speaker, out_dir: Path, top: int = 0) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in STALE:
        stale = out_dir / name
        if stale.is_dir():
            shutil.rmtree(stale)
        else:
            stale.unlink(missing_ok=True)
    data = generator_data(spk, out_dir / "clips.wav")
    (out_dir / "generator.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    matrix = transition_matrix(spk, top)
    (out_dir / "matrix.json").write_text(json.dumps(matrix, ensure_ascii=False, separators=(",", ":")) + "\n")
    log.info("web export: %d entries, %d clips, %d contexts, %d x %d matrix -> %s", len(data["clips"]),
             sum(len(v) for v in data["clips"].values()), len(data["table"]), len(matrix["rows"]),
             len(matrix["columns"]), out_dir)
    return [out_dir / "generator.json", out_dir / "clips.wav", out_dir / "matrix.json"]
