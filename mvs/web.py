"""Export files for the web page: a preview recording with sentence timings, and the transition matrix."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from . import audio

from .chain import END, START, load_chain
from .clips import load_clips
from .config import Speaker
from .deliver import preview
from .vocab import load_vocab

log = logging.getLogger("mvs.web")
PEAKS = 1000


def waveform_peaks(path: Path, n: int = PEAKS) -> list[float]:
    """Peak level per slice, 0-1, so the page can draw the waveform without decoding audio."""
    x, _ = audio.read_wav(path)
    edges = np.linspace(0, len(x), n + 1).astype(int)
    peaks = np.array([np.abs(x[a:b]).max() if b > a else 0.0 for a, b in zip(edges, edges[1:])])
    top = peaks.max() or 1.0
    return [round(float(p / top), 3) for p in peaks]


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


def export_web(spk: Speaker, out_dir: Path, n: int = 8, top: int = 0, seed: int | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    timeline: list[dict] = []
    preview(spk, n, out_dir / "preview.wav", seed=seed, timeline=timeline)
    meta = {
        "name": spk.name,
        "title": spk.stream_title,
        "disclaimer": spk.disclaimer_text,
        "description": spk.stream_description,
        "duration": timeline[-1]["end"] if timeline else 0,
        "peaks": waveform_peaks(out_dir / "preview.wav"),
        "segments": timeline,
    }
    (out_dir / "preview.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    matrix = transition_matrix(spk, top)
    (out_dir / "matrix.json").write_text(json.dumps(matrix, ensure_ascii=False, separators=(",", ":")) + "\n")
    log.info("web export: %d segments, %d x %d matrix -> %s", len(timeline), len(matrix["rows"]),
             len(matrix["columns"]), out_dir)
    return [out_dir / "preview.wav", out_dir / "preview.json", out_dir / "matrix.json"]
