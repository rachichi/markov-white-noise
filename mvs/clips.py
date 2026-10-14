"""Stage 5: cut, score and keep the best clips for every vocabulary entry."""

from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from . import audio
from .config import Speaker
from .text import slugify
from .vocab import Instance, find_instances, load_vocab

log = logging.getLogger("mvs.clips")
EDGE_FADE_MS = 5.0
NOISE_WINDOW_S = 3.0


@dataclass
class Candidate:
    inst: Instance
    score: float
    snr_db: float
    noise_dbfs: float
    flatness: float


def spectral_flatness(x: np.ndarray) -> float:
    if len(x) < 64:
        return 1.0
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2 + 1e-12
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def score_instance(x: np.ndarray, sr: int, rms_frames: np.ndarray, inst: Instance, cfg: dict) -> Candidate | None:
    """Score one occurrence; None means it fails a hard check (duration, SNR, silence)."""
    n_words = len(inst.entry.split())
    dur_ms = (inst.end - inst.start) * 1000
    if not cfg["min_ms_per_word"] * n_words <= dur_ms <= cfg["max_ms_per_word"] * n_words:
        return None
    seg = x[int(inst.start * sr):int(inst.end * sr)]
    if len(seg) == 0:
        return None
    rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
    i0 = max(0, int((inst.start - NOISE_WINDOW_S) / 0.02))
    i1 = int((inst.end + NOISE_WINDOW_S) / 0.02) + 1
    noise = float(np.percentile(rms_frames[i0:i1], 10)) + 1e-6
    snr = 20 * np.log10(max(rms, 1e-9) / noise)
    if snr < cfg["min_snr_db"]:
        return None
    noise_dbfs = 20 * np.log10(noise)
    flat = spectral_flatness(seg)

    score = 0.5 * inst.score + 0.5 * float(np.clip((snr - cfg["min_snr_db"]) / 20, 0, 1))
    if noise_dbfs > -45:  # a constant bed under the pauses: music, crowd, applause, room noise
        score *= 0.7
    if flat > 0.4:  # noise-like spectrum: applause or heavy background
        score *= 0.6
    if np.max(np.abs(seg)) >= 0.98:  # clipped source
        score *= 0.5
    return Candidate(inst, round(score, 4), round(float(snr), 1), round(float(noise_dbfs), 1), round(flat, 3))


def select_variants(cands: list[Candidate], k: int) -> list[Candidate]:
    """Best clips first, spread over distinct original next words; one slot is kept for a sentence-final take."""
    mid = sorted((c for c in cands if not c.inst.final), key=lambda c: -c.score)
    final = sorted((c for c in cands if c.inst.final), key=lambda c: -c.score)
    reserve = 1 if final and k >= 3 else 0
    chosen, seen_next = [], set()
    for c in mid:
        if len(chosen) >= k - reserve:
            break
        if c.inst.next not in seen_next:
            chosen.append(c)
            seen_next.add(c.inst.next)
    for c in mid:
        if len(chosen) >= k - reserve:
            break
        if c not in chosen:
            chosen.append(c)
    for c in final:
        if len(chosen) >= k:
            break
        chosen.append(c)
    return chosen


def load_clips(spk: Speaker) -> dict:
    if not spk.clips_json.exists():
        raise RuntimeError(f"no clips yet: run 'mvs clips {spk.slug}'")
    return json.loads(spk.clips_json.read_text())


def build_clips(spk: Speaker) -> dict:
    vocab = load_vocab(spk)
    v, cc, ac = spk["vocab"], spk["clips"], spk["audio"]
    sr = ac["sample_rate"]
    entries = vocab["entries"]
    instances = find_instances(spk, {e["text"] for e in entries}, vocab["max_phrase_len"])
    frequent = {e["text"] for e in sorted(entries, key=lambda e: -e["count"])[:v["frequent_entries"]]}

    by_source: dict[str, list[Instance]] = defaultdict(list)
    for entry, insts in instances.items():
        # Cap the work for very common words, keeping the most confidently aligned.
        for inst in sorted(insts, key=lambda i: -i.score)[:cc["max_candidates"]]:
            by_source[inst.source_id].append(inst)

    candidates: dict[str, list[Candidate]] = defaultdict(list)
    for sid, insts in by_source.items():
        x, xsr = audio.read_wav(spk.wav_dir / f"{sid}.wav")
        x = audio.resample(x, xsr, sr)
        frames = audio.frame_rms(x, sr, 20.0)
        for inst in insts:
            c = score_instance(x, sr, frames, inst, cc)
            if c is not None and c.score >= cc["min_score"]:
                candidates[inst.entry].append(c)

    chosen: dict[str, list[Candidate]] = {}
    for e in entries:
        k = v["frequent_max_variants"] if e["text"] in frequent else v["max_variants_per_entry"]
        chosen[e["text"]] = select_variants(candidates.get(e["text"], []), max(k, v["max_variants_per_entry"]))

    if spk.clips_dir.exists():
        shutil.rmtree(spk.clips_dir)
    slugs: dict[str, str] = {}
    for text in chosen:
        slug = slugify(text)
        while slug in slugs.values():
            slug += "_"
        slugs[text] = slug

    records: dict[str, list[dict]] = {text: [] for text in chosen}
    to_cut: dict[str, list[tuple[str, int, Candidate]]] = defaultdict(list)
    for text, cands in chosen.items():
        for n, c in enumerate(cands):
            to_cut[c.inst.source_id].append((text, n, c))
    pad = ac["clip_pad_ms"] / 1000
    for sid, items in to_cut.items():
        x, xsr = audio.read_wav(spk.wav_dir / f"{sid}.wav")
        x = audio.resample(x, xsr, sr)
        for text, n, c in items:
            seg = x[max(0, int((c.inst.start - pad) * sr)):int((c.inst.end + pad) * sr)]
            seg = audio.trim_silence(seg, sr, keep_ms=ac["clip_pad_ms"])
            seg = audio.normalize(audio.fade(seg, sr, EDGE_FADE_MS), sr, ac["target_lufs"], ac["peak_dbfs"], gated=False)
            rel = f"clips/{slugs[text]}/{n}.wav"
            audio.write_wav(spk.work_dir / rel, seg, sr)
            rec = {"path": rel, **{k: val for k, val in asdict(c.inst).items() if k != "entry"},
                   "clip_score": c.score, "snr_db": c.snr_db, "noise_dbfs": c.noise_dbfs, "flatness": c.flatness}
            records[text].append(rec)
    for recs in records.values():
        recs.sort(key=lambda r: r["path"])

    clean = {e["text"]: len(candidates.get(e["text"], [])) for e in entries}
    data = {"speaker": spk.slug, "sample_rate": sr, "entries": records, "clean_counts": clean}
    spk.clips_json.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    _write_report(spk, entries, clean, records)
    return data


def _write_report(spk: Speaker, entries: list[dict], clean: dict, records: dict) -> None:
    need = spk["vocab"]["min_clean_instances"]
    ok = [e for e in entries if clean[e["text"]] >= need]
    pct = 100 * len(ok) / len(entries) if entries else 0.0
    empty = [e["text"] for e in entries if not records[e["text"]]]
    low = [e for e in entries if 0 < clean[e["text"]] < need]
    lines = [
        f"# Clip report: {spk.name}",
        "",
        f"- Entries: {len(entries)}",
        f"- Entries with at least {need} clean instances: {len(ok)} ({pct:.1f}%, target 90%)",
        f"- Clips written: {sum(len(r) for r in records.values())}",
        f"- Entries with no clip (never played): {len(empty)}",
        "",
        f"## Missing (no clean instance) ({len(empty)})", "",
    ]
    lines += [f"- `{t}`" for t in empty] or ["None."]
    lines += ["", f"## Below {need} clean instances ({len(low)})", ""]
    lines += [f"- `{e['text']}`: {clean[e['text']]}" for e in low] or ["None."]
    spk.clips_report.write_text("\n".join(lines) + "\n")
    level = logging.INFO if pct >= 90 else logging.WARNING
    log.log(level, "clips: %.1f%% of entries have >= %d clean instances; report at %s", pct, need, spk.clips_report)
