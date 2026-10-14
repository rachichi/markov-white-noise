"""Stages 2-3: WhisperX transcription with word alignment, then pyannote speaker isolation."""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from . import audio
from .config import Speaker
from .ingest import ALIGN_SR
from .sources import Manifest
from .text import ends_sentence, normalize_word

log = logging.getLogger("mvs.transcribe")


def _device(setting: str) -> str:
    if setting != "auto":
        return setting
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_16k(spk: Speaker, sid: str) -> np.ndarray:
    x, sr = audio.read_wav(spk.wav16k_dir / f"{sid}.wav")
    return audio.resample(x, sr, ALIGN_SR)


def transcribe(spk: Speaker, force: bool = False) -> None:
    """Stage 2: write aligned/<id>.json with word timestamps and confidence scores."""
    todo = [s for s in Manifest(spk).sources if force or not (spk.aligned_dir / f"{s['id']}.json").exists()]
    if not todo:
        log.info("all sources already transcribed")
        return
    try:
        import whisperx
    except ImportError as e:
        raise RuntimeError("WhisperX is not installed: pip install -e '.[align]'") from e

    tc = spk["transcribe"]
    device = _device(tc["device"])
    compute = tc["compute_type"] if tc["compute_type"] != "auto" else ("float16" if device == "cuda" else "int8")
    model = whisperx.load_model(tc["model"], device, compute_type=compute, language=tc["language"])
    aligners: dict[str, tuple] = {}
    spk.aligned_dir.mkdir(parents=True, exist_ok=True)

    for src in todo:
        sid = src["id"]
        log.info("%s: transcribing", sid)
        wav = _load_16k(spk, sid)
        result = model.transcribe(wav, batch_size=tc["batch_size"], language=tc["language"])
        lang = result.get("language") or tc["language"]
        if lang not in aligners:
            aligners[lang] = whisperx.load_align_model(language_code=lang, device=device)
        align_model, meta = aligners[lang]
        aligned = whisperx.align(result["segments"], align_model, meta, wav, device, return_char_alignments=False)

        words = []
        for seg in aligned["segments"]:
            for w in seg.get("words", []):
                raw = w.get("word", "")
                words.append({
                    "raw": raw,
                    "word": normalize_word(raw),
                    "start": w.get("start"),
                    "end": w.get("end"),
                    "score": w.get("score"),
                    "sent_end": ends_sentence(raw),
                })
        if words:
            words[-1]["sent_end"] = True
        out = {"source_id": sid, "model": tc["model"], "language": lang,
               "duration_s": round(len(wav) / ALIGN_SR, 2), "words": words}
        (spk.aligned_dir / f"{sid}.json").write_text(json.dumps(out, ensure_ascii=False))
        log.info("%s: %d words", sid, len(words))


def _hf_load(cls, name: str, token: str | None):
    try:
        obj = cls.from_pretrained(name, token=token)
    except TypeError:  # pyannote.audio 3.x
        obj = cls.from_pretrained(name, use_auth_token=token)
    if obj is None:
        raise RuntimeError(f"could not load {name}: accept its terms on Hugging Face and set the token env var")
    return obj


def _waveform(x: np.ndarray):
    import torch

    return {"waveform": torch.from_numpy(np.ascontiguousarray(x, np.float32))[None], "sample_rate": ALIGN_SR}


def _diarize(pipeline, wav: np.ndarray) -> list[tuple[float, float, str]]:
    out = pipeline(_waveform(wav))
    ann = getattr(out, "speaker_diarization", out)  # pyannote 4 wraps the annotation
    return [(turn.start, turn.end, label) for turn, _, label in ann.itertracks(yield_label=True)]


def _overlaps(segs: list[tuple[float, float, str]]) -> list[tuple[float, float]]:
    events = sorted([(s, 1) for s, _, _ in segs] + [(e, -1) for _, e, _ in segs])
    regions, active, start = [], 0, None
    for t, d in events:
        active += d
        if active >= 2 and start is None:
            start = t
        elif active < 2 and start is not None:
            if t > start:
                regions.append((start, t))
            start = None
    return regions


class _Embedder:
    def __init__(self, iso: dict, token: str | None):
        from pyannote.audio import Inference, Model

        self.inference = Inference(_hf_load(Model, iso["embedding_model"], token), window="whole")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        e = np.asarray(self.inference(_waveform(x)), dtype=np.float64).reshape(-1)
        return e / (np.linalg.norm(e) + 1e-12)


def _pick_target(segs, wav, embedder, enrollment, threshold) -> str | None:
    totals: dict[str, float] = {}
    for s, e, label in segs:
        totals[label] = totals.get(label, 0.0) + (e - s)
    if not totals:
        return None
    if enrollment is None:
        return max(totals, key=totals.get)  # dominant speaker
    best, best_sim = None, -1.0
    for label in totals:
        pieces, total = [], 0.0
        for s, e, lab in sorted(segs, key=lambda t: t[1] - t[0], reverse=True):
            if lab == label and e - s >= 1.0 and total < 60:
                pieces.append(wav[int(s * ALIGN_SR):int(e * ALIGN_SR)])
                total += e - s
        if not pieces:
            continue
        sim = float(np.dot(embedder(np.concatenate(pieces)), enrollment))
        log.info("  speaker %s: similarity %.2f to enrollment", label, sim)
        if sim > best_sim:
            best, best_sim = label, sim
    return best if best_sim >= threshold else None


def isolate(spk: Speaker, force: bool = False) -> None:
    """Stage 3: write isolated/<id>.json, marking each word keep=True only if the target speaker said it alone."""
    iso = spk["isolation"]
    spk.isolated_dir.mkdir(parents=True, exist_ok=True)
    pipeline = embedder = enrollment = None
    token = os.environ.get(iso["hf_token_env"]) if iso["hf_token_env"] else None

    for src in Manifest(spk).sources:
        sid = src["id"]
        aligned_path, out_path = spk.aligned_dir / f"{sid}.json", spk.isolated_dir / f"{sid}.json"
        if not aligned_path.exists():
            log.warning("%s: not transcribed yet, skipping isolation", sid)
            continue
        if out_path.exists() and not force and out_path.stat().st_mtime >= aligned_path.stat().st_mtime:
            continue
        data = json.loads(aligned_path.read_text())
        words = data["words"]

        if not iso["enabled"] or src.get("solo"):
            for w in words:
                w["keep"] = w.get("start") is not None and w.get("end") is not None
            data["isolation"] = {"method": "solo"}
        else:
            if pipeline is None:
                try:
                    from pyannote.audio import Pipeline
                except ImportError as e:
                    raise RuntimeError("pyannote.audio is not installed: pip install -e '.[align]'") from e
                pipeline = _hf_load(Pipeline, iso["diarization_model"], token)
                try:
                    import torch

                    if torch.cuda.is_available():
                        pipeline.to(torch.device("cuda"))
                except ImportError:
                    pass
                if iso["enrollment"]:
                    embedder = _Embedder(iso, token)
                    ex, esr = audio.read_wav(spk.path(iso["enrollment"]))
                    enrollment = embedder(audio.resample(ex, esr, ALIGN_SR))
            log.info("%s: diarizing", sid)
            wav = _load_16k(spk, sid)
            segs = _diarize(pipeline, wav)
            target = _pick_target(segs, wav, embedder, enrollment, iso["enrollment_threshold"])
            overlaps = _overlaps(segs)
            if target is None:
                log.warning("%s: target speaker not found; no words kept", sid)
            for w in words:
                s, e = w.get("start"), w.get("end")
                if s is None or e is None or target is None:
                    w["keep"] = False
                    continue
                mid = (s + e) / 2
                labels = {lab for a, b, lab in segs if a <= mid <= b}
                w["speaker"] = next(iter(labels)) if len(labels) == 1 else None
                w["keep"] = labels == {target} and not any(a < e and s < b for a, b in overlaps)
            data["isolation"] = {"method": "enrollment" if enrollment is not None else "dominant",
                                 "target": target, "speakers": sorted({lab for _, _, lab in segs}),
                                 "overlap_s": round(sum(b - a for a, b in overlaps), 2)}
        kept = sum(1 for w in words if w["keep"])
        log.info("%s: kept %d of %d words", sid, kept, len(words))
        out_path.write_text(json.dumps(data, ensure_ascii=False))
