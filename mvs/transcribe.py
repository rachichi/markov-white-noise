"""Stages 2-3: WhisperX transcription with word alignment, then keep the timed words."""

from __future__ import annotations

import json
import logging

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


def isolate(spk: Speaker, force: bool = False) -> None:
    """Stage 3: write isolated/<id>.json, keeping every timed word. Sources are solo recordings of the speaker."""
    spk.isolated_dir.mkdir(parents=True, exist_ok=True)
    for src in Manifest(spk).sources:
        sid = src["id"]
        aligned_path, out_path = spk.aligned_dir / f"{sid}.json", spk.isolated_dir / f"{sid}.json"
        if not aligned_path.exists():
            log.warning("%s: not transcribed yet, skipping", sid)
            continue
        if out_path.exists() and not force and out_path.stat().st_mtime >= aligned_path.stat().st_mtime:
            continue
        data = json.loads(aligned_path.read_text())
        words = data["words"]
        for w in words:
            w["keep"] = w.get("start") is not None and w.get("end") is not None
        log.info("%s: kept %d of %d words", sid, sum(w["keep"] for w in words), len(words))
        out_path.write_text(json.dumps(data, ensure_ascii=False))
