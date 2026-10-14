"""The spoken disclaimer, read by a generic local TTS voice (never the speaker's), rendered once and cached."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import audio
from .config import Speaker

log = logging.getLogger("mvs.tts")
AUTO_ORDER = ("piper", "kokoro", "espeak-ng", "say")
TAIL_MS = 300  # silence after the disclaimer; the preceding sentence already ends with a pause (gaps stay < 1 s)


def _piper(text: str, cfg: dict, out: Path) -> bool:
    if not cfg.get("model"):
        return False  # piper needs a voice model (.onnx) path
    exe = shutil.which("piper")
    cmd = [exe] if exe else ["python3", "-m", "piper"]
    subprocess.run(cmd + ["--model", cfg["model"], "--output_file", str(out)], input=text.encode(),
                   check=True, capture_output=True)
    return True


def _kokoro(text: str, cfg: dict, out: Path) -> bool:
    try:
        from kokoro import KPipeline
    except ImportError:
        return False
    pipe = KPipeline(lang_code="a")
    chunks = [np.asarray(a, np.float32) for _, _, a in pipe(text, voice=cfg.get("voice") or "af_heart")]
    audio.write_wav(out, np.concatenate(chunks), 24000)
    return True


def _espeak(text: str, cfg: dict, out: Path) -> bool:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        return False
    cmd = [exe, "-w", str(out)] + (["-v", cfg["voice"]] if cfg.get("voice") else []) + [text]
    subprocess.run(cmd, check=True, capture_output=True)
    return True


def _say(text: str, cfg: dict, out: Path) -> bool:
    if not shutil.which("say"):
        return False
    cmd = ["say", "--file-format=WAVE", "--data-format=LEI16@22050", "-o", str(out)]
    subprocess.run(cmd + (["-v", cfg["voice"]] if cfg.get("voice") else []) + [text], check=True, capture_output=True)
    return True


ENGINES = {"piper": _piper, "kokoro": _kokoro, "espeak-ng": _espeak, "say": _say}


def synthesize(text: str, cfg: dict, out: Path) -> str:
    """Write text as speech to out (WAV). Returns the engine used."""
    names = AUTO_ORDER if cfg.get("engine", "auto") == "auto" else (cfg["engine"],)
    errors = []
    for name in names:
        if name not in ENGINES:
            raise ValueError(f"unknown TTS engine '{name}' (choose from {', '.join(ENGINES)})")
        try:
            if ENGINES[name](text, cfg, out):
                return name
            errors.append(f"{name}: not available")
        except (OSError, subprocess.CalledProcessError) as e:
            errors.append(f"{name}: {e}")
    raise RuntimeError("no TTS engine could read the disclaimer, so output cannot start. Tried: " + "; ".join(errors))


def disclaimer_audio(spk: Speaker) -> np.ndarray:
    """The disclaimer at the output sample rate and target loudness, followed by a short silence."""
    o, a = spk["output"], spk["audio"]
    key = hashlib.sha1(json.dumps([o["disclaimer_text"], o["tts"], a["sample_rate"], a["target_lufs"]],
                                  sort_keys=True).encode()).hexdigest()[:12]
    cached = spk.cache_dir / f"disclaimer-{key}.wav"
    if cached.exists():
        return audio.read_wav(cached)[0]
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "tts.wav"
        engine = synthesize(o["disclaimer_text"], o["tts"], raw)
        x, sr = audio.read_wav(raw)
    sr_out = a["sample_rate"]
    x = audio.trim_silence(audio.resample(x, sr, sr_out), sr_out, keep_ms=50)
    x = audio.normalize(audio.fade(x, sr_out, 10), sr_out, a["target_lufs"], a["peak_dbfs"])
    x = np.concatenate([x, audio.silence(sr_out, TAIL_MS)])
    audio.write_wav(cached, x, sr_out)
    log.info("disclaimer rendered with %s (%.1f s), cached at %s", engine, len(x) / sr_out, cached)
    return x
