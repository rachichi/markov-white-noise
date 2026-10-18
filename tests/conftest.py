"""A synthetic speaker: tone-burst 'words' with known timestamps stand in for WhisperX output."""

from __future__ import annotations

import json
import random
import shutil
import zlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from mvs import audio, tts
from mvs.config import load_speaker
from mvs.sources import Manifest

REPO = Path(__file__).resolve().parents[1]
SR = 16000

FUNCTION = ["the", "we", "and", "a", "to", "is", "of", "it", "very", "that"]
SIGNATURE = ["tremendous", "believe me", "frankly", "beautiful", "folks", "incredible", "winning", "nobody"]
NOUNS = ["country", "people", "job", "deal", "economy", "thing"]
BLOCKED = ["vote", "kill"]


def make_sentences(rng: random.Random, n: int) -> list[list[str]]:
    out = []
    for _ in range(n):
        s = [rng.choice(["we", "folks", "frankly", "nobody"]), rng.choice(["have", "is", "love"]),
             "a", rng.choice(["tremendous", "beautiful", "incredible", "very"]), rng.choice(NOUNS)]
        if rng.random() < 0.5:
            s += ["and", "it", "is", rng.choice(["winning", "beautiful", "tremendous"])]
        if rng.random() < 0.3:
            s += ["believe", "me"]
        if rng.random() < 0.1:
            s.insert(1, rng.choice(BLOCKED))
        out.append(s)
    return out


def tone_word(word: str, dur: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    f0 = 120 + zlib.crc32(word.encode()) % 120
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    env = np.sin(np.pi * t / dur) ** 0.5
    return (0.25 * x * env / 2.3 + rng.normal(0, 1e-4, len(t))).astype(np.float32)


def write_source(spk, sid: str, sentences: list[list[str]], seed: int) -> None:
    rng = np.random.default_rng(seed)
    words, t = [], 0.5
    for s in sentences:
        for i, w in enumerate(s):
            dur = 0.18 + 0.02 * len(w)
            raw = w.capitalize() if i == 0 else w
            words.append({"raw": raw + ("." if i == len(s) - 1 else ""), "word": w, "start": round(t, 3),
                          "end": round(t + dur, 3), "score": 0.9, "sent_end": i == len(s) - 1, "keep": True})
            t += dur + (0.6 if i == len(s) - 1 else 0.06)
    x = np.zeros(int((t + 0.5) * SR), np.float32) + rng.normal(0, 1e-4, int((t + 0.5) * SR)).astype(np.float32)
    for w in words:
        seg = tone_word(w["word"], w["end"] - w["start"], rng)
        s0 = int(w["start"] * SR)
        x[s0:s0 + len(seg)] = seg
    audio.write_wav(spk.wav_dir / f"{sid}.wav", x, SR)
    audio.write_wav(spk.wav16k_dir / f"{sid}.wav", x, SR)
    spk.isolated_dir.mkdir(parents=True, exist_ok=True)
    (spk.isolated_dir / f"{sid}.json").write_text(json.dumps({"source_id": sid, "words": words}))
    spk.sources_dir.mkdir(parents=True, exist_ok=True)
    (spk.sources_dir / f"{sid}.wav").write_bytes((spk.wav_dir / f"{sid}.wav").read_bytes())


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "speakers").mkdir(parents=True)
    shutil.copytree(REPO / "config", root / "config")
    ref = root / "data" / "reference" / "general_english.txt"
    ref.parent.mkdir(parents=True)
    rng = random.Random(1)
    common = FUNCTION + NOUNS + ["have", "is", "love", "go", "see", "make", "good", "great", "time", "me", "believe"]
    ref.write_text("\n".join(" ".join(rng.choice(common) for _ in range(12)) + "." for _ in range(3000)))
    cfg = {
        "name": "Test Speaker",
        "slug": "test-speaker",
        "vocab": {"function_words": 10, "signature_words": 12, "min_count": 2, "min_clean_instances": 3,
                  "frequent_entries": 5},
        "chain": {"min_sentence_words": 3, "max_sentence_words": 20, "recent_sentences": 20},
        "audio": {"sample_rate": SR},
        "output": {"mode": "both", "format": "wav", "disclaimer_every_minutes": 0.5, "file_chunk_minutes": 1.0,
                   "tts": {"engine": "say"}},
    }
    (root / "speakers" / "test-speaker.yaml").write_text(yaml.safe_dump(cfg))
    return root


@pytest.fixture
def speaker(project: Path):
    spk = load_speaker("test-speaker", project)
    rng = random.Random(7)
    manifest = Manifest(spk)
    sidecar = {}
    for k, sid in enumerate(("speech-a", "speech-b")):
        write_source(spk, sid, make_sentences(rng, 60), seed=k)
        meta = {"title": f"Speech {k}", "url": f"https://example.org/{sid}", "date": "2024-01-01",
                "license": "test fixture"}
        sidecar[f"{sid}.wav"] = meta
        manifest.upsert({"id": sid, "file": f"{sid}.wav", **meta, "duration_s": 0, "added": "2024-01-01T00:00:00Z"})
    manifest.save()
    (spk.sidecar_path).write_text(yaml.safe_dump(sidecar))
    return spk


@pytest.fixture(autouse=True)
def fake_tts(monkeypatch):
    """A 2 s synthetic beep-voice instead of a real TTS engine."""

    def synth(text, cfg, out):
        t = np.arange(2 * 22050) / 22050
        audio.write_wav(out, (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 22050)
        return "fake"

    monkeypatch.setattr(tts, "synthesize", synth)
