"""Speaker config loading, defaults, validation and the disclosure safety gate."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path

import yaml

DEFAULT_DISCLAIMER = (
    "This is an automated Markov chain remix of public speech clips. "
    "Nothing you hear was actually said in this order."
)
# Listeners join mid-stream and files get shared, so the disclaimer interval has a hard ceiling.
MAX_DISCLAIMER_MINUTES = 10
_BRANDING_WORDS = ("official", "campaign", "endorsed", "approved by", "paid for by")

DEFAULTS: dict = {
    "name": None,
    "slug": None,
    "work_dir": None,  # default: data/<slug>
    "sources_dir": None,  # default: data/<slug>/audio
    "reference_corpus": "data/reference/general_english.txt",
    "transcribe": {
        "model": "large-v3",
        "language": "en",
        "device": "auto",
        "compute_type": "auto",
        "batch_size": 8,
        "min_word_score": 0.5,
    },
    "isolation": {
        "enabled": True,
        "enrollment": None,  # path to a clean sample of the target speaker
        "enrollment_threshold": 0.5,
        "hf_token_env": "HF_TOKEN",
        "diarization_model": "pyannote/speaker-diarization-3.1",
        "embedding_model": "pyannote/wespeaker-voxceleb-resnet34-LM",
    },
    "vocab": {
        "function_words": 100,
        "signature_words": 200,
        "max_phrase_len": 3,
        "min_count": 5,
        "min_clean_instances": 3,
        "max_variants_per_entry": 3,
        "frequent_entries": 30,  # the most frequent entries get more variants
        "frequent_max_variants": 6,
        "prior_strength": 1000.0,
    },
    "clips": {
        "min_ms_per_word": 80,
        "max_ms_per_word": 1000,
        "min_snr_db": 10.0,
        "min_score": 0.4,
        "max_candidates": 300,
    },
    "chain": {
        "order": 2,
        "backoff_min_options": 2,
        "end_token_pause_ms": [400, 800],
        "min_sentence_words": 3,
        "max_sentence_words": 20,
        "recent_sentences": 500,
        "recent_clips": 40,
        "seed": None,
    },
    "audio": {
        "sample_rate": 24000,
        "target_lufs": -16.0,
        "crossfade_ms": 15,
        "clip_pad_ms": 20,
        "peak_dbfs": -1.0,
    },
    "output": {
        "mode": "stream",
        "disclaimer_enabled": True,
        "metadata_labels": True,
        "disclaimer_every_minutes": MAX_DISCLAIMER_MINUTES,
        "disclaimer_text": DEFAULT_DISCLAIMER,
        "tts": {"engine": "auto", "model": None, "voice": None},
        "blocklist": "config/blocklist.txt",
        "stream_title": None,  # default: "Markov Chain Remix: <name> (not real statements)"
        "stream_description": None,
        "stream_genre": "Generated audio (Markov chain remix)",
        "file_chunk_minutes": 60,
        "format": "mp3",
        "bitrate_kbps": 128,
        "buffer_sentences": 3,
        "health_port": 8090,
        "out_dir": None,  # default: <work_dir>/out
    },
}


class ConfigError(Exception):
    pass


class SafetyError(ConfigError):
    """Raised when disclosure settings would let unlabelled audio out."""


class Speaker:
    """A loaded speaker config plus the on-disk layout of its artifacts."""

    def __init__(self, data: dict, root: Path, config_path: Path):
        self.data = data
        self.root = root
        self.config_path = config_path
        self.name: str = data["name"]
        self.slug: str = data["slug"]

    def __getitem__(self, key):
        return self.data[key]

    def path(self, p) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.root / p

    @property
    def work_dir(self) -> Path:
        return self.path(self.data["work_dir"])

    @property
    def sources_dir(self) -> Path:
        return self.path(self.data["sources_dir"])

    @property
    def sidecar_path(self) -> Path:
        return self.sources_dir / "sources.yaml"

    @property
    def manifest_path(self) -> Path:
        return self.work_dir / "sources.json"

    @property
    def sources_md(self) -> Path:
        return self.work_dir / "SOURCES.md"

    @property
    def wav16k_dir(self) -> Path:
        return self.work_dir / "wav16k"

    @property
    def wav_dir(self) -> Path:
        return self.work_dir / "wav"

    @property
    def aligned_dir(self) -> Path:
        return self.work_dir / "aligned"

    @property
    def isolated_dir(self) -> Path:
        return self.work_dir / "isolated"

    @property
    def vocab_path(self) -> Path:
        return self.work_dir / "vocab.json"

    @property
    def vocab_report(self) -> Path:
        return self.work_dir / "vocab_report.md"

    @property
    def clips_dir(self) -> Path:
        return self.work_dir / "clips"

    @property
    def clips_json(self) -> Path:
        return self.work_dir / "clips.json"

    @property
    def clips_report(self) -> Path:
        return self.work_dir / "clips_report.md"

    @property
    def chain_path(self) -> Path:
        return self.work_dir / "chain.json"

    @property
    def cache_dir(self) -> Path:
        return self.work_dir / "cache"

    @property
    def logs_dir(self) -> Path:
        return self.work_dir / "logs"

    @property
    def out_dir(self) -> Path:
        return self.path(self.data["output"]["out_dir"] or self.work_dir / "out")

    @property
    def blocklist_path(self) -> Path:
        return self.path(self.data["output"]["blocklist"])

    @property
    def stream_title(self) -> str:
        return self.data["output"]["stream_title"]

    @property
    def disclaimer_text(self) -> str:
        return self.data["output"]["disclaimer_text"]

    @property
    def stream_description(self) -> str:
        return self.data["output"]["stream_description"]


def _merge(base: dict, over: dict, where: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if key not in base:
            raise ConfigError(f"unknown config key '{where}{key}'")
        if isinstance(base[key], dict) and value is not None:
            if not isinstance(value, dict):
                raise ConfigError(f"'{where}{key}' must be a mapping")
            out[key] = _merge(base[key], value, f"{where}{key}.")
        else:
            out[key] = value
    return out


def find_config(name_or_path: str, root: Path) -> Path:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml"):
        return p if p.is_absolute() else root / p
    return root / "speakers" / f"{name_or_path}.yaml"


def load_speaker(name_or_path: str, root: str | Path | None = None) -> Speaker:
    root = Path(root or os.getcwd()).resolve()
    path = find_config(name_or_path, root)
    if not path.exists():
        raise ConfigError(f"no speaker config at {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping")
    data = _merge(DEFAULTS, raw)
    _validate(data, path)
    return Speaker(data, root, path)


def _validate(d: dict, path: Path) -> None:
    def need(cond, msg):
        if not cond:
            raise ConfigError(f"{path.name}: {msg}")

    need(isinstance(d["name"], str) and d["name"].strip(), "'name' is required")
    need(isinstance(d["slug"], str) and re.fullmatch(r"[a-z0-9][a-z0-9-]*", d["slug"] or ""),
         "'slug' is required (lowercase letters, digits, dashes)")
    d["work_dir"] = d["work_dir"] or f"data/{d['slug']}"
    d["sources_dir"] = d["sources_dir"] or f"{d['work_dir']}/audio"

    v, c, a, o = d["vocab"], d["chain"], d["audio"], d["output"]
    for key in ("function_words", "signature_words", "max_phrase_len", "min_count",
                "min_clean_instances", "max_variants_per_entry", "frequent_max_variants"):
        need(isinstance(v[key], int) and v[key] >= 0, f"vocab.{key} must be a non-negative integer")
    need(1 <= v["max_phrase_len"] <= 4, "vocab.max_phrase_len must be 1-4")
    need(v["max_variants_per_entry"] >= 1, "vocab.max_variants_per_entry must be at least 1")
    need(isinstance(c["order"], int) and 1 <= c["order"] <= 4, "chain.order must be 1-4")
    need(isinstance(c["backoff_min_options"], int) and c["backoff_min_options"] >= 1,
         "chain.backoff_min_options must be at least 1")
    pause = c["end_token_pause_ms"]
    need(isinstance(pause, list) and len(pause) == 2 and 0 <= pause[0] <= pause[1] < 1000,
         "chain.end_token_pause_ms must be [min, max] below 1000 ms (no gaps over 1 s)")
    need(1 <= c["min_sentence_words"] <= c["max_sentence_words"],
         "chain.min_sentence_words must be between 1 and max_sentence_words")
    need(a["sample_rate"] in (16000, 22050, 24000, 32000, 44100, 48000), "audio.sample_rate is not supported")
    need(0 <= a["crossfade_ms"] <= 100, "audio.crossfade_ms must be 0-100")
    need(o["mode"] in ("stream", "files", "both"), "output.mode must be stream, files or both")
    need(o["format"] in ("mp3", "opus", "wav"), "output.format must be mp3, opus or wav")
    need(o["file_chunk_minutes"] > 0, "output.file_chunk_minutes must be positive")
    need(isinstance(o["buffer_sentences"], int) and o["buffer_sentences"] >= 3,
         "output.buffer_sentences must be at least 3")
    o["stream_title"] = o["stream_title"] or f"Markov Chain Remix: {d['name']} (not real statements)"
    o["stream_description"] = o["stream_description"] or (
        f"Generated audio. A Markov chain transition matrix built from transcripts of public speech "
        f"by {d['name']} picks each next word, and each word is a short real clip. "
        f"These sequences were never actually said. Not affiliated with or endorsed by {d['name']}."
    )


def require_disclosure(spk: Speaker) -> None:
    """Refuse to produce output unless disclaimers and metadata labels are on and well-formed."""
    o = spk["output"]
    if o["disclaimer_enabled"] is not True:
        raise SafetyError("output.disclaimer_enabled is off; refusing to produce audio without the spoken disclaimer")
    if o["metadata_labels"] is not True:
        raise SafetyError("output.metadata_labels is off; refusing to produce audio without disclosure metadata")
    every = o["disclaimer_every_minutes"]
    if not isinstance(every, (int, float)) or not 0 < every <= MAX_DISCLAIMER_MINUTES:
        raise SafetyError(f"output.disclaimer_every_minutes must be above 0 and at most {MAX_DISCLAIMER_MINUTES}")
    text = (o["disclaimer_text"] or "").lower()
    if "markov" not in text or "said" not in text:
        raise SafetyError("output.disclaimer_text must say it is a Markov chain remix and was not actually said")
    title = o["stream_title"].lower()
    if "markov" not in title or "not real" not in title:
        raise SafetyError("output.stream_title must contain 'Markov' and 'not real statements'")
    desc = o["stream_description"].lower()
    if "markov" not in desc or "never" not in desc:
        raise SafetyError("output.stream_description must say it is a Markov chain remix that was never said")
    for word in _BRANDING_WORDS:
        if word in title or word in desc.replace("not affiliated with or endorsed by", ""):
            raise SafetyError(f"stream title/description must not use branding or endorsement language ('{word}')")
    if not spk.blocklist_path.exists():
        raise SafetyError(f"blocklist {spk.blocklist_path} is missing")


def list_speaker_configs(root: Path) -> list[Path]:
    return sorted((root / "speakers").glob("*.yaml"))
