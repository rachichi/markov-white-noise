"""Source manifest (sources.json, rendered as SOURCES.md) and the per-folder sources.yaml sidecar."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import Speaker
from .text import slugify

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm", ".mp4", ".mkv"}
REQUIRED_FIELDS = ("title", "url", "date", "license")


def source_id(filename: str) -> str:
    return slugify(Path(filename).stem).replace("_", "-")


def read_sidecar(spk: Speaker) -> dict[str, dict]:
    if not spk.sidecar_path.exists():
        return {}
    data = yaml.safe_load(spk.sidecar_path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{spk.sidecar_path} must map file names to metadata")
    return {str(k): (v or {}) for k, v in data.items()}


def write_sidecar(spk: Speaker, data: dict[str, dict]) -> None:
    spk.sources_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "# One entry per audio file in this folder. title, url, date and license are required.\n"
        "# Set solo: true for recordings with only the target speaker (skips diarization).\n"
    )
    spk.sidecar_path.write_text(header + yaml.safe_dump(data, sort_keys=True, allow_unicode=True))


def missing_fields(meta: dict) -> list[str]:
    return [f for f in REQUIRED_FIELDS if not str(meta.get(f) or "").strip()]


class Manifest:
    """Every ingested source file. sources.json is the record; SOURCES.md is regenerated from it."""

    def __init__(self, spk: Speaker):
        self.spk = spk
        self.sources: list[dict] = []
        if spk.manifest_path.exists():
            self.sources = json.loads(spk.manifest_path.read_text())["sources"]

    def get(self, sid: str) -> dict | None:
        return next((s for s in self.sources if s["id"] == sid), None)

    def find(self, query: str) -> dict | None:
        q = Path(query)
        for s in self.sources:
            if query in (s["id"], s["file"]) or q.name == s["file"]:
                return s
        return None

    def upsert(self, entry: dict) -> None:
        self.sources = [s for s in self.sources if s["id"] != entry["id"]] + [entry]
        self.sources.sort(key=lambda s: s["id"])

    def remove(self, sid: str) -> None:
        self.sources = [s for s in self.sources if s["id"] != sid]

    def save(self) -> None:
        self.spk.work_dir.mkdir(parents=True, exist_ok=True)
        self.spk.manifest_path.write_text(json.dumps({"sources": self.sources}, indent=2, ensure_ascii=False) + "\n")
        self.spk.sources_md.write_text(render_sources_md(self.spk, self.sources))


def _cell(value) -> str:
    return str(value if value not in (None, "") else "").replace("|", "\\|").replace("\n", " ")


def render_sources_md(spk: Speaker, sources: list[dict]) -> str:
    lines = [
        f"# Sources: {spk.name}",
        "",
        f"Every recording that clips for **{spk.stream_title}** were cut from. "
        "The output is a Markov chain remix; none of its word sequences were actually said.",
        "",
        "To remove a source and every clip derived from it: `mvs purge --source <id>`.",
        "",
        "| ID | Title | URL | Date | License / terms | Solo | File | Added |",
        "| -- | ----- | --- | ---- | --------------- | ---- | ---- | ----- |",
    ]
    for s in sources:
        lines.append("| " + " | ".join(_cell(s.get(k)) for k in
                     ("id", "title", "url", "date", "license", "solo", "file", "added")) + " |")
    if not sources:
        lines.append("| _none yet_ | | | | | | | |")
    return "\n".join(lines) + "\n"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
