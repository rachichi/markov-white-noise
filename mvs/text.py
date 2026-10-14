"""Word normalization shared by every stage, so transcripts, vocabulary, chain and blocklist agree."""

from __future__ import annotations

import re

_INNER_DOT = re.compile(r"(?<=[a-z0-9])\.(?=[a-z0-9])")
_EDGE = re.compile(r"^[^a-z0-9]+|[^a-z0-9]+$")
_WORD = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")
_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")
_CLOSERS = "\"')]”’"


def normalize_word(raw: str) -> str | None:
    """Lowercase a transcript word and strip punctuation. Returns None for non-words ("--", "...")."""
    w = raw.strip().lower().replace("’", "'").replace("‘", "'")
    w = _INNER_DOT.sub("", w)  # "u.s." -> "us."
    w = _EDGE.sub("", w)
    return w if w and _WORD.fullmatch(w) else None


def ends_sentence(raw: str) -> bool:
    return raw.strip().rstrip(_CLOSERS).endswith((".", "?", "!"))


def split_words(text: str) -> list[str]:
    return [w for w in (normalize_word(t) for t in text.split()) if w]


def iter_text_sentences(text: str):
    """Yield normalized word lists, one per sentence, from free text."""
    for chunk in _SENT_SPLIT.split(text):
        words = split_words(chunk)
        if words:
            yield words


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "x"
