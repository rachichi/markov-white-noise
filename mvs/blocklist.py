"""Sentence blocklist: whole-word terms and phrases, trailing-* prefixes, regexes, @include and [category] headers."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from .text import normalize_word


class Blocklist:
    def __init__(self):
        self.terms: dict[str, list[tuple[str, ...]]] = defaultdict(list)  # first word -> terms
        self.prefixes: list[tuple[tuple[str, ...], str]] = []  # (leading words, prefix of last word)
        self.patterns: list[re.Pattern] = []
        self.category_sizes: dict[str, int] = defaultdict(int)

    @classmethod
    def load(cls, path: str | Path) -> "Blocklist":
        bl = cls()
        bl._load(Path(path), set(), "uncategorized")
        return bl

    def _load(self, path: Path, seen: set, category: str) -> None:
        path = path.resolve()
        if path in seen:
            return
        seen.add(path)
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            line = line.split(" #", 1)[0].strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                category = line[1:-1].strip().lower()
                self.category_sizes.setdefault(category, 0)
            elif line.startswith("@include "):
                self._load(path.parent / line[len("@include "):].strip(), seen, category)
            elif line.startswith("re:"):
                try:
                    self.patterns.append(re.compile(line[3:].strip(), re.IGNORECASE))
                except re.error as e:
                    raise ValueError(f"{path}:{lineno}: bad regex: {e}") from e
                self.category_sizes[category] += 1
            else:
                self.add(line)
                self.category_sizes[category] += 1

    def add(self, term: str) -> None:
        prefix = term.endswith("*")
        words = [normalize_word(w) for w in term.rstrip("*").split()]
        if not words or None in words:
            raise ValueError(f"blocklist term '{term}' has no usable words")
        if prefix:
            self.prefixes.append((tuple(words[:-1]), words[-1]))
        else:
            self.terms[words[0]].append(tuple(words))

    def check(self, sentence: str | list[str]) -> str | None:
        """Return the first blocked term found in the sentence, or None. Tokens may be multi-word phrases."""
        text = sentence if isinstance(sentence, str) else " ".join(sentence)
        words = [w for w in (normalize_word(t) for t in text.split()) if w]
        for i, w in enumerate(words):
            for term in self.terms.get(w, ()):
                if tuple(words[i:i + len(term)]) == term:
                    return " ".join(term)
            for head, prefix in self.prefixes:
                j = i + len(head)
                if j < len(words) and tuple(words[i:j]) == head and words[j].startswith(prefix):
                    return " ".join(head + (prefix + "*",))
        joined = " ".join(words)
        for p in self.patterns:
            if p.search(joined):
                return f"re:{p.pattern}"
        return None

    def empty_categories(self) -> list[str]:
        return [c for c, n in self.category_sizes.items() if n == 0]
