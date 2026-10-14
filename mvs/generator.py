"""Stage 7: sample sentences from the chain, filter them through the blocklist, and pick a clip per token."""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .blocklist import Blocklist
from .chain import END, START, Chain, load_chain
from .clips import load_clips
from .config import Speaker


log = logging.getLogger("mvs.generator")


class GenerationError(RuntimeError):
    pass


@dataclass
class Sentence:
    text: str
    tokens: list[str]
    clips: list[Path]


class SentenceGenerator:
    """Iterating yields (text, list_of_clip_paths) forever."""

    def __init__(self, spk: Speaker, chain: Chain | None = None, clips: dict | None = None,
                 blocklist: Blocklist | None = None, seed: int | None = None):
        cc = spk["chain"]
        self.spk = spk
        self.chain = chain or load_chain(spk)
        self.clips = {t: recs for t, recs in (clips or load_clips(spk))["entries"].items() if recs}
        self.blocklist = blocklist or Blocklist.load(spk.blocklist_path)
        for category in self.blocklist.empty_categories():
            log.warning("blocklist category [%s] is empty; fill it before publishing", category)
        self.rng = random.Random(cc["seed"] if seed is None else seed)
        self.order = self.chain.order
        self.backoff_min = cc["backoff_min_options"]
        self.min_len, self.max_len = cc["min_sentence_words"], cc["max_sentence_words"]
        self.recent_sentences: deque[str] = deque(maxlen=cc["recent_sentences"])
        self.recent_clips: deque[str] = deque(maxlen=cc["recent_clips"])
        self.stats = {"sentences": 0, "restarts": 0, "blocked": 0, "repeats": 0, "length": 0}
        if not self.clips:
            raise GenerationError("no clips available: run 'mvs clips' first")

    def _next(self, context: list[str]) -> str | None:
        for n in range(self.order, 0, -1):
            options = {t: c for t, c in self.chain.options(tuple(context[-n:])).items()
                       if t == END or t in self.clips}
            if n > 1 and len(options) < self.backoff_min:
                continue  # too few choices: back off to a shorter context
            if options:
                tokens = list(options)
                return self.rng.choices(tokens, weights=[options[t] for t in tokens])[0]
        return None

    def sample_tokens(self) -> list[str] | None:
        context, out = [START] * self.order, []
        while True:
            token = self._next(context)
            if token is None:
                self.stats["restarts"] += 1
                return None
            if token == END:
                if len(out) < self.min_len:
                    self.stats["length"] += 1
                    return None
                return out
            out.append(token)
            if len(out) > self.max_len:
                self.stats["length"] += 1
                return None
            context = context[1:] + [token]

    def pick_clips(self, tokens: list[str]) -> list[Path]:
        """Prefer takes whose original neighbours match this sentence (natural joins); avoid recent takes."""
        paths = []
        for i, token in enumerate(tokens):
            nxt = tokens[i + 1].split()[0] if i + 1 < len(tokens) else None
            prv = tokens[i - 1].split()[-1] if i > 0 else None
            last = i + 1 == len(tokens)

            def rank(rec):
                s = rec["clip_score"] + self.rng.random() * 0.5
                if nxt is not None and rec["next"] == nxt:
                    s += 2.0
                if prv is not None and rec["prev"] == prv:
                    s += 1.0
                s += 1.0 if rec["final"] == last else -1.0  # falling pitch only at the end
                if rec["path"] in self.recent_clips:
                    s -= 3.0
                return s

            rec = max(self.clips[token], key=rank)
            self.recent_clips.append(rec["path"])
            paths.append(self.spk.work_dir / rec["path"])
        return paths

    def next_sentence(self, max_attempts: int = 5000) -> Sentence:
        for attempt in range(max_attempts):
            tokens = self.sample_tokens()
            if tokens is None:
                continue
            text = " ".join(tokens)
            if self.blocklist.check(text):
                self.stats["blocked"] += 1
                continue
            # A tiny chain may not have enough distinct sentences; after half the budget, allow a repeat.
            if text in self.recent_sentences and attempt < max_attempts // 2:
                self.stats["repeats"] += 1
                continue
            self.recent_sentences.append(text)
            self.stats["sentences"] += 1
            return Sentence(text, tokens, self.pick_clips(tokens))
        raise GenerationError(f"no acceptable sentence in {max_attempts} attempts ({self.stats})")

    def __iter__(self):
        while True:
            s = self.next_sentence()
            yield s.text, s.clips
