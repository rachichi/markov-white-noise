"""Stage 6: word-level Markov chain over vocabulary entries, with counts for every context length up to order."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict

from .config import Speaker
from .vocab import iter_sentences, load_vocab

log = logging.getLogger("mvs.chain")
START, END = "<s>", "</s>"


class Chain:
    def __init__(self, order: int):
        self.order = order
        self.table: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        self.sentences = 0

    def add_sentence(self, tokens: list[str]) -> None:
        padded = [START] * self.order + tokens + [END]
        for j in range(self.order, len(padded)):
            for n in range(1, self.order + 1):
                self.table[tuple(padded[j - n:j])][padded[j]] += 1
        self.sentences += 1

    def options(self, context: tuple[str, ...]) -> Counter:
        return self.table.get(tuple(context), Counter())

    def to_json(self) -> dict:
        return {"order": self.order, "sentences": self.sentences,
                "table": {"\t".join(ctx): dict(nxt) for ctx, nxt in self.table.items()}}

    @classmethod
    def from_json(cls, data: dict) -> "Chain":
        chain = cls(data["order"])
        chain.sentences = data.get("sentences", 0)
        for ctx, nxt in data["table"].items():
            chain.table[tuple(ctx.split("\t"))] = Counter(nxt)
        return chain


def map_to_entries(words: list[str | None], entries: set[str], max_n: int) -> list[str]:
    """Greedy longest match of words onto vocabulary entries. Out-of-vocabulary words are collapsed (skipped)."""
    out, i = [], 0
    while i < len(words):
        for n in range(min(max_n, len(words) - i), 0, -1):
            gram = words[i:i + n]
            if None not in gram and " ".join(gram) in entries:
                out.append(" ".join(gram))
                i += n
                break
        else:
            i += 1
    return out


def build_chain(spk: Speaker) -> Chain:
    """Built from the full transcripts, not just the clipped instances."""
    vocab = load_vocab(spk)
    entries = {e["text"] for e in vocab["entries"]}
    chain = Chain(spk["chain"]["order"])
    for _, sent in iter_sentences(spk):
        tokens = map_to_entries([w.get("word") for w in sent], entries, vocab["max_phrase_len"])
        if tokens:
            chain.add_sentence(tokens)
    spk.chain_path.write_text(json.dumps(chain.to_json(), ensure_ascii=False) + "\n")
    log.info("chain: %d sentences, %d contexts, %d transitions", chain.sentences, len(chain.table),
             sum(len(c) for c in chain.table.values()))
    return chain


def load_chain(spk: Speaker) -> Chain:
    if not spk.chain_path.exists():
        raise RuntimeError(f"no chain yet: run 'mvs chain {spk.slug}'")
    return Chain.from_json(json.loads(spk.chain_path.read_text()))
