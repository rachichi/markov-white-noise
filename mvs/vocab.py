"""Stage 4: pick function words (most frequent) and signature words/phrases (log-odds vs a reference corpus)."""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .blocklist import Blocklist
from .config import Speaker
from .sources import Manifest
from .text import iter_text_sentences

log = logging.getLogger("mvs.vocab")
SENTENCE_GAP_S = 1.5  # a pause this long also ends a sentence


def iter_sentences(spk: Speaker):
    """Yield (source_id, words) for the target speaker's speech, one sentence per item.

    Sentences end at final punctuation, long pauses, and wherever words were dropped
    (other speakers, overlap, missing timestamps).
    """
    for src in Manifest(spk).sources:
        path = spk.isolated_dir / f"{src['id']}.json"
        if not path.exists():
            log.warning("%s: not isolated yet, skipping", src["id"])
            continue
        cur, prev_end = [], None
        for w in json.loads(path.read_text())["words"]:
            if not w.get("keep"):
                if cur:
                    yield src["id"], cur
                cur, prev_end = [], None
                continue
            if cur and prev_end is not None and w["start"] - prev_end > SENTENCE_GAP_S:
                yield src["id"], cur
                cur = []
            cur.append(w)
            prev_end = w["end"]
            if w.get("sent_end"):
                yield src["id"], cur
                cur, prev_end = [], None
        if cur:
            yield src["id"], cur


def count_ngrams(sentences, max_n: int) -> list[Counter]:
    counts = [Counter() for _ in range(max_n + 1)]
    for words in sentences:
        for i in range(len(words)):
            for n in range(1, max_n + 1):
                gram = words[i:i + n]
                if len(gram) < n or None in gram:
                    break
                counts[n][" ".join(gram)] += 1
    return counts


def reference_counts(path: Path, max_n: int) -> list[Counter]:
    if not path.exists():
        raise RuntimeError(f"reference corpus {path} not found (see data/reference/README.md)")

    def sentences():
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield from iter_text_sentences(line)

    return count_ngrams(sentences(), max_n)


def log_odds_z(speaker: Counter, reference: Counter, prior_strength: float) -> dict[str, float]:
    """Log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn 2008), as z-scores."""
    ny, nr = sum(speaker.values()), sum(reference.values())
    if ny == 0:
        return {}
    total = ny + nr
    z = {}
    for gram, y in speaker.items():
        r = reference.get(gram, 0)
        a = prior_strength * (y + r) / total
        delta = math.log((y + a) / (ny + prior_strength - y - a)) - math.log((r + a) / (nr + prior_strength - r - a))
        z[gram] = delta / math.sqrt(1 / (y + a) + 1 / (r + a))
    return z


@dataclass
class Instance:
    entry: str
    source_id: str
    start: float
    end: float
    score: float
    prev: str | None
    next: str | None
    final: bool  # sentence-final (falling pitch)


def find_instances(spk: Speaker, entries: set[str], max_n: int) -> dict[str, list[Instance]]:
    """Every timed, confidently aligned occurrence of each entry in the target speaker's speech."""
    min_score = spk["transcribe"]["min_word_score"]
    out: dict[str, list[Instance]] = defaultdict(list)
    for sid, sent in iter_sentences(spk):
        words = [w.get("word") for w in sent]
        for i in range(len(sent)):
            for n in range(1, max_n + 1):
                gram = words[i:i + n]
                if len(gram) < n or None in gram:
                    break
                text = " ".join(gram)
                if text not in entries:
                    continue
                ws = sent[i:i + n]
                score = min((w.get("score") or 0.0) for w in ws)
                if score < min_score or ws[-1]["end"] <= ws[0]["start"]:
                    continue
                out[text].append(Instance(
                    entry=text, source_id=sid, start=ws[0]["start"], end=ws[-1]["end"], score=score,
                    prev=words[i - 1] if i > 0 else None,
                    next=words[i + n] if i + n < len(words) else None,
                    final=i + n == len(sent),
                ))
    return out


def load_vocab(spk: Speaker) -> dict:
    if not spk.vocab_path.exists():
        raise RuntimeError(f"no vocabulary yet: run 'mvs vocab {spk.slug}'")
    return json.loads(spk.vocab_path.read_text())


def build_vocab(spk: Speaker) -> dict:
    v = spk["vocab"]
    max_n = v["max_phrase_len"]
    sentences = [[w.get("word") for w in sent] for _, sent in iter_sentences(spk)]
    if not sentences:
        raise RuntimeError("no transcribed speech for the target speaker: run 'mvs transcribe' first")
    counts = count_ngrams(sentences, max_n)
    ref = reference_counts(spk.path(spk["reference_corpus"]), max_n)

    function = [w for w, _ in counts[1].most_common(v["function_words"])]
    function_set = set(function)
    scored = []
    for n in range(1, max_n + 1):
        for gram, z in log_odds_z(counts[n], ref[n], v["prior_strength"]).items():
            if counts[n][gram] >= v["min_count"] and gram not in function_set and z > 0:
                scored.append((z, gram, n))
    scored.sort(key=lambda t: (-t[0], t[1]))
    signature = scored[:v["signature_words"]]

    entries = [{"text": w, "kind": "function", "n": 1, "count": counts[1][w], "z": None} for w in function]
    entries += [{"text": g, "kind": "signature", "n": n, "count": counts[n][g], "z": round(z, 3)}
                for z, g, n in signature]
    instances = find_instances(spk, {e["text"] for e in entries}, max_n)
    for e in entries:
        e["usable"] = len(instances.get(e["text"], []))

    vocab = {"speaker": spk.slug, "max_phrase_len": max_n, "entries": entries}
    spk.work_dir.mkdir(parents=True, exist_ok=True)
    spk.vocab_path.write_text(json.dumps(vocab, indent=1, ensure_ascii=False) + "\n")
    _write_report(spk, vocab, sum(len(s) for s in sentences))
    return vocab


def _write_report(spk: Speaker, vocab: dict, total_words: int) -> None:
    need = spk["vocab"]["min_clean_instances"]
    entries = vocab["entries"]
    low = [e for e in entries if e["usable"] < need]
    covered = len(entries) - len(low)
    pct = 100 * covered / len(entries) if entries else 0.0
    blocked = []
    if spk.blocklist_path.exists():
        bl = Blocklist.load(spk.blocklist_path)
        blocked = [(e["text"], bl.check(e["text"])) for e in entries if bl.check(e["text"])]
    lines = [
        f"# Vocabulary report: {spk.name}",
        "",
        f"- Words of target-speaker speech: {total_words}",
        f"- Entries: {len(entries)} ({sum(e['kind'] == 'function' for e in entries)} function, "
        f"{sum(e['kind'] == 'signature' for e in entries)} signature)",
        f"- Entries with at least {need} usable aligned instances: {covered} ({pct:.1f}%, target 90%)",
        "",
        "Usable means timed and aligned with confidence; `mvs clips` reports how many are clean.",
        "",
        f"## Below {need} usable instances ({len(low)})",
        "",
    ]
    lines += [f"- `{e['text']}` ({e['kind']}): {e['usable']}" for e in low] or ["None."]
    lines += ["", f"## Entries that hit the blocklist ({len(blocked)})", "",
              "Sentences containing these are always discarded, so they never play.", ""]
    lines += [f"- `{t}` (matches `{m}`)" for t, m in blocked] or ["None."]
    lines += ["", "## Signature entries", "", "| Entry | Count | z |", "| ----- | ----- | - |"]
    lines += [f"| {e['text']} | {e['count']} | {e['z']} |" for e in entries if e["kind"] == "signature"]
    spk.vocab_report.write_text("\n".join(lines) + "\n")
    log.info("vocabulary: %d entries, %.1f%% with >= %d usable instances; report at %s",
             len(entries), pct, need, spk.vocab_report)
