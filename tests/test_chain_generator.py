from collections import Counter

from mvs.blocklist import Blocklist
from mvs.chain import END, START, Chain, map_to_entries
from mvs.config import load_speaker
from mvs.generator import SentenceGenerator


def test_map_to_entries_longest_match_and_oov():
    entries = {"thank you", "thank", "very", "much", "you"}
    assert map_to_entries(["thank", "you", "so", "very", "much"], entries, 3) == ["thank you", "very", "much"]
    assert map_to_entries(["xyz", None, "you"], entries, 3) == ["you"]


def test_chain_counts_all_orders():
    c = Chain(2)
    c.add_sentence(["a", "b", "c"])
    c.add_sentence(["a", "b", "d"])
    assert c.options((START, START)) == Counter({"a": 2})
    assert c.options(("a", "b")) == Counter({"c": 1, "d": 1})
    assert c.options(("b",)) == Counter({"c": 1, "d": 1})
    assert c.options(("c",)) == Counter({END: 1})
    assert Chain.from_json(c.to_json()).table == c.table


def _clips(*tokens, **overrides):
    entries = {}
    for t in tokens:
        entries[t] = [{"path": f"clips/{t}/0.wav", "prev": None, "next": None, "final": False, "clip_score": 0.5}]
    entries.update(overrides)
    return {"entries": entries}


def test_backoff_to_order_one(project, tmp_path):
    spk = load_speaker("test-speaker", project)
    c = Chain(2)
    c.add_sentence(["x", "a", "b", "c"])
    c.add_sentence(["y", "z", "b", "d"])
    gen = SentenceGenerator(spk, chain=c, clips=_clips("x", "y", "z", "a", "b", "c", "d"), seed=1)
    # ("a","b") has one option (c): below backoff_min_options=2, so order 1 ("b",) gives c or d.
    seen = {gen._next(["a", "b"]) for _ in range(200)}
    assert seen == {"c", "d"}


def test_restart_on_dead_end_and_length_limits(project):
    spk = load_speaker("test-speaker", project)
    c = Chain(2)
    c.add_sentence(["alpha", "beta", "gamma", "delta"])
    c.add_sentence(["alpha", "beta"])  # too short: min_sentence_words is 3
    gen = SentenceGenerator(spk, chain=c, clips=_clips("alpha", "beta", "gamma", "delta"), seed=3)
    for _ in range(20):
        s = gen.next_sentence()
        assert 3 <= len(s.tokens) <= 20
        assert s.text == "alpha beta gamma delta"


def test_blocklisted_sentences_never_emitted(project, tmp_path):
    spk = load_speaker("test-speaker", project)
    (tmp_path / "bl.txt").write_text("vote\n")
    c = Chain(1)
    for _ in range(5):
        c.add_sentence(["we", "vote", "today"])
    c.add_sentence(["we", "win", "today"])
    gen = SentenceGenerator(spk, chain=c, clips=_clips("we", "vote", "win", "today"),
                            blocklist=Blocklist.load(tmp_path / "bl.txt"), seed=0)
    texts = {gen.next_sentence().text for _ in range(30)}
    assert texts == {"we win today"}
    assert gen.stats["blocked"] > 0


def test_tokens_without_clips_are_skipped(project):
    spk = load_speaker("test-speaker", project)
    c = Chain(1)
    c.add_sentence(["a", "b", "c"])
    c.add_sentence(["a", "nope", "c", "b"])
    clips = _clips("a", "b", "c", nope=[])
    gen = SentenceGenerator(spk, chain=c, clips=clips, seed=0)
    for _ in range(20):
        assert "nope" not in gen.next_sentence().tokens


def test_variant_prefers_matching_neighbour(project):
    spk = load_speaker("test-speaker", project)
    c = Chain(1)
    c.add_sentence(["big", "deal", "now"])
    variants = [
        {"path": "clips/big/0.wav", "prev": None, "next": "house", "final": False, "clip_score": 0.9},
        {"path": "clips/big/1.wav", "prev": None, "next": "deal", "final": False, "clip_score": 0.5},
    ]
    gen = SentenceGenerator(spk, chain=c, clips=_clips("deal", "now", big=variants), seed=0)
    picks = Counter(gen.pick_clips(["big", "deal", "now"])[0].name for _ in range(50))
    assert picks["1.wav"] > picks["0.wav"]
