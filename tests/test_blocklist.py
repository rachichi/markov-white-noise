from pathlib import Path

import pytest

from mvs.blocklist import Blocklist

REPO_BLOCKLIST = Path(__file__).resolve().parents[1] / "config" / "blocklist.txt"


@pytest.fixture(scope="module")
def bl():
    return Blocklist.load(REPO_BLOCKLIST)


@pytest.mark.parametrize("sentence", [
    "go and vote for the people",
    "the election is coming",
    "we were reelected tremendous",
    "register to vote folks",
    "please donate to the cause",
    "call your senator today",
    "join us in the streets",
    "they want to kill the deal",
    "we will bomb them",
    "storm the building",
    "my phone number is",
    "five five five one two three four",
    "the number is 5551234",
    "write to me at gmail dot com",
    "Election.",  # punctuation and case are normalized
])
def test_blocks(bl, sentence):
    assert bl.check(sentence) is not None


@pytest.mark.parametrize("sentence", [
    "we are devoted to the people",  # "vote" only matches whole words
    "a beautiful economy believe me",
    "nobody has seen anything like it",
    "the pollster was wrong",
    "it is one of the great things",
    "two thousand people came",
])
def test_allows(bl, sentence):
    assert bl.check(sentence) is None


def test_phrase_tokens(bl):
    assert bl.check(["believe me", "call your", "friends"]) == "call your"
    assert bl.check(["call", "the", "your"]) is None


def test_slurs_category_reported_empty(bl):
    assert "slurs" in bl.empty_categories()


def test_include_prefix_and_regex(tmp_path):
    (tmp_path / "extra.txt").write_text("forbidden phrase\n")
    (tmp_path / "main.txt").write_text("[x]\n@include extra.txt\nbad*\nre:\\bfoo\\s+bar\\b\n# comment\n")
    bl = Blocklist.load(tmp_path / "main.txt")
    assert bl.check("a forbidden phrase here")
    assert bl.check("badly done")
    assert bl.check("foo  bar")
    assert bl.check("forbidden") is None
    assert bl.check("foobar") is None


def test_include_cycle(tmp_path):
    (tmp_path / "a.txt").write_text("@include b.txt\nalpha\n")
    (tmp_path / "b.txt").write_text("@include a.txt\nbeta\n")
    bl = Blocklist.load(tmp_path / "a.txt")
    assert bl.check("beta") and bl.check("alpha")
