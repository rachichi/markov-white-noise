"""Stages 4-9 end to end on the synthetic speaker, plus purge."""

import json

import numpy as np
import pytest

from mvs import audio
from mvs.chain import build_chain
from mvs.clips import build_clips
from mvs.config import SafetyError, load_speaker
from mvs.deliver import preview, render
from mvs.generator import SentenceGenerator
from mvs.program import Program
from mvs.purge import find_source, purge
from mvs.tts import disclaimer_audio
from mvs.vocab import build_vocab


@pytest.fixture
def built(speaker):
    build_vocab(speaker)
    build_clips(speaker)
    build_chain(speaker)
    return speaker


def test_vocab_selects_function_and_signature(speaker):
    vocab = build_vocab(speaker)
    kinds = {e["text"]: e["kind"] for e in vocab["entries"]}
    assert kinds["a"] == "function"
    assert kinds.get("believe me") == "signature"  # a phrase
    assert sum(k == "signature" for k in kinds.values()) <= 12
    report = speaker.vocab_report.read_text()
    assert "usable aligned instances" in report and "hit the blocklist" in report


def test_clips_written_and_ranked(built):
    data = json.loads(built.clips_json.read_text())
    recs = data["entries"]["a"]
    assert 1 <= len(recs) <= 6
    for r in recs:
        x, sr = audio.read_wav(built.work_dir / r["path"])
        assert sr == 16000 and len(x) > 0
        assert audio.loudness(x, sr, gated=False) == pytest.approx(-16, abs=1.0)
    assert "clean instances" in built.clips_report.read_text()


def test_generated_text_has_no_blocked_words(built):
    gen = SentenceGenerator(built, seed=0)
    for _ in range(100):
        text = gen.next_sentence().text
        assert "vote" not in text.split() and "kill" not in text.split()


def test_program_disclaimer_schedule_and_gaps(built):
    sr = built["audio"]["sample_rate"]
    interval = built["output"]["disclaimer_every_minutes"] * 60 * sr
    disc = disclaimer_audio(built)
    program = Program(built, SentenceGenerator(built, seed=1), disc)
    starts, t, pieces = [], 0, []
    try:
        for seg in program.session():
            if seg.kind == "disclaimer":
                starts.append(t)
            pieces.append(seg.audio)
            t += len(seg.audio)
            if t > 3 * interval:
                break
    finally:
        program.close()
    # no silence over 1 s anywhere, including across sentence/disclaimer boundaries
    quiet = audio.frame_rms(np.concatenate(pieces), sr, 20) < 1e-3
    run = longest = 0
    for q in quiet:
        run = run + 1 if q else 0
        longest = max(longest, run)
    assert longest * 0.02 < 1.0
    assert starts[0] == 0
    assert len(starts) >= 3
    assert all(b - a <= interval for a, b in zip(starts, starts[1:]))


def test_preview_and_render_files_carry_disclosure(built, tmp_path):
    texts = preview(built, 3, tmp_path / "p.wav", seed=2)
    assert len(texts) == 3
    tags = audio.read_wav_info(tmp_path / "p.wav")
    assert "not real statements" in tags["title"] and "Markov" in tags["comment"]
    x, sr = audio.read_wav(tmp_path / "p.wav")
    disc = disclaimer_audio(built)
    assert np.allclose(x[:len(disc)], disc, atol=2e-4)  # opens with the disclaimer

    paths = render(built, hours=2 / 60, out_dir=tmp_path / "out", rss_base_url="https://example.org/mvs", seed=3)
    assert len(paths) == 2
    for p in paths:
        y, _ = audio.read_wav(p)
        assert 60 <= len(y) / sr < 75
        assert np.allclose(y[:len(disc)], disc, atol=2e-4)
        tags = audio.read_wav_info(p)
        assert all(tags[k] for k in ("title", "artist", "album", "comment"))
        assert "not real statements" in tags["artist"]
    assert (tmp_path / "out" / "feed.xml").exists()
    assert "Not real statements" in (tmp_path / "out" / "index.html").read_text()


def test_output_refused_when_disclosure_disabled(built, tmp_path):
    built.data["output"]["disclaimer_enabled"] = False
    with pytest.raises(SafetyError):
        preview(built, 1, tmp_path / "p.wav")
    built.data["output"]["disclaimer_enabled"] = True
    built.data["output"]["metadata_labels"] = False
    with pytest.raises(SafetyError):
        render(built, hours=0.01, out_dir=tmp_path)


def test_purge_removes_source_and_derived_clips(built):
    before = json.loads(built.clips_json.read_text())
    assert any(r["source_id"] == "speech-a" for recs in before["entries"].values() for r in recs)
    chain_before = built.chain_path.read_text()

    [(spk, src)] = find_source(built.root, "speech-a.wav")
    purge(spk, src)

    spk = load_speaker("test-speaker", built.root)
    after = json.loads(spk.clips_json.read_text())
    assert not any(r["source_id"] == "speech-a" for recs in after["entries"].values() for r in recs)
    assert all(r["source_id"] == "speech-b" for recs in after["entries"].values() for r in recs)
    for p in (spk.sources_dir / "speech-a.wav", spk.wav_dir / "speech-a.wav", spk.isolated_dir / "speech-a.json"):
        assert not p.exists()
    assert "speech-a" not in spk.sources_md.read_text()
    assert "speech-a" not in spk.sidecar_path.read_text()
    assert spk.chain_path.read_text() != chain_before
    assert "purged speech-a" in (spk.work_dir / "purge_log.txt").read_text()


def test_export_web(built, tmp_path):
    from mvs.web import export_web

    export_web(built, tmp_path / "web", n=3, seed=4)
    meta = json.loads((tmp_path / "web" / "preview.json").read_text())
    segs = meta["segments"]
    assert segs[0]["kind"] == "disclaimer" and segs[0]["start"] == 0
    assert sum(s["kind"] == "sentence" for s in segs) == 3
    assert all(a["end"] == pytest.approx(b["start"], abs=1e-3) for a, b in zip(segs, segs[1:]))
    x, sr = audio.read_wav(tmp_path / "web" / "preview.wav")
    assert len(x) / sr == pytest.approx(segs[-1]["end"], abs=0.01)

    m = json.loads((tmp_path / "web" / "matrix.json").read_text())
    assert m["rows"][0]["label"] == m["start"] and len(m["rows"]) == len(m["labels"]) + 1
    assert m["columns"][-1] == m["end"]
    for row in m["rows"]:
        if row["total"]:
            assert sum(p for _, p in row["cells"]) == pytest.approx(1.0, abs=0.01)  # top=0 keeps every column
