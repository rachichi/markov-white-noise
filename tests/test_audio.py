import numpy as np
import pytest

from mvs import audio


def sine(freq, seconds, sr, amp=1.0):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.mark.parametrize("sr", [48000, 24000, 16000])
def test_loudness_reference_tone(sr):
    # BS.1770: a full-scale 997 Hz sine in one channel reads -3.01 LUFS.
    assert audio.loudness(sine(997, 5, sr), sr) == pytest.approx(-3.01, abs=0.15)


def test_normalize_hits_target_and_respects_ceiling():
    sr = 24000
    x = sine(300, 3, sr, 0.05)
    y = audio.normalize(x, sr, -16.0, -1.0)
    assert audio.loudness(y, sr) == pytest.approx(-16.0, abs=0.3)
    assert np.max(np.abs(y)) <= 10 ** (-1 / 20) + 1e-6


def test_limiter_caps_peaks():
    sr = 24000
    x = sine(200, 1, sr, 0.3)
    x[1000:1010] = 0.99
    y = audio.limit(x, sr, -6.0)
    assert np.max(np.abs(y)) <= 10 ** (-6 / 20) + 1e-6


def test_crossfade_length_and_continuity():
    sr = 16000
    a, b = sine(220, 0.3, sr, 0.5), sine(220, 0.3, sr, 0.5)
    y = audio.concat_crossfade([audio.fade(a, sr, 5), audio.fade(b, sr, 5)], sr, 15)
    assert len(y) == len(a) + len(b) - int(sr * 0.015)
    assert np.max(np.abs(np.diff(y))) < 0.1  # no clicks at the join


def test_wav_roundtrip_with_info(tmp_path):
    x = sine(440, 0.5, 16000, 0.5)
    tags = {"title": "Markov Chain Remix: X (not real statements)", "comment": "never said, ünïcode"}
    audio.write_wav(tmp_path / "t.wav", x, 16000, tags)
    y, sr = audio.read_wav(tmp_path / "t.wav")
    assert sr == 16000 and np.allclose(x, y, atol=1e-4)
    assert audio.read_wav_info(tmp_path / "t.wav") == tags


def test_trim_silence_keeps_pad():
    sr = 16000
    x = np.concatenate([np.zeros(sr), sine(300, 0.2, sr), np.zeros(sr)])
    y = audio.trim_silence(x, sr, keep_ms=20)
    assert 0.2 * sr <= len(y) <= 0.26 * sr


def test_resample_length():
    x = sine(300, 1, 44100)
    assert len(audio.resample(x, 44100, 24000)) == 24000
