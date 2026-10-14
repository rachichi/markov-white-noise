"""Audio primitives: WAV I/O with INFO tags, resampling, BS.1770 loudness, limiting, fades and crossfades."""

from __future__ import annotations

import struct
import wave
from math import gcd
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.signal import lfilter, resample_poly

# RIFF INFO ids used for the disclosure tags in WAV output.
INFO_IDS = {"title": b"INAM", "artist": b"IART", "album": b"IPRD", "comment": b"ICMT", "genre": b"IGNR"}


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read an integer PCM WAV as mono float32 in [-1, 1]."""
    try:
        with wave.open(str(path), "rb") as w:
            ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
    except wave.Error as e:
        raise ValueError(f"{path}: unsupported WAV ({e}); convert it to 16-bit PCM with ffmpeg") from e
    if width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = (np.frombuffer(raw, "<i4").astype(np.float64) / 2147483648.0).astype(np.float32)
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        i = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        i = np.where(i >= 1 << 23, i - (1 << 24), i)
        x = (i / float(1 << 23)).astype(np.float32)
    elif width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"{path}: unsupported sample width {width}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x.astype(np.float32), sr


def to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()


class WavWriter:
    """Incremental mono 16-bit WAV writer that appends a RIFF INFO chunk (title, artist, ...) on close."""

    def __init__(self, path: str | Path, sample_rate: int, info: dict[str, str] | None = None):
        self.path = Path(path)
        self.sr = sample_rate
        self.info = info or {}
        self.f = open(self.path, "wb")
        self.f.write(b"\0" * 44)
        self.data_bytes = 0

    def write(self, x: np.ndarray) -> None:
        b = to_pcm16(x)
        self.f.write(b)
        self.data_bytes += len(b)

    @property
    def seconds(self) -> float:
        return self.data_bytes / 2 / self.sr

    def close(self) -> None:
        if self.data_bytes % 2:
            self.f.write(b"\0")
        info = b""
        for key, value in self.info.items():
            data = value.encode("utf-8") + b"\0"
            if len(data) % 2:
                data += b"\0"
            info += INFO_IDS[key] + struct.pack("<I", len(data)) + data
        list_chunk = b"LIST" + struct.pack("<I", 4 + len(info)) + b"INFO" + info if info else b""
        self.f.write(list_chunk)
        riff_size = 4 + (8 + 16) + (8 + self.data_bytes + self.data_bytes % 2) + len(list_chunk)
        self.f.seek(0)
        self.f.write(b"RIFF" + struct.pack("<I", riff_size) + b"WAVE")
        self.f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, self.sr, self.sr * 2, 2, 16))
        self.f.write(b"data" + struct.pack("<I", self.data_bytes))
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_wav(path: str | Path, x: np.ndarray, sample_rate: int, info: dict[str, str] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with WavWriter(path, sample_rate, info) as w:
        w.write(x)


def read_wav_info(path: str | Path) -> dict[str, str]:
    """Return the RIFF INFO tags of a WAV file, keyed like INFO_IDS."""
    names = {v: k for k, v in INFO_IDS.items()}
    blob = Path(path).read_bytes()
    out, pos = {}, 12
    while pos + 8 <= len(blob):
        cid, size = blob[pos:pos + 4], struct.unpack("<I", blob[pos + 4:pos + 8])[0]
        body = blob[pos + 8:pos + 8 + size]
        if cid == b"LIST" and body[:4] == b"INFO":
            p = 4
            while p + 8 <= len(body):
                sid, ssize = body[p:p + 4], struct.unpack("<I", body[p + 4:p + 8])[0]
                if sid in names:
                    out[names[sid]] = body[p + 8:p + 8 + ssize].rstrip(b"\0").decode("utf-8")
                p += 8 + ssize + ssize % 2
        pos += 8 + size + size % 2
    return out


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x.astype(np.float32)
    g = gcd(sr_from, sr_to)
    return resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)


def _k_filters(sr: int):
    # BS.1770 K-weighting: high shelf (+4 dB at ~1.5 kHz) then high-pass at ~38 Hz.
    G, Q, fc = 4.0, 1 / np.sqrt(2), 1500.0
    A, w0 = 10 ** (G / 40), 2 * np.pi * fc / sr
    alpha, c, sA = np.sin(w0) / (2 * Q), np.cos(w0), np.sqrt(10 ** (G / 40))
    shelf = (
        [A * ((A + 1) + (A - 1) * c + 2 * sA * alpha), -2 * A * ((A - 1) + (A + 1) * c),
         A * ((A + 1) + (A - 1) * c - 2 * sA * alpha)],
        [(A + 1) - (A - 1) * c + 2 * sA * alpha, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - 2 * sA * alpha],
    )
    Q, fc = 0.5, 38.0
    w0 = 2 * np.pi * fc / sr
    alpha, c = np.sin(w0) / (2 * Q), np.cos(w0)
    highpass = ([(1 + c) / 2, -(1 + c), (1 + c) / 2], [1 + alpha, -2 * c, 1 - alpha])
    return shelf, highpass


def k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    (b1, a1), (b2, a2) = _k_filters(sr)
    return lfilter(b2, a2, lfilter(b1, a1, x.astype(np.float64)))


def loudness(x: np.ndarray, sr: int, gated: bool = True) -> float:
    """Integrated loudness in LUFS (BS.1770-4). Audio under 400 ms falls back to ungated mean square."""
    if len(x) == 0:
        return float("-inf")
    y = k_weight(x, sr)
    block, hop = int(0.4 * sr), int(0.1 * sr)
    if not gated or len(y) < block:
        ms = float(np.mean(y ** 2))
        return -0.691 + 10 * np.log10(ms) if ms > 0 else float("-inf")
    cs = np.concatenate([[0.0], np.cumsum(y ** 2)])
    starts = np.arange(1 + (len(y) - block) // hop) * hop
    ms = (cs[starts + block] - cs[starts]) / block
    ms = ms[-0.691 + 10 * np.log10(np.maximum(ms, 1e-20)) > -70]
    if len(ms) == 0:
        return float("-inf")
    rel = -0.691 + 10 * np.log10(ms.mean()) - 10
    ms = ms[-0.691 + 10 * np.log10(ms) > rel]
    return float(-0.691 + 10 * np.log10(ms.mean()))


def limit(x: np.ndarray, sr: int, ceiling_dbfs: float) -> np.ndarray:
    """Transparent-ish peak limiter: smoothed gain reduction around peaks above the ceiling."""
    ceiling = 10 ** (ceiling_dbfs / 20)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak <= ceiling:
        return x
    w = max(1, int(0.005 * sr))
    gain = np.minimum(1.0, ceiling / np.maximum(np.abs(x), 1e-9))
    gain = minimum_filter1d(gain, size=2 * w + 1)
    gain = np.convolve(gain, np.ones(w) / w, mode="same")
    return np.clip(x * gain, -ceiling, ceiling).astype(np.float32)


def normalize(x: np.ndarray, sr: int, target_lufs: float, ceiling_dbfs: float, gated: bool = True) -> np.ndarray:
    current = loudness(x, sr, gated=gated)
    if not np.isfinite(current):
        return x
    y = (x * 10 ** ((target_lufs - current) / 20)).astype(np.float32)
    return limit(y, sr, ceiling_dbfs)


def frame_rms(x: np.ndarray, sr: int, frame_ms: float = 20.0) -> np.ndarray:
    n = max(1, int(sr * frame_ms / 1000))
    usable = len(x) // n * n
    if usable == 0:
        return np.array([np.sqrt(np.mean(x ** 2))]) if len(x) else np.zeros(1)
    return np.sqrt(np.mean(x[:usable].reshape(-1, n).astype(np.float64) ** 2, axis=1))


def trim_silence(x: np.ndarray, sr: int, keep_ms: float, threshold_db: float = -35.0) -> np.ndarray:
    """Trim leading/trailing audio quieter than threshold_db below the loudest 10 ms frame, keeping keep_ms."""
    hop = max(1, int(sr * 0.01))
    rms = frame_rms(x, sr, 10.0)
    if len(rms) == 0 or rms.max() <= 0:
        return x
    loud = np.nonzero(rms >= rms.max() * 10 ** (threshold_db / 20))[0]
    keep = int(sr * keep_ms / 1000)
    start = max(0, loud[0] * hop - keep)
    end = min(len(x), (loud[-1] + 1) * hop + keep)
    return x[start:end]


def fade(x: np.ndarray, sr: int, ms: float) -> np.ndarray:
    n = min(int(sr * ms / 1000), len(x) // 2)
    if n <= 0:
        return x
    y = x.astype(np.float32).copy()
    ramp = np.sin(np.linspace(0, np.pi / 2, n, endpoint=False)) ** 2
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def concat_crossfade(parts: list[np.ndarray], sr: int, ms: float) -> np.ndarray:
    """Join clips with equal-power crossfades of ms milliseconds."""
    if not parts:
        return np.zeros(0, np.float32)
    n = int(sr * ms / 1000)
    pieces = [parts[0].astype(np.float32)]
    for p in parts[1:]:
        prev = pieces[-1]
        k = min(n, len(prev), len(p))
        if k == 0:
            pieces.append(p.astype(np.float32))
            continue
        t = np.linspace(0, np.pi / 2, k, endpoint=False)
        pieces[-1] = prev[:-k]
        pieces.append((prev[-k:] * np.cos(t) + p[:k] * np.sin(t)).astype(np.float32))
        pieces.append(p[k:].astype(np.float32))
    return np.concatenate(pieces)


def silence(sr: int, ms: float) -> np.ndarray:
    return np.zeros(int(sr * ms / 1000), np.float32)
