"""Stage 8: render sentences to audio and sequence them with scheduled disclaimers into a gapless programme."""

from __future__ import annotations

import queue
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from . import audio
from .config import Speaker, require_disclosure
from .generator import SentenceGenerator


@dataclass
class Segment:
    kind: str  # "disclaimer" or "sentence"
    text: str
    audio: np.ndarray


class Assembler:
    def __init__(self, spk: Speaker, seed: int | None = None):
        a = spk["audio"]
        self.sr = a["sample_rate"]
        self.crossfade_ms = a["crossfade_ms"]
        self.target, self.peak = a["target_lufs"], a["peak_dbfs"]
        self.pause_ms = spk["chain"]["end_token_pause_ms"]
        self.rng = random.Random(seed)
        self._cache: dict[Path, np.ndarray] = {}

    def clip(self, path: Path) -> np.ndarray:
        if path not in self._cache:
            x, sr = audio.read_wav(path)
            self._cache[path] = audio.resample(x, sr, self.sr)
        return self._cache[path]

    def render(self, clip_paths: list[Path]) -> np.ndarray:
        x = audio.concat_crossfade([self.clip(p) for p in clip_paths], self.sr, self.crossfade_ms)
        x = audio.normalize(x, self.sr, self.target, self.peak, gated=len(x) >= 0.4 * self.sr)
        return np.concatenate([x, audio.silence(self.sr, self.rng.uniform(*self.pause_ms))])


class _Prefetch:
    """Renders upcoming sentences on a background thread so playback never waits on generation."""

    def __init__(self, generator: SentenceGenerator, assembler: Assembler, size: int):
        self.q: queue.Queue = queue.Queue(maxsize=size)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(generator, assembler), daemon=True)
        self.thread.start()

    def _run(self, generator, assembler):
        try:
            for text, clips in generator:
                item = Segment("sentence", text, assembler.render(clips))
                while not self.stop.is_set():
                    try:
                        self.q.put(item, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if self.stop.is_set():
                    return
        except Exception as e:  # surfaced to the consumer
            self.q.put(e)

    def get(self) -> Segment:
        item = self.q.get()
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.stop.set()


class Program:
    """Endless sentence audio with a disclaimer at the start of every session and at most every N minutes."""

    def __init__(self, spk: Speaker, generator: SentenceGenerator, disclaimer: np.ndarray,
                 assembler: Assembler | None = None):
        require_disclosure(spk)
        if len(disclaimer) == 0:
            raise RuntimeError("disclaimer audio is empty")
        self.sr = spk["audio"]["sample_rate"]
        self.interval = int(spk["output"]["disclaimer_every_minutes"] * 60 * self.sr)
        self.disclaimer = Segment("disclaimer", spk.disclaimer_text, disclaimer)
        if len(disclaimer) >= self.interval:
            raise RuntimeError("disclaimer is longer than the disclaimer interval")
        self.prefetch = _Prefetch(generator, assembler or Assembler(spk), spk["output"]["buffer_sentences"])

    def session(self) -> Iterator[Segment]:
        """A fresh session (stream connection or file): starts with the disclaimer.

        A disclaimer is inserted before any sentence that would end more than `interval` after the last
        disclaimer started, so the spacing between disclaimer starts never exceeds the interval.
        """
        yield self.disclaimer
        t, last = len(self.disclaimer.audio), 0
        while True:
            seg = self.prefetch.get()
            if t + len(seg.audio) - last > self.interval:
                yield self.disclaimer
                last = t
                t += len(self.disclaimer.audio)
            yield seg
            t += len(seg.audio)

    def close(self):
        self.prefetch.close()
