import { Fragment, useEffect, useRef, useState } from "react";
import { Generator, parseWav, type Rendered } from "./engine";
import type { GeneratorData } from "./types";

const BAR_WIDTH = 2;
const BAR_GAP = 4;
const PX_PER_S = 60; // scroll speed of the waveform
const LOOKAHEAD_S = 6; // audio scheduled ahead of the playhead; covers the right half of the waveform

interface Scheduled {
  item: Rendered;
  startAt: number; // AudioContext time
}

function lastIndex<T>(xs: T[], pred: (x: T) => boolean) {
  for (let i = xs.length - 1; i >= 0; i--) if (pred(xs[i])) return i;
  return -1;
}

/**
 * Endless programme, generated word by word in the browser: the disclaimer, then new sentences forever. The
 * disclaimer is repeated before any sentence that would end more than `disclaimer_every` seconds after the last
 * one started, matching mvs/program.py.
 */
class Programme {
  private started = false;
  private t = 0;
  private last = 0;
  private pending: Rendered | null = null;

  constructor(private gen: Generator, private every: number) {}

  next(): Rendered {
    const disclaimer = this.gen.disclaimer();
    if (!this.started) {
      this.started = true;
      this.t = disclaimer.duration;
      return disclaimer;
    }
    const sentence = this.pending ?? this.gen.nextSentence();
    if (!this.pending && this.t + sentence.duration - this.last > this.every) {
      this.pending = sentence;
      this.last = this.t;
      this.t += disclaimer.duration;
      return disclaimer;
    }
    this.pending = null;
    this.t += sentence.duration;
    return sentence;
  }
}

export function Player({ data, spriteUrl }: { data: GeneratorData; spriteUrl: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const gen = useRef<Generator | null>(null);
  const ctx = useRef<AudioContext | null>(null);
  const queue = useRef<Scheduled[]>([]);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [current, setCurrent] = useState<Scheduled | null>(null);
  const [time, setTime] = useState(0);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    let alive = true;
    fetch(spriteUrl)
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(r.statusText))))
      .then((buf) => {
        if (!alive) return;
        gen.current = new Generator(data, parseWav(buf).samples);
        setReady(true);
      })
      .catch(() => alive && setFailed(true));
    return () => {
      alive = false;
      void ctx.current?.close();
    };
  }, [data, spriteUrl]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  /** Scrolling waveform: bars move right to left past a playhead in the middle; played audio is solid. */
  const draw = (now: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const w = canvas.clientWidth;
    const height = canvas.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== w * dpr) {
      canvas.width = w * dpr;
      canvas.height = height * dpr;
    }
    const g = canvas.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, height);
    g.fillStyle = getComputedStyle(canvas).color;
    const step = BAR_WIDTH + BAR_GAP;
    const mid = w / 2;
    // Bars sit on a fixed time grid so they slide smoothly instead of shimmering.
    const first = Math.floor((now - mid / PX_PER_S) * (PX_PER_S / step));
    const last = Math.ceil((now + mid / PX_PER_S) * (PX_PER_S / step));
    let k = 0;
    for (let b = first; b <= last; b++) {
      const t0 = (b * step) / PX_PER_S;
      const t1 = t0 + step / PX_PER_S;
      const x = mid + (t0 - now) * PX_PER_S;
      while (k < queue.current.length - 1 && queue.current[k].startAt + queue.current[k].item.duration <= t0) k++;
      let level = 0;
      for (let j = k; j < queue.current.length && queue.current[j].startAt < t1; j++) {
        const { item, startAt } = queue.current[j];
        const n = item.peaks.length;
        const a = Math.max(0, Math.floor(((t0 - startAt) / item.duration) * n));
        const z = Math.min(n, Math.ceil(((t1 - startAt) / item.duration) * n));
        for (let i = a; i < z; i++) level = Math.max(level, item.peaks[i]);
      }
      const h = Math.max(2, level * height);
      g.globalAlpha = t0 < now && queue.current.length ? 1 : 0.3;
      g.fillRect(x, (height - h) / 2, BAR_WIDTH, h);
    }
    g.globalAlpha = 1;
  };

  useEffect(() => {
    draw(ctx.current?.currentTime ?? 0);
  }, [width]);

  /** Start the audio graph: schedule items back to back on the AudioContext clock, so joins are sample-exact. */
  const start = () => {
    const ac = new AudioContext();
    ctx.current = ac;
    const programme = new Programme(gen.current!, data.disclaimer_every);
    queue.current = [];
    let nextTime = ac.currentTime + 0.1;

    const schedule = () => {
      while (nextTime - ac.currentTime < LOOKAHEAD_S) {
        const item = programme.next();
        const buffer = ac.createBuffer(1, item.audio.length, data.sample_rate);
        buffer.copyToChannel(item.audio as Float32Array<ArrayBuffer>, 0);
        const source = ac.createBufferSource();
        source.buffer = buffer;
        source.connect(ac.destination);
        source.start(nextTime);
        queue.current.push({ item, startAt: nextTime });
        nextTime += item.duration;
      }
      const q = queue.current;
      while (q.length > 1 && q[1].startAt + q[1].item.duration < ac.currentTime - 10) q.shift();
    };

    let shown: Scheduled | null = null;
    const tick = () => {
      if (ac.state === "closed") return;
      const now = ac.currentTime;
      const playingNow = queue.current[lastIndex(queue.current, (s) => s.startAt <= now)] ?? null;
      if (playingNow !== shown) setCurrent((shown = playingNow));
      setTime(playingNow ? now - playingNow.startAt : 0);
      draw(now);
      requestAnimationFrame(tick);
    };

    schedule();
    const timer = window.setInterval(schedule, 250);
    ac.addEventListener("statechange", () => ac.state === "closed" && window.clearInterval(timer));
    requestAnimationFrame(tick);
    setPlaying(true);
  };

  const toggle = () => {
    const ac = ctx.current;
    if (!ac) start();
    else if (ac.state === "running") void ac.suspend().then(() => setPlaying(false));
    else void ac.resume().then(() => setPlaying(true));
  };

  const item = current?.item;
  const activeWord = item ? lastIndex(item.wordStarts, (s) => s <= time) : -1;

  return (
    <div className="player">
      <div className="player__controls">
        <button
          type="button"
          className="player__play"
          onClick={toggle}
          disabled={!ready}
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? (
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <rect x="3" y="2" width="3.5" height="12" />
              <rect x="9.5" y="2" width="3.5" height="12" />
            </svg>
          ) : (
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path d="M4.5 2 L14 8 L4.5 14 Z" />
            </svg>
          )}
        </button>
        <canvas ref={canvasRef} className="player__wave" aria-label="Waveform" role="img" />
      </div>
      <p
        className={`player__text${!item || item.kind === "disclaimer" ? " player__text--disclaimer" : ""}`}
        aria-live="polite"
      >
        {failed
          ? "Could not load the word clips."
          : !ready
            ? "Loading word clips…"
            : !item || item.kind === "disclaimer"
              ? data.disclaimer_text
              : item.tokens.map((token, i) => (
                  <Fragment key={i}>
                    <span className={i === activeWord ? "player__word player__word--active" : "player__word"}>
                      {i === 0 ? token.charAt(0).toUpperCase() + token.slice(1) : token}
                    </span>
                    {i + 1 === item.tokens.length ? "." : " "}
                  </Fragment>
                ))}
      </p>
    </div>
  );
}
