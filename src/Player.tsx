import { useEffect, useRef, useState } from "react";
import type { Segment } from "./types";

const BAR_WIDTH = 2;
const BAR_GAP = 4;

function sentenceCase(text: string) {
  return text.charAt(0).toUpperCase() + text.slice(1) + ".";
}

/** Resample the exported peaks to one value per drawn bar. */
function barPeaks(peaks: number[], bars: number) {
  const out = new Float32Array(bars);
  for (let i = 0; i < bars; i++) {
    const a = Math.floor((i * peaks.length) / bars);
    const b = Math.max(a + 1, Math.floor(((i + 1) * peaks.length) / bars));
    out[i] = Math.max(...peaks.slice(a, b));
  }
  return out;
}

export function Player({ src, segments, peaks, duration }: { src: string; segments: Segment[]; peaks: number[]; duration: number }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  // Smooth progress while playing; timeupdate alone only fires a few times a second.
  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    const tick = () => {
      setTime(audioRef.current?.currentTime ?? 0);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !width) return;
    const dpr = window.devicePixelRatio || 1;
    const height = canvas.clientHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const g = canvas.getContext("2d")!;
    g.scale(dpr, dpr);
    g.clearRect(0, 0, width, height);
    const bars = Math.floor(width / (BAR_WIDTH + BAR_GAP));
    const levels = peaks.length ? barPeaks(peaks, bars) : new Float32Array(bars).fill(0.04);
    const progress = duration ? time / duration : 0;
    const ink = getComputedStyle(canvas).color;
    for (let i = 0; i < bars; i++) {
      const h = Math.max(2, levels[i] * height);
      g.fillStyle = ink;
      g.globalAlpha = i / bars < progress ? 1 : 0.3;
      g.fillRect(i * (BAR_WIDTH + BAR_GAP), (height - h) / 2, BAR_WIDTH, h);
    }
  }, [peaks, duration, width, time]);

  const toggle = () => {
    const el = audioRef.current;
    if (!el) return;
    if (el.paused) void el.play();
    else el.pause();
  };

  const seek = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const el = audioRef.current;
    if (!el || !duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    el.currentTime = ((e.clientX - rect.left) / rect.width) * duration;
    setTime(el.currentTime);
  };

  const current = segments.find((s) => time >= s.start && time < s.end) ?? segments[0];

  return (
    <div className="player">
      <div className="player__controls">
        <button
          type="button"
          className="rl-button player__play"
          onClick={toggle}
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? (
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <rect x="3" y="2" width="3.5" height="12" />
              <rect x="9.5" y="2" width="3.5" height="12" />
            </svg>
          ) : (
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path d="M4 2 L14 8 L4 14 Z" />
            </svg>
          )}
        </button>
        <canvas ref={canvasRef} className="player__wave" onClick={seek} aria-label="Waveform" role="img" />
      </div>
      <audio
        ref={audioRef}
        src={src}
        preload="auto"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onTimeUpdate={(e) => !playing && setTime(e.currentTarget.currentTime)}
      />
      {current && (
        <p className={`player__text${current.kind === "disclaimer" ? " player__text--disclaimer" : ""}`} aria-live="polite">
          {current.kind === "disclaimer" ? current.text : sentenceCase(current.text)}
        </p>
      )}
    </div>
  );
}
