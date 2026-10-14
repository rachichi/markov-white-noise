// In-browser port of mvs/generator.py and the sentence assembly in mvs/program.py: walks the Markov chain word by
// word, rejects blocklisted sentences, picks a recording of each word and crossfades them together.
import type { GeneratorData } from "./types";

export interface Rendered {
  kind: "disclaimer" | "sentence";
  text: string;
  tokens: string[];
  wordStarts: number[]; // seconds from the start of the item, one per token
  audio: Float32Array;
  duration: number;
  peaks: number[];
}

const PEAKS = 200;

/** Mono 16-bit PCM WAV to floats. Parsed by hand so it doesn't depend on decodeAudioData. */
export function parseWav(buf: ArrayBuffer): { sampleRate: number; samples: Float32Array } {
  const view = new DataView(buf);
  let pos = 12;
  let sampleRate = 0;
  while (pos + 8 <= view.byteLength) {
    const id = String.fromCharCode(...new Uint8Array(buf, pos, 4));
    const size = view.getUint32(pos + 4, true);
    if (id === "fmt ") sampleRate = view.getUint32(pos + 12, true);
    if (id === "data") {
      const pcm = new Int16Array(buf.slice(pos + 8, pos + 8 + size));
      const samples = new Float32Array(pcm.length);
      for (let i = 0; i < pcm.length; i++) samples[i] = pcm[i] / 32768;
      return { sampleRate, samples };
    }
    pos += 8 + size + (size % 2);
  }
  throw new Error("no data chunk in WAV");
}

function peaksOf(x: Float32Array, n = PEAKS) {
  const out = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    const a = Math.floor((i * x.length) / n);
    const b = Math.floor(((i + 1) * x.length) / n);
    for (let j = a; j < b; j++) out[i] = Math.max(out[i], Math.abs(x[j]));
  }
  const top = Math.max(...out) || 1;
  return out.map((p) => p / top);
}

function weighted(options: [string, number][]) {
  let r = Math.random() * options.reduce((s, [, c]) => s + c, 0);
  for (const [t, c] of options) if ((r -= c) < 0) return t;
  return options[options.length - 1][0];
}

class Recent<T> {
  private items: T[] = [];
  constructor(private max: number) {}
  add(x: T) {
    this.items.push(x);
    if (this.items.length > this.max) this.items.shift();
  }
  has(x: T) {
    return this.items.includes(x);
  }
}

export class Generator {
  private recentSentences: Recent<string>;
  private recentClips: Recent<number>;
  private patterns: RegExp[];

  constructor(private d: GeneratorData, private pcm: Float32Array) {
    this.recentSentences = new Recent(d.recent_sentences);
    this.recentClips = new Recent(d.recent_clips);
    this.patterns = d.blocklist.patterns.map((p) => new RegExp(p, "i"));
  }

  private next(context: string[]): string | null {
    for (let n = this.d.order; n >= 1; n--) {
      const options = Object.entries(this.d.table[context.slice(-n).join("\t")] ?? {});
      if (n > 1 && options.length < this.d.backoff_min_options) continue; // too few choices: shorter context
      if (options.length) return weighted(options);
    }
    return null;
  }

  private sampleTokens(): string[] | null {
    const { order, start, end, min_sentence_words, max_sentence_words } = this.d;
    let context = Array<string>(order).fill(start);
    const out: string[] = [];
    for (;;) {
      const token = this.next(context);
      if (token === null) return null;
      if (token === end) return out.length >= min_sentence_words ? out : null;
      out.push(token);
      if (out.length > max_sentence_words) return null;
      context = [...context.slice(1), token];
    }
  }

  blocked(text: string): boolean {
    const words = text.split(" ");
    const { terms, prefixes } = this.d.blocklist;
    for (let i = 0; i < words.length; i++) {
      for (const term of terms) if (term.every((w, k) => words[i + k] === w)) return true;
      for (const [head, prefix] of prefixes) {
        const j = i + head.length;
        if (j < words.length && head.every((w, k) => words[i + k] === w) && words[j].startsWith(prefix)) return true;
      }
    }
    return this.patterns.some((p) => p.test(text));
  }

  /** Prefer takes whose original neighbours match this sentence (natural joins); avoid recent takes. */
  private pickClips(tokens: string[]): Float32Array[] {
    return tokens.map((token, i) => {
      const nxt = i + 1 < tokens.length ? tokens[i + 1].split(" ")[0] : null;
      const prv = i > 0 ? tokens[i - 1].split(" ").at(-1)! : null;
      const last = i + 1 === tokens.length;
      let best = this.d.clips[token][0];
      let bestScore = -Infinity;
      for (const clip of this.d.clips[token]) {
        const [offset, , prev, next, final, score] = clip;
        let s = score + Math.random() * 0.5;
        if (nxt !== null && next === nxt) s += 2;
        if (prv !== null && prev === prv) s += 1;
        s += final === last ? 1 : -1; // falling pitch only at the end
        if (this.recentClips.has(offset)) s -= 3;
        if (s > bestScore) [best, bestScore] = [clip, s];
      }
      this.recentClips.add(best[0]);
      return this.pcm.subarray(best[0], best[0] + best[1]);
    });
  }

  nextSentence(maxAttempts = 5000): Rendered {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const tokens = this.sampleTokens();
      if (!tokens) continue;
      const text = tokens.join(" ");
      if (this.blocked(text)) continue;
      // A tiny chain may not have enough distinct sentences; after half the budget, allow a repeat.
      if (this.recentSentences.has(text) && attempt < maxAttempts / 2) continue;
      this.recentSentences.add(text);
      return this.render(tokens, this.pickClips(tokens));
    }
    throw new Error("no acceptable sentence");
  }

  disclaimer(): Rendered {
    const [offset, length] = this.d.disclaimer;
    const audio = this.pcm.subarray(offset, offset + length);
    const text = this.d.disclaimer_text;
    return { kind: "disclaimer", text, tokens: [text], wordStarts: [0], audio, duration: length / this.d.sample_rate, peaks: peaksOf(audio) };
  }

  /** Equal-power crossfades between clips, a peak ceiling, then a pause before the next sentence. */
  private render(tokens: string[], parts: Float32Array[]): Rendered {
    const sr = this.d.sample_rate;
    const fade = Math.floor((sr * this.d.crossfade_ms) / 1000);
    const [pmin, pmax] = this.d.pause_ms;
    const pause = Math.floor((sr * (pmin + Math.random() * (pmax - pmin))) / 1000);
    const total = parts.reduce((s, p) => s + p.length, 0) + pause;
    const out = new Float32Array(total);
    const wordStarts: number[] = [];
    let pos = 0;
    parts.forEach((p, i) => {
      const k = i === 0 ? 0 : Math.min(fade, pos, p.length);
      const at = pos - k;
      wordStarts.push(at / sr);
      for (let j = 0; j < p.length; j++) {
        if (j < k) {
          const t = ((j / k) * Math.PI) / 2;
          out[at + j] = out[at + j] * Math.cos(t) + p[j] * Math.sin(t);
        } else out[at + j] = p[j];
      }
      pos = at + p.length;
    });
    const audio = out.subarray(0, pos + pause);
    let peak = 0;
    for (let i = 0; i < pos; i++) peak = Math.max(peak, Math.abs(audio[i]));
    const ceiling = 10 ** (this.d.peak_dbfs / 20);
    if (peak > ceiling) for (let i = 0; i < pos; i++) audio[i] *= ceiling / peak;
    return { kind: "sentence", text: tokens.join(" "), tokens, wordStarts, audio, duration: audio.length / sr, peaks: peaksOf(audio) };
  }
}
