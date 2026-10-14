/** What `mvs export-web` writes to generator.json: the chain, clip offsets into clips.wav, and settings. */
export interface GeneratorData {
  name: string;
  title: string;
  description: string;
  disclaimer_text: string;
  disclaimer_every: number; // seconds between disclaimer starts, at most
  sample_rate: number;
  sprite: string;
  disclaimer: [number, number]; // [offset, length] in samples
  order: number;
  start: string;
  end: string;
  backoff_min_options: number;
  min_sentence_words: number;
  max_sentence_words: number;
  recent_sentences: number;
  recent_clips: number;
  pause_ms: [number, number];
  crossfade_ms: number;
  peak_dbfs: number;
  table: Record<string, Record<string, number>>; // tab-joined context -> next token -> count
  clips: Record<string, [number, number, string | null, string | null, boolean, number][]>; // offset, length, prev, next, final, score
  blocklist: { terms: string[][]; prefixes: [string[], string][]; patterns: string[] };
}

export interface Matrix {
  order: number;
  labels: string[];
  kinds: ("function" | "signature")[];
  counts: number[];
  columns: string[];
  rows: { label: string; total: number; cells: [number, number][] }[];
  start: string;
  end: string;
}
