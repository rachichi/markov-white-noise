export interface Segment {
  kind: "disclaimer" | "sentence";
  text: string;
  start: number;
  end: number;
}

export interface PreviewMeta {
  name: string;
  title: string;
  disclaimer: string;
  description: string;
  duration: number;
  peaks: number[]; // waveform, 0-1 per slice
  segments: Segment[];
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
