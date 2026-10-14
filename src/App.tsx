import { useEffect, useState } from "react";
import { Label, PageShell, SiteFooter, SiteHeader } from "@rachichi/design";
import { Carousel } from "./Carousel";
import { MatrixLegend, MatrixView } from "./MatrixView";
import { Player } from "./Player";
import { dataUrl, speakers } from "./speakers";
import type { GeneratorData, Matrix } from "./types";

async function getJson<T>(url: string): Promise<T | null> {
  try {
    const r = await fetch(url);
    return r.ok ? ((await r.json()) as T) : null;
  } catch {
    return null;
  }
}

export default function App() {
  const [index, setIndex] = useState(speakers.findIndex((s) => s.slug === "omer"));
  const speaker = speakers[index];
  const [meta, setMeta] = useState<GeneratorData | null>(null);
  const [matrix, setMatrix] = useState<Matrix | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let current = true;
    setLoaded(false);
    Promise.all([
      getJson<GeneratorData>(dataUrl(speaker.slug, "generator.json")),
      getJson<Matrix>(dataUrl(speaker.slug, "matrix.json")),
    ]).then(([m, x]) => {
      if (!current) return;
      setMeta(m);
      setMatrix(x);
      setLoaded(true);
    });
    return () => {
      current = false;
    };
  }, [speaker.slug]);

  return (
    <PageShell
      header={<SiteHeader title="MARKOV WHITE NOISE" />}
      footer={<SiteFooter stack="python, whisperx, react, typescript" />}
    >
      <main className="page">
        <p className="intro">
          Professor Omar El Housni taught us about Markov Chains transition matrixes as a generative tool, so I
          created one using a recording of the lecture in which he taught us about Markov Chains transition
          matrixes. Now you can tune in to his lectures as you study for the prelim next Monday.
        </p>

        <section className="speaker">
          <Carousel items={speakers} index={index} onChange={setIndex} />
          <p className="speaker__name">{speaker.name}</p>
          {loaded && meta ? (
            <Player key={speaker.slug} data={meta} spriteUrl={dataUrl(speaker.slug, meta.sprite)} />
          ) : (
            loaded && (
              <p className="player__text player__text--disclaimer">
                No recordings of {speaker.name} yet. Add audio under <code>data/{speaker.slug}/audio/</code>, then run{" "}
                <code>mvs export-web {speaker.slug}</code>.
              </p>
            )
          )}
          <Label variant="note" as="p" className="speaker__notice">
            Markov chain remix of real clips · not real statements
          </Label>
        </section>

        {loaded && matrix && (
          <details className="transitions" key={speaker.slug}>
            <summary>
              <Label>Transition matrix</Label>
              <span className="transitions__size">
                {matrix.labels.length} × {matrix.labels.length}
              </span>
            </summary>
            <p className="transitions__caption">
              Probability of each next word or phrase given the current one, across {speaker.name}'s{" "}
              {matrix.labels.length} most frequent playable entries.
            </p>
            <MatrixLegend />
            <MatrixView matrix={matrix} />
          </details>
        )}
      </main>
    </PageShell>
  );
}
