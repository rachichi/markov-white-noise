import { useEffect, useState } from "react";
import { Label, PageShell, SiteFooter, SiteHeader } from "@rachichi/design";
import { MatrixView } from "./MatrixView";
import { Player } from "./Player";
import { dataUrl, speakers } from "./speakers";
import type { Matrix, PreviewMeta } from "./types";

const speaker = speakers[0];

async function getJson<T>(url: string): Promise<T | null> {
  try {
    const r = await fetch(url);
    return r.ok ? ((await r.json()) as T) : null;
  } catch {
    return null;
  }
}

export default function App() {
  const [meta, setMeta] = useState<PreviewMeta | null>(null);
  const [matrix, setMatrix] = useState<Matrix | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    Promise.all([
      getJson<PreviewMeta>(dataUrl(speaker.slug, "preview.json")),
      getJson<Matrix>(dataUrl(speaker.slug, "matrix.json")),
    ]).then(([m, x]) => {
      setMeta(m);
      setMatrix(x);
      setLoaded(true);
    });
  }, []);

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
          <img className="speaker__portrait" src={speaker.portrait} alt={`Portrait of ${speaker.name}`} />
          {meta ? (
            <Player
              src={dataUrl(speaker.slug, "preview.wav")}
              segments={meta.segments}
              peaks={meta.peaks}
              duration={meta.duration}
            />
          ) : (
            loaded && (
              <p className="player__text player__text--disclaimer">
                No audio yet. Run <code>mvs export-web {speaker.slug}</code> after building the pipeline.
              </p>
            )
          )}
          <Label variant="note" as="p" className="speaker__notice">
            Markov chain remix of real clips · not real statements
          </Label>
        </section>

        {matrix && (
          <section className="transitions">
            <Label>Transition matrix</Label>
            <p className="transitions__caption">
              Probability of each next word or phrase given the current one, across {speaker.name}'s{" "}
              {matrix.labels.length} most frequent playable entries. Darker is likelier.
            </p>
            <MatrixView matrix={matrix} />
          </section>
        )}
      </main>
    </PageShell>
  );
}
