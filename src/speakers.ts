import mamdani from "./assets/mamdani.png";
import omer from "./assets/omer.png";
import trump from "./assets/trump.png";

export interface SpeakerPage {
  slug: string; // matches speakers/<slug>.yaml and public/speakers/<slug>/ from `mvs export-web`
  name: string;
  portrait: string;
  /** Transparent margin under the figure, as a fraction of image height, so everyone stands on the podium. */
  bottomPadding?: number;
  /** Blurb shown above the carousel while this speaker is at the podium. */
  intro?: string;
  /** Recordings the chain was built from, listed under the intro. */
  sources?: { title: string; url: string; used: string; words: number }[];
}

// Carousel order: the middle entry is shown first, with its neighbours on either side.
export const speakers: SpeakerPage[] = [
  { slug: "mamdani", name: "Zohran Mamdani", portrait: mamdani, bottomPadding: 0.078 },
  {
    slug: "omer",
    name: "Omar El Housni",
    portrait: omer,
    intro:
      "Professor Omar El Housni taught us about Markov Chains transition matrixes as a generative tool, so I " +
      "created one using a recording of the lecture in which he taught us about Markov Chains transition " +
      "matrixes. Now you can tune in to his lectures as you study for the prelim next Monday.",
  },
  {
    slug: "trump",
    name: "Donald Trump",
    portrait: trump,
    intro:
      "The transcripts I used included 1,545 unique words in total. The Markov Chain transition matrix uses " +
      "only 296 entries — the 100 most frequent function words (e.g. the, and, of), 200 of his more " +
      "characteristic words and phrases, minus the 4 clips that were not playable/clear. As a reminder, this " +
      "is an automated Markov Chain remix of public speech clips. Nothing you hear was actually said in this order.",
    sources: [
      {
        title: "Inaugural Address, 20 January 2017",
        url: "https://www.youtube.com/watch?v=zHp1SGTfRMs",
        used: "01:00–16:50",
        words: 1425,
      },
      {
        title: "Republican National Convention acceptance speech, 21 July 2016",
        url: "https://www.youtube.com/watch?v=va9ilyjMyik",
        used: "02:00–1:10:45",
        words: 4929,
      },
    ],
  },
];

export const dataUrl = (slug: string, file: string) => `${import.meta.env.BASE_URL}speakers/${slug}/${file}`;
