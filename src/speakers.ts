import mamdani from "./assets/mamdani.png";
import omer from "./assets/omer.png";
import trump from "./assets/trump.png";

export interface SpeakerPage {
  slug: string; // matches speakers/<slug>.yaml and public/speakers/<slug>/ from `mvs export-web`
  name: string;
  portrait: string;
  /** Transparent margin under the figure, as a fraction of image height, so everyone stands on the podium. */
  bottomPadding?: number;
}

// Carousel order: the middle entry is shown first, with its neighbours on either side.
export const speakers: SpeakerPage[] = [
  { slug: "mamdani", name: "Zohran Mamdani", portrait: mamdani, bottomPadding: 0.078 },
  { slug: "omer", name: "Omar El Housni", portrait: omer },
  { slug: "trump", name: "Donald Trump", portrait: trump },
];

export const dataUrl = (slug: string, file: string) => `${import.meta.env.BASE_URL}speakers/${slug}/${file}`;
