import omer from "./assets/omer.png";

export interface SpeakerPage {
  slug: string; // matches speakers/<slug>.yaml and public/speakers/<slug>/ from `mvs export-web`
  name: string;
  portrait: string;
}

export const speakers: SpeakerPage[] = [{ slug: "omer", name: "Omar El Housni", portrait: omer }];

export const dataUrl = (slug: string, file: string) => `${import.meta.env.BASE_URL}speakers/${slug}/${file}`;
