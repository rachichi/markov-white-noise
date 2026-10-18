# markov-white-noise

Experimentation of transition matrixes in Markov chains on audio transcripts.

`mvs` (Markov Voice Stream) turns recordings of a person's public speech into an endless, **audio-only**, clearly labelled remix: an order-2 word-level Markov chain built from their transcripts chooses each next word, and every word is a real aligned clip cut from the source audio. There's no voice cloning or synthesis of the speaker.

A speaker is just `speakers/<slug>.yaml` plus a folder of audio, so adding one needs no code changes.

## Disclosure (always on)

These rules are enforced, not optional. `preview`, `render` and `stream` refuse to start if any of them is switched off.

- **Spoken disclaimer:** a generic, clearly synthetic TTS voice (never the speaker's) reads it at the start of every stream session and every file, then at most every `disclaimer_every_minutes` (10 or fewer).
- **Metadata:** title, artist, album, comment and the stream name and description all say it is a Markov chain remix that was never actually said. Branding or endorsement wording is rejected.
- **Blocklist:** every sentence passes through [`config/blocklist.txt`](config/blocklist.txt) (election terms, calls to action, threats, personal information, slurs) and is resampled on a hit. **`config/blocklist-slurs.txt` ships empty.** Fill it from a maintained list before publishing; `mvs` warns while it is empty.
- **Traceability:** `SOURCES.md` lists every source (title, URL, date, license). `mvs purge --source <id>` deletes a source and every clip derived from it, then rebuilds the clips and chain.

## Setup

```sh
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'   # core: generator, assembly, delivery
.venv/bin/pip install -e '.[align]'                              # stages 2-3: WhisperX (torch)
brew install ffmpeg                                              # MP3/Opus output, streaming, non-WAV input
```

Signature-word scoring needs a reference corpus at `data/reference/general_english.txt` (see [its README](data/reference/README.md)).

Disclaimer voice: Piper (`output.tts.engine: piper` with `model: path/to/voice.onnx`), Kokoro, espeak-ng, or macOS `say` for local testing. `auto` tries them in that order.

## Pipeline

```sh
mvs ingest <speaker>        # 1  convert audio in sources_dir (described in its sources.yaml) -> SOURCES.md
mvs transcribe <speaker>    # 2-3 WhisperX words + timestamps
mvs vocab <speaker>         # 4  function + signature words/phrases; vocab_report.md lists gaps
mvs clips <speaker>         # 5  cut, score, keep the best variants; clips_report.md
mvs chain <speaker>         # 6  order-n transition counts with back-off -> chain.json
mvs text <speaker> -n 10    #    text-only sentences
mvs preview <speaker> -n 5  # 7-8 disclaimer + 5 sentences to data/<slug>/out/preview.wav
mvs render <speaker> --hours 10 [--rss-base-url URL]   # files mode (output.mode: files|both)
mvs stream <speaker> --icecast icecast://source:PASS@host:8000/mount   # stream mode
mvs purge --source <id or file>
```

Each stage reads and writes files under `data/<slug>/` and can be re-run on its own. Source audio goes in `data/<slug>/audio/` with a `sources.yaml` entry per file; files without title, URL, date and license are not ingested. `ingest --url` downloads a directly linked public file only (no logins, paywalls or scraping).

The stream writes through `ffmpeg -re` to Icecast, reconnects with backoff (each reconnect replays the disclaimer), logs to `data/<slug>/logs/` with rotation, and serves `GET http://127.0.0.1:8090/health`. Deployment: [`Dockerfile`](Dockerfile) and [`deploy/mvs-stream@.service`](deploy/mvs-stream@.service).

## Tests

```sh
.venv/bin/pytest
```

The suite runs stages 4-9 and purge on a synthetic speaker (tone-burst "words" with known timestamps), plus the blocklist suite, the disclosure gate, loudness against the BS.1770 reference tone, and the disclaimer schedule and maximum-gap checks. WhisperX, ffmpeg and Icecast are not exercised by the tests.

## Web page

`npm install && npm run dev` serves a Vite + React page built with `@rachichi/design` at http://localhost:5173.

- **Speaker carousel:** portraits stand behind a fixed podium (`src/assets/podium.png`). Clicking ‹ ›, or a side portrait, turns the stage so the next speaker rotates in.
- **Speaker blurb:** the paragraph above the carousel belongs to the speaker at the podium and slides in when the stage turns. It comes from that speaker's `intro` in `src/speakers.ts`, with an optional list of the source recordings (title, link, part used, word count).
- **Endless player:** sentences are generated live in the browser, word by word, from the speaker's transition matrix. `src/engine.ts` is a port of `mvs/generator.py` (order-2 chain with back-off, blocklist, choice of take per word, equal-power crossfades). Each sentence is scheduled on the Web Audio clock a few seconds ahead, so the waveform scrolls past the playhead showing real upcoming audio. The word being spoken is underlined. The spoken disclaimer plays first and then at most every `output.disclaimer_every_minutes`.
- **Transition matrix:** collapsible, with a colour legend mapping shades to probabilities. It shows first-order P(next | word); the generator itself uses order 2 and backs off to order 1.

The page reads what `mvs export-web <speaker> [--top N]` writes to `public/speakers/<slug>/`:

| File | Contents |
| ---- | -------- |
| `generator.json` | the chain (only transitions to playable entries), clip offsets, generator settings, the disclaimer text and interval, and the blocklist terms the chain could actually produce |
| `clips.wav` | the spoken disclaimer followed by every playable word clip, as one 16-bit mono WAV |
| `matrix.json` | first-order P(next \| word) for every playable entry (`--top N` keeps the N most frequent) |

Source recordings are always git-ignored. Exports under `public/speakers/` are git-ignored too, except for `omer`, who agreed to publication, and `trump`, whose clips come from public speeches. To publish another speaker's export, get their permission (or check that the source terms allow it), note it in their `sources.yaml` license field, and add a `!public/speakers/<slug>/` line to `.gitignore`.

## Adding a person

**Before you start:** only use recordings you are allowed to use. That means a public, attributable source whose terms allow reuse, or your own recording made with the speaker's permission. Record the permission in `sources.yaml`. Don't use private conversations, and don't use anyone's voice to imitate endorsements, election messages or statements they didn't make. The output is a labelled remix of real clips, not a cloned voice, and the disclosure below can't be switched off.

1. **Audio:** put the recordings in `data/<slug>/audio/` (any of wav, mp3, m4a, flac, ogg, opus, webm, mp4). Describe each file in `data/<slug>/audio/sources.yaml`:

   ```yaml
   20261014_lecture.m4a:
     title: "Lecture, 14 October 2026"
     url: "https://... (or: local recording, where it was made)"
     date: 2026-10-14
     license: "Recorded with permission"   # or the source's license / terms of use
   ```

   **Every recording must be solo:** only this person speaking, with no questions, interviewers or introductions. There's no speaker separation, so trim each recording to the stretch where they speak alone, and keep the untrimmed original out of `sources_dir`:

   ```sh
   ffmpeg -ss 00:02:00 -to 01:10:45 -i original.mp3 -map 0:a -ac 1 -c:a flac data/<slug>/audio/2016-07-21_speech.flac
   ```

   Re-encode rather than `-c copy`: downloads named `.mp3` are sometimes Opus inside, and copying the stream breaks the timestamps.

   Files without a title, URL, date and license are skipped.

   **How much audio:** aim for **about 2 hours of the person actually speaking**. The minimum is 1 hour, and gains get small after 3–4 hours.

   | Solo speech | Result |
   | ----------- | ------ |
   | 10 min | ~1,400 words; under 40% of vocabulary entries have 3+ clean takes, so takes repeat and sentences lean on a few common words |
   | 1 hour | common words well covered; still needs the lowered vocab settings below |
   | **2 hours** | ~17,000 words; signature phrases get enough clean takes and the default settings work |
   | 3–4 hours | more take variety and smoother joins; the chain itself barely improves |

   Recording length isn't speech time. Only about 45% of a lecture survives the quality filters (silences, questions, echo), so 2 hours of speech can take around 3 hours of recordings. A close microphone, and several sessions on different days help most. Stop adding audio once the coverage in `vocab_report.md` and `clips_report.md` stops climbing toward 90%.

2. **Config:** copy [`speakers/example-speaker.yaml`](speakers/example-speaker.yaml) to `speakers/<slug>.yaml` and set `name`, `slug`, `sources_dir` and `output.stream_title`. Keep "remix" and "not real statements" (or similar) in the title.

   With under an hour of audio, lower the vocabulary targets as [`speakers/omer.yaml`](speakers/omer.yaml) does (`vocab.min_count: 3`, fewer `signature_words`). With 2 hours or more, the defaults work. `mvs vocab` and `mvs clips` print coverage reports listing which words still need more takes.

3. **Build:**

   ```sh
   mvs ingest <slug> && mvs transcribe <slug> && mvs vocab <slug> && mvs clips <slug> && mvs chain <slug>
   mvs text <slug> -n 10        # sanity-check the sentences
   mvs export-web <slug>        # writes public/speakers/<slug>/
   ```

4. **Page:** add a portrait (a transparent PNG cut out at the chest works best behind the podium) to `src/assets/`. Then add an entry to `src/speakers.ts`:

   ```ts
   import ada from "./assets/ada.png";
   {
     slug: "ada",
     name: "Ada Lovelace",
     portrait: ada,
     bottomPadding: 0.05,
     intro: "What these recordings are and how the chain was built from them.",
     sources: [{ title: "Lecture, 14 October 2026", url: "https://...", used: "00:00–52:10", words: 7300 }],
   },
   ```

   `intro` and `sources` are optional and fill the blurb above the carousel. `bottomPadding` is the transparent margin below the figure as a fraction of the image height. Set it if the person floats above the podium. The order of `speakers` is the carousel order. A speaker with no export yet shows a "No recordings yet" note.

**Adding more audio for an existing person:** drop the new files into `data/<slug>/audio/`, add them to `sources.yaml`, and rerun step 3. Only new files are transcribed. **Removing a recording:** `mvs purge --source <id or file name>` deletes it and every clip cut from it, rebuilds the clips and chain, and then you rerun `mvs export-web <slug>`.

### Disclaimers to keep in place

- The spoken disclaimer, the "Markov chain remix of real clips · not real statements" note under the player, and the "not real statements" stream title must stay visible and audible on any copy you publish.
- `config/blocklist-slurs.txt` is empty. Fill it from a maintained list (for example LDNOOBW) before publishing anything.
- `SOURCES.md` in each speaker's work folder lists every recording used. Publish it alongside the audio, and honour removal requests with `mvs purge`.
- Public figures are included for parody and study of speech patterns. Don't publish output that could be mistaken for a genuine statement, and don't publish it around elections.
