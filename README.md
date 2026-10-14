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
.venv/bin/pip install -e '.[align]'                              # stages 2-3: WhisperX + pyannote (torch)
brew install ffmpeg                                              # MP3/Opus output, streaming, non-WAV input
```

Diarization uses gated pyannote models: accept their terms on Hugging Face and set `HF_TOKEN`. Signature-word scoring needs a reference corpus at `data/reference/general_english.txt` (see [its README](data/reference/README.md)).

Disclaimer voice: Piper (`output.tts.engine: piper` with `model: path/to/voice.onnx`), Kokoro, espeak-ng, or macOS `say` for local testing. `auto` tries them in that order.

## Pipeline

```sh
mvs ingest <speaker>        # 1  convert audio in sources_dir (described in its sources.yaml) -> SOURCES.md
mvs transcribe <speaker>    # 2-3 WhisperX words + timestamps, pyannote speaker isolation
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

The suite runs stages 4-9 and purge on a synthetic speaker (tone-burst "words" with known timestamps), plus the blocklist suite, the disclosure gate, loudness against the BS.1770 reference tone, and the disclaimer schedule and maximum-gap checks. WhisperX, pyannote, ffmpeg and Icecast are not exercised by the tests.

## Web page

`npm run dev` serves a Vite + React page built with `@rachichi/design`: the speaker's portrait, a play/pause button with the waveform of a preview recording, the sentence currently playing underneath, and the full scrollable transition matrix of their most frequent words and phrases.

The page reads what `mvs export-web <speaker>` writes to `public/speakers/<slug>/`:

- `preview.wav`: the spoken disclaimer plus `-n` generated sentences
- `preview.json`: when each sentence starts and ends, so the text follows playback
- `matrix.json`: first-order P(next | word) for every playable entry (`--top N` to trim)

Speakers shown on the page are listed in `src/speakers.ts` (portraits in `src/assets/`). Exported `.wav` files are git-ignored.
