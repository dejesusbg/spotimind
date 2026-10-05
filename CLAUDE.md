# Project: "Spotimind" — match songs by predicted brain response (TRIBE v2), portfolio demo

You are building a portfolio project end to end. Read this whole document before writing any code, then start with a short plan and proceed in the order given in "Delivery order".

---

## 1. The idea in one paragraph

Streaming services match songs using hand-picked statistics (valence, tempo, energy, artist). Statistics can make songs look alike that don't actually feel alike. This project instead simulates how a human brain responds to each song, using Meta's open-source brain-encoding model **TRIBE v2** (predicts fMRI responses from audio/video/text without a scanner), and matches songs by similarity of their *predicted brain-response patterns*. The demo: pick a song from a local catalog (~100 mp3s), see its nearest neighbors in "brain space", watch an animated 3D cortex light up in sync with the audio, and compare brain-space neighbors against neighbors from a plain audio embedding.

This was inspired by "Reeled In" (https://github.com/sxnnywu/reeled-in), a hackathon project that used TRIBE v2 to score short-form video, and by a demo by @Baconbrix showing TRIBE v2 on songs with a 3D brain. We are doing the music-matching version, heavily simplified.

## 2. Hard constraints and non-goals

- **Audio-only.** No video, no lyrics/text channel in v1.
- **No Spotify, no streaming APIs.** Audio comes from local mp3 files the user puts in a folder.
- **No auth, no database server, no Modal, no Base44, no Auth0, no MongoDB.** Storage is files (npz/parquet/json). Search is plain numpy cosine similarity.
- **Two deliverables:** (A) a Google Colab notebook that runs TRIBE v2 on a free T4 and produces data files; (B) a local app (FastAPI backend + Next.js frontend) that consumes those files. The app needs no GPU.
- **No evaluation harness** beyond the built-in side-by-side comparison described below.
- **License:** TRIBE v2 weights/code are CC BY-NC. This is a non-commercial portfolio project. State this in the README. Never commit mp3s, model weights, or API keys. Add a proper `.gitignore`.
- Your own environment has no GPU, so you **cannot run TRIBE v2 yourself**. Everything GPU-related must be written defensively, with clear cell-by-cell sanity checks. Everything else (embedding math, API, frontend) must be testable locally, using the mock data described in section 4.
- **Colab MCP (check first):** the user may have a Colab MCP server connected to your session (e.g. Google's official `googlecolab/colab-mcp`, which links local agents to a Colab session open in the user's browser). At the start, check which Colab-related MCP tools you have. If you do, use them to run and debug the notebooks on the user's free T4 yourself: execute cells, read errors, fix, and rerun, instead of asking the user to copy and paste. If you don't have them, fall back to asking the user to run the notebooks and paste back outputs. Rules when using Colab MCP: never print, log, or write the Hugging Face token or any secret (ask the user to add it via Colab Secrets or enter it themselves in the notebook UI); if Drive mounting needs a consent click, ask the user to do it; free-tier sessions disconnect, so rely on the resumable per-song checkpointing; downloading results back into the local `data/` folder should go through Drive or the MCP's file/artifact tools, and you should tell the user if a manual copy is needed.

## 3. References to read first (use WebFetch; treat as reference, don't copy code verbatim)

- TRIBE v2 repo: https://github.com/facebookresearch/tribev2 (inference API, output shape/space, how audio-only is passed, install instructions)
- DataCamp tutorial "TRIBE v2 Tutorial: Simulating Human Brain Activity from Video, Audio, and Text" (https://www.datacamp.com/tutorial/tribe-v2-tutorial). Key facts from it: audio branch is Wav2Vec-BERT 2.0 at ~2 Hz; the model was trained with modality dropout so audio-only inference is supported; in Colab you must pin `numpy<2.1` before installing tribev2 and restart the runtime; install via `pip install 'tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git'`.
- Reeled In repo docs, for ideas on reducing TRIBE output to interpretable networks and on normalization: `HOW_TRIBE_V2_WORKS.md`, `SCORING_SCIENCE.md`, `NORMALIZATION_DECISION.md`, `TECH_ARCHITECTURE.md` at https://github.com/sxnnywu/reeled-in
- Known caveats from those docs: TRIBE outputs roughly **one reading per second**, so it is reliable at the few-second level, not frame by frame; it was trained on movies/audiobooks, **not music specifically**, so results on music are directional.
- Sources disagree on the number of predicted brain points (~20k cortical vertices vs ~29k vs ~70k voxels). **Do not assume. Verify the actual output shape and space in Phase 0** and design the mesh/parcellation around what you find. (Likely fsaverage5, ~20,484 cortical vertices, but confirm.)
- Notes: LLaMA 3.2-3B is a gated Hugging Face model and TRIBE v2's text extractor depends on it; the user will need an HF token with access even for audio-only if the library loads all extractors. Find out whether audio-only inference can skip loading the video and text extractors (important for the T4's 16 GB).

## 4. Data contract (define this first, in `docs/DATA_CONTRACT.md`, and keep it as the single source of truth)

The notebook produces, and the app consumes, a `data/` directory:

- `data/songs.json`: list of `{ id, title, artist (if known), filename, duration_s, processed_duration_s, bin_count }`. Parse title/artist from ID3 tags if present, otherwise from filename.
- `data/timelines/{id}.npy`: float32 array `[T, P]`, the predicted parcel-level response over time (T ≈ seconds processed, P = number of parcels). Timeline already trimmed of the initial transient (see 5.1).
- `data/embeddings_brain.npy`: float32 `[N, D_brain]`, one row per song, row order = order in `songs.json`.
- `data/embeddings_audio.npy`: float32 `[N, D_audio]`, the baseline embedding from the raw audio features (mean-pooled Wav2Vec-BERT features, or equivalent), same row order.
- `data/networks.json`: the parcel→network mapping actually used, with documented assumptions and the atlas it came from.
- `data/meta.json`: run configuration (clip cap/window, trim seconds, binning, TRIBE version/commit, parcellation name, mesh space, date).
- `data/mesh/cortex.bin` (+ `cortex.json` descriptor): static cortex mesh (positions, faces) and a per-vertex parcel-index array, exported once by a script (e.g. via nilearn `fsaverage`), matching the space TRIBE outputs.
- `audio/`: the user's mp3s (gitignored). The backend streams from here with HTTP range support.

**Mock mode (required):** write `scripts/make_mock_data.py` that generates synthetic data with *exactly the same schema* (random smooth timelines, fake songs, a real cortex mesh if available offline, otherwise a simple sphere mesh with parcel labels, and short generated sine/noise wav files so the audio player works). The backend and frontend must run end-to-end on mock data before any real data exists. Clearly flag mock data in the UI (a visible "MOCK DATA" badge driven by `meta.json`).

## 5. Deliverable A: Colab notebook(s)

Create `notebooks/00_phase0_feasibility.ipynb` and `notebooks/01_embed_catalog.ipynb`. Keep cells small and named, with an assertion or printed check after each step, so failures are easy to report back.

### 5.0 Phase 0 — feasibility notebook (small, runs first)
Goals, each printed clearly at the end as a checklist:
1. Install correctly in Colab (numpy pin, restart note, tribev2 install, HF login for gated models). Document the exact working install sequence.
2. Run TRIBE v2 **audio-only** on one mp3. Confirm it works and record how audio is passed in.
3. Record: output array shape, what the axes mean, output space (fsaverage5 or other), temporal resolution, and whether outputs are time-shifted relative to the stimulus (hemodynamic lag). Decide the trim of the initial transient and document it.
4. Record GPU memory used and **seconds of compute per minute of audio** on the T4. Use this to recommend: full songs (cap ~5 min) vs a 90-second window. Print the estimated total time for 100 songs.
5. Check whether the extractors for video/text can be skipped for audio-only, and if so, how.
6. Sanity check the output: e.g. predicted activity for music should be strongest in auditory cortex regions; render a quick plot of per-network mean activation over time for one song. If the output looks like noise, say so loudly.
7. Verify we can also obtain the raw audio-feature embedding (Wav2Vec-BERT features) for the baseline: either from TRIBE's internals or by loading `facebook/w2v-bert-2.0` via transformers separately.

### 5.1 Main notebook — catalog embedding
- Mount Drive; read mp3s from a configurable folder (e.g. `MyDrive/spotimind/audio`). Config cell: `MAX_SECONDS` (default 300 for full songs), `WINDOW_MODE` flag (use a 90 s window from the song's middle instead), `TRIM_SECONDS` (initial transient), `N_BINS = 16`.
- **Incremental and resumable:** after each song, write its timeline and a per-song feature record to Drive. On rerun, skip songs already processed. (Free Colab sessions disconnect; this is mandatory. It is also how new songs get added later: drop the mp3 in the folder and rerun.)
- Reduce the TRIBE output to **parcels** using an established atlas that matches the output space (e.g. Schaefer or Glasser/HCP-MMP via nilearn, a few hundred parcels). Save parcel-level timelines `[T, P]`.
- **Networks:** group parcels into a small set of interpretable networks (aim for ~6: auditory, language, visual, motion/somatomotor, attention, default-mode), using a published mapping (e.g. Yeo 7/17 networks overlaid on the parcellation, plus an explicit auditory-cortex selection from the atlas's labeled regions). **Only real averages over parcels, no invented scores.** Document every assumption in `networks.json`; if a network can't be defined cleanly from the atlas, drop it rather than fake it.
- **Brain embedding per song:** concatenate three blocks computed from the parcel timeline: (a) mean over time `[P]`, (b) standard deviation over time `[P]`, (c) the timeline resampled to `N_BINS` bins then averaged within bins, giving `[N_BINS × P]` (consider reducing P for this block, e.g. via network-level or PCA, to keep dimensionality sane). Standardize each feature across the catalog (z-score), L2-normalize each block, then weight blocks (config, default equal) and concatenate. Make the normalization approach explicit and documented (compare per-song vs catalog-level normalization; read Reeled In's `NORMALIZATION_DECISION.md` for context; choose and justify).
- **Audio baseline embedding:** mean-pooled raw audio features per song, standardized and L2-normalized. Same song order.
- Write all files in the data contract. Finish with a summary cell: songs processed, failures, total compute time, and a quick printed neighbor list for 3 random songs in both spaces so the user can eyeball results.
- Provide a `scripts/export_mesh.py` (runnable locally or in Colab) for the cortex mesh + per-vertex parcel indices.

## 6. Deliverable B: local app

### 6.1 Backend (FastAPI, Python, in `backend/`)
Load everything into memory at startup. No DB. Endpoints:
- `GET /api/meta`: run config + mock flag.
- `GET /api/songs`: list.
- `GET /api/songs/{id}`: metadata plus per-network mean activation (real averages from `networks.json`).
- `GET /api/songs/{id}/timeline`: parcel timeline `[T, P]` (as compact JSON or binary float32; choose binary + small header for speed) .
- `GET /api/songs/{id}/neighbors?k=10&space=brain|audio`: cosine similarity over the chosen embedding matrix, excluding the song itself; return ids, titles, scores.
- `GET /api/compare/{id}?k=10`: both lists plus the overlap between them (number of shared songs in the top-k, and Jaccard). This is the built-in test of whether brain space adds anything over raw audio.
- `GET /api/audio/{id}`: stream the mp3 with HTTP range support (needed for seeking/sync).
- `GET /api/mesh`: cortex mesh and per-vertex parcel indices.
- `POST /api/songs/{id}/explain` with body `{ other_id }`: Gemini explanation of why two songs match (see below). Must degrade gracefully if no key is set.
- CORS enabled for `http://localhost:3000`.
- Tests (pytest) for the similarity math, the neighbor endpoints, and the overlap metric, run against mock data.

**Gemini explanation:** use the official Google GenAI Python SDK; read `GEMINI_API_KEY` from env; read the model name from env `GEMINI_MODEL` (check current docs for a sensible default, don't hardcode a stale one). Build the prompt only from numbers we computed: both songs' per-network mean activations, the cosine similarity in both spaces, and the biggest network differences. Instruct the model to use *only* the provided numbers, not to invent musical facts, and to keep it to 3-4 sentences. The UI labels the output "Interpretation, not a measurement". If the key is missing, the endpoint returns a clear message and the UI hides or disables the button.

### 6.2 Frontend (Next.js App Router + TypeScript + Tailwind, in `frontend/`)
Runs on `localhost:3000`, calls the FastAPI backend (configurable base URL via env). Features:
- **Song list** with search; clicking selects a song.
- **Audio player** (HTML audio element hitting `/api/audio/{id}`), play/pause/seek.
- **3D cortex** with react-three-fiber/three.js: load the static mesh once; for the current audio time, interpolate the parcel timeline (1 Hz readings, linear interpolation between them), map parcel values to per-vertex colors through the vertex→parcel index array, update the vertex color buffer in a `useFrame` loop using typed arrays (no per-frame allocations, target smooth 30+ fps). Drag to orbit. Use a perceptually sensible colormap, with a documented per-song display normalization (e.g. percentile clipping) so the animation is readable. When paused, show the frame at the current time.
- **Network bars** for the selected song: mean activation per network (real averages only), plus a live marker for the current time's value per network if cheap to do.
- **Neighbors, side by side:** two columns, "Brain-space neighbors" and "Audio-embedding neighbors", each with similarity scores; highlight songs that appear in both; show the overlap number from `/api/compare`. Clicking a neighbor selects it. A "compare" mode that shows the two songs' network bars next to each other.
- **"Why do these match?"** button on a neighbor row, calling the explain endpoint; show the response with the "Interpretation, not a measurement" label.
- **Honest-limits note** in the UI footer/info panel: predictions are for an average brain, were learned from movie/audiobook data rather than music, and are a proxy for similarity, not for personal taste. TRIBE v2 is CC BY-NC (non-commercial).
- A visible **MOCK DATA** badge when `meta.json` says so.
- Reasonable, polished design suited to a portfolio piece (dark theme is fine). Keep the component structure clean.

## 7. Repo layout (suggested)

```
spotimind/
  README.md
  .gitignore
  docs/DATA_CONTRACT.md
  notebooks/00_phase0_feasibility.ipynb
  notebooks/01_embed_catalog.ipynb
  scripts/make_mock_data.py
  scripts/export_mesh.py
  backend/ (FastAPI app, tests/)
  frontend/ (Next.js app)
  data/    (gitignored except mock sample if small)
  audio/   (gitignored)
```

README must include: what it is, honest limits, license note, Colab steps (including HF token and the numpy pin/restart), how to copy `data/` from Drive to the repo, how to run backend and frontend, how to add new songs (drop mp3 in Drive folder, rerun notebook), and how to run in mock mode.

## 8. Delivery order

1. Short plan and any questions that block you (ask only if truly blocking).
2. Repo scaffold, `DATA_CONTRACT.md`, `make_mock_data.py`, `.gitignore`.
3. `00_phase0_feasibility.ipynb`. If you have Colab MCP tools, run it on the T4 yourself, fix failures, and report the checklist output to the user. Otherwise **tell the user to run it on Colab and paste back the checklist output.** Either way, continue with steps 4-5 while it runs or while waiting (they don't depend on it).
4. Backend with tests, running on mock data.
5. Frontend running on mock data end to end (player, 3D cortex, bars, side-by-side neighbors, Gemini button).
6. After the user reports Phase 0 results, finalize `01_embed_catalog.ipynb` and `export_mesh.py` for the actual output space, adjust the data contract if reality differs from assumptions, and update the README.
7. Final pass: run the tests, run the app on mock data, and list any remaining unknowns honestly.

## 9. Working style

- Don't silently assume things about TRIBE v2's API or output; verify against the repo and mark anything unverified in code comments and the README.
- Prefer simple, readable code over clever abstractions. Type hints in Python, strict TypeScript.
- When something in this spec conflicts with what you find in the TRIBE v2 repo, follow reality, update `DATA_CONTRACT.md`, and tell the user what changed and why.
