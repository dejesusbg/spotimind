# Spotimind

Match songs by how a brain is *predicted* to respond to them, instead of by tags like tempo or genre.

Spotimind runs Meta's open brain-encoding model [TRIBE v2](https://github.com/facebookresearch/tribev2) on a local music catalog. For every second of a song it predicts the fMRI response of an average human cortex (20,484 surface points), reduces that to 400 brain regions, and summarizes each song as a "brain-response fingerprint". Songs are matched by cosine similarity of those fingerprints. The app plays a song while a 3D cortex lights up in sync, and shows the brain-space neighbors next to neighbors from a plain audio embedding (Wav2Vec-BERT), so you can see whether brain space actually groups songs differently.

Inspired by [Reeled In](https://github.com/sxnnywu/reeled-in) (TRIBE v2 for short-form video) and a TRIBE-on-songs demo by @Baconbrix.

## Honest limits

- **Average brain, not yours.** TRIBE v2 predicts the response of an averaged subject. It says nothing about personal taste.
- **Not trained on music.** The model learned from fMRI recorded during movies and audiobooks. Results on music are directional. In our Phase 0 check, music clearly raised auditory cortex, but lateral occipital (visual) areas rose even more, most likely a cross-modal habit learned from movies.
- **Slow and smeared.** One reading per second, blurred by a few seconds. Trust shapes over seconds, not beats.
- **A proxy.** Brain-space similarity is a hypothesis about "feels alike", not a measurement of it. The side-by-side audio baseline is there to keep it honest.
- **Explanations are interpretations.** The optional Gemini "why do these match?" text is generated only from the computed numbers and labelled "Interpretation, not a measurement".

## License

TRIBE v2 code and weights are **CC BY-NC 4.0**. This is a non-commercial portfolio project. Never commit audio, model weights or API keys (see `.gitignore`).

## How it works

```
mp3s (Google Drive) ──► Colab T4: TRIBE v2, audio only ──► data/ (npy + json) ──► FastAPI ──► Next.js + three.js
```

1. **Inference** (`notebooks/01_embed_catalog.ipynb`): each song (capped at 5 min) goes through TRIBE v2 audio-only → `[T seconds, 20484 vertices]`. The first 3 s and last 2 s are trimmed (model edge effects measured in Phase 0).
2. **Parcels and networks**: vertices are averaged into the Schaefer-400 parcellation (native fsaverage5). Parcels are grouped into Yeo-7 networks, plus an auditory network carved out of somatomotor using the Destrieux auditory labels. All values are plain averages; see `data/networks.json` for every assumption.
3. **Brain embedding**: per-parcel mean and std over time, plus 16 time bins of the network-level timeline. Each feature is z-scored across the catalog, each block L2-normalized, then the blocks are concatenated (928 dims).
4. **Audio baseline**: TRIBE's own Wav2Vec-BERT 2.0 features (the model's audio input), mean-pooled over the same span (2048 dims).
5. **App**: numpy cosine similarity, a binary timeline endpoint, and a react-three-fiber cortex recolored per frame from the parcel timeline.

The full file and API spec is in [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md). Phase 0 findings (install, output shape, lag, speed, sanity checks) are recorded at the top of [`notebooks/00_phase0_feasibility.ipynb`](notebooks/00_phase0_feasibility.ipynb).

## Repo layout

```
docs/DATA_CONTRACT.md          file + API spec (single source of truth)
notebooks/00_phase0_feasibility.ipynb
notebooks/01_embed_catalog.ipynb
scripts/export_mesh.py         cortex mesh + per-vertex parcel ids -> data/mesh/
scripts/make_mock_data.py      synthetic data with the same schema -> mock/
scripts/atlas.py               Schaefer/Yeo/Destrieux helpers
backend/                       FastAPI app + pytest suite
frontend/                      Next.js app
data/   audio/   mock/         gitignored
```

## Run it on mock data (no GPU, no songs)

Requires [uv](https://docs.astral.sh/uv/) and Node 20+.

```bash
# 1. synthetic data (real fsaverage5 mesh if online; add --no-real-mesh for an offline sphere)
uv run --with numpy --with nilearn --with nibabel scripts/make_mock_data.py

# 2. backend on :8000
cd backend
SPOTIMIND_DATA_DIR=../mock/data SPOTIMIND_AUDIO_DIR=../mock/audio uv run uvicorn app.main:app --port 8000

# 3. frontend on :3000 (another terminal)
cd frontend && npm install && npm run dev
```

The UI shows a red **MOCK DATA** badge (driven by `meta.json`).

Tests: `cd backend && uv run pytest` (they generate their own mock data).

## Run it on your own songs

### 1. Colab (free T4, ~25 s per song)

1. Put your mp3s in a Google Drive folder. The notebook defaults to `MyDrive/spotimind/audio/` (change `AUDIO_DIR` in cell C1).
2. Open `notebooks/01_embed_catalog.ipynb` in Colab and pick **Runtime → Change runtime type → T4 GPU**.
3. Run the install cell. It installs TRIBE v2 at a pinned commit, then forces `torch==2.6.0 torchaudio==2.6.0 numpy==2.2.6` and **restarts the runtime** (expected). The older "pin numpy<2.1" advice no longer applies: tribev2 itself pins numpy 2.2.6.
4. **Hugging Face token: not needed.** Audio-only inference skips the gated LLaMA text model (verified in Phase 0). Only `facebook/tribev2` and `facebook/w2v-bert-2.0` are downloaded, and both are public. If you later add the text/video branches you'll need an HF token with access to `meta-llama/Llama-3.2-3B`; add it with Colab's **Secrets** panel, never in a cell.
5. Mount Drive. If the `drive.mount()` popup fails, use the **Files sidebar → Mount Drive** button instead.
6. Run the cells in order. The catalog cell runs in a background thread; poll it with the progress cell. Each finished song is checkpointed to `MyDrive/spotimind/output/work/{id}.npz`, so if Colab disconnects, rerun everything and finished songs are skipped.
7. The last cells write `MyDrive/spotimind/output/data/` and a `spotimind_data.zip`, and print neighbors for three random songs in both spaces.

### 2. Copy the results to the repo

```bash
# download spotimind_data.zip from MyDrive/spotimind/output/ (Drive web UI), then:
mkdir -p data && unzip -o ~/Downloads/spotimind_data.zip -d data
# the cortex mesh is exported locally (once):
uv run --with nilearn --with nibabel scripts/export_mesh.py --out data/mesh
# the app streams audio from audio/: download the Drive folder and put the mp3s there
mkdir -p audio && cp /path/to/your/mp3s/*.mp3 audio/
```

### 3. Run

```bash
cd backend && uv run uvicorn app.main:app --port 8000     # defaults: ../data and ../audio
cd frontend && npm run dev
```

Optional explanations: put the line `GEMINI_API_KEY=your-key` in `backend/.env` (gitignored; never put a real key in any committed file) or export it before starting the backend. The model defaults to `gemini-3.8-flash` (`GEMINI_MODEL`). When Gemini is overloaded (429/503), the backend retries twice, then tries `gemini-3.5-flash` (`GEMINI_FALLBACK_MODEL`), then shows a "try again in a minute" message. Without a key the "why?" buttons are disabled.

Frontend env: `NEXT_PUBLIC_API_BASE` (default `http://localhost:8000`), e.g. in `frontend/.env.local`.

### Adding songs later

Drop new mp3s into the Drive folder and rerun the notebook. Already-processed songs are skipped. Embeddings are recomputed for the whole catalog at the end, because z-scoring is catalog-level. Then re-download the zip and copy the new mp3s into `audio/`.

## First catalog run (100 songs, 2026-10-04)

- 100/100 songs, 0 failures, 38 min on a free T4 (356 min of audio, 6.4 s compute per audio-minute including Drive I/O).
- Neighbors in raw-audio space cluster mostly by artist (same voice and production). Brain-space neighbors cross artists and languages.
- Mean top-10 overlap between the two spaces: Jaccard 0.138, vs about 0.053 for random lists. The spaces are related but clearly not the same.
- Catalog-average network means are highest for auditory (+0.061), then visual (+0.037), consistent with the Phase 0 caveat.

## Unverified / known issues

- TRIBE predicts in 100 s windows without overlap, which gives a mild discontinuity at 100 s and 200 s into a song. It is left in, and documented.
- The vertex order (left then right hemisphere) was checked indirectly (left/right symmetry r=0.97), not against an official statement in the TRIBE repo.
- Five lateral superior-temporal parcels overlap auditory anatomy but stay in Yeo's default network, because the carve-out only takes somatomotor parcels.
