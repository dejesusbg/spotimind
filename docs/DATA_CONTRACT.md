# Data contract

Single source of truth for the files the Colab notebook (`notebooks/01_embed_catalog.ipynb`) produces and the local app (`backend/`, `frontend/`) consumes. `scripts/make_mock_data.py` produces the exact same schema with synthetic values.

All arrays are little-endian. All `.npy` files are plain NumPy v1 format (no pickles).

Facts below marked **(verified)** were measured in Phase 0 (`notebooks/00_phase0_feasibility.ipynb`, TRIBE v2 @ `af58661`, Colab T4).

## Spaces and atlas

| Item | Value |
|---|---|
| TRIBE output | `float32 [T, 20484]` per song, fsaverage5, left hemisphere vertices 0–10241 then right 10242–20483 **(verified)** |
| Temporal resolution | 1 Hz (TR = 1 s). Hemodynamic lag already compensated by the model (5 s offset) **(verified)** |
| Parcellation | Schaefer 2018, 400 parcels, 7-network order, FreeSurfer fsaverage5 `.annot` (CBIG). Native fsaverage5, no projection. |
| Parcel index `p` | `0..199` = left-hemisphere parcels 1..200, `200..399` = right-hemisphere parcels 1..200 (Schaefer order). Medial wall / unlabeled vertices = `-1`. |
| Networks | Yeo-7 (from Schaefer parcel names) + an **Auditory** network carved out of SomMot, see `networks.json`. |

## `data/` layout

```
data/
  songs.json
  meta.json
  networks.json
  embeddings_brain.npy
  embeddings_audio.npy
  timelines/{id}.npy
  mesh/cortex.json
  mesh/cortex.bin
audio/                      # the user's audio files (gitignored); backend streams from here
```

### `songs.json`

A JSON array. Row order defines row order of both embedding matrices.

```json
{
  "id": "daft-punk-one-more-time",       // slug of the filename stem, unique, stable across reruns
  "title": "One More Time",               // ID3 title, else filename stem
  "artist": "Daft Punk",                  // ID3 artist, else parsed from "Artist - Title" stem, else null
  "filename": "Daft Punk - One More Time.mp3",   // file name inside audio/
  "duration_s": 320.4,                    // full audio duration
  "processed_duration_s": 295.0,          // seconds of audio represented by the timeline (= bin_count * 1 s)
  "bin_count": 295,                       // T, number of 1 Hz rows in the timeline
  "timeline_start_s": 3.0                 // audio time of timeline row 0
}
```

Timeline row `i` is the predicted response for the audio second `[timeline_start_s + i, timeline_start_s + i + 1)`. The frontend treats row `i` as the value at `t = timeline_start_s + i + 0.5` and linearly interpolates between rows; before the first / after the last row it holds the edge value.

### `timelines/{id}.npy`

`float32 [T, P]`, `P = 400`. Each column is the mean TRIBE prediction over the fsaverage5 vertices of parcel `p`. **No per-song normalization** (raw model units, roughly z-scored by TRIBE's training targets). Already trimmed:

- `TRIM_START_S = 3`: model edge transient at the start of any input **(verified: first ~2 s unreliable)**.
- `TRIM_END_S = 2`: same at the end.
- Songs longer than `MAX_SECONDS = 300` are cut at 300 s before inference (default mode). With `WINDOW_MODE = true`, a 90 s window from the middle of the song is used instead and `timeline_start_s` points into the middle of the song.

Known artifact: TRIBE predicts in non-overlapping 100 s windows, so there is a mild discontinuity at processed seconds 100 and 200 **(verified, ~2.3x the median frame-to-frame change)**. Left as is and documented.

### `embeddings_brain.npy`

`float32 [N, D_brain]`, L2-normalized rows, so cosine similarity = dot product. Built from the parcel timeline `X [T, P]` of each song:

| Block | Shape | Definition |
|---|---|---|
| `mean` | `[P]` = 400 | mean over time per parcel |
| `std` | `[P]` = 400 | standard deviation over time per parcel |
| `shape` | `[N_BINS * K]` = 16 × 8 = 128 | network-level timeline (mean over parcels in each of the K = 8 networks), split into `N_BINS = 16` equal time bins, mean within each bin |

Normalization (catalog-level, chosen on purpose):
1. Each raw feature is z-scored **across the catalog** (mean/std over the N songs, std floored at 1e-6). This makes similarity depend on how songs differ from each other, not on features that are high for every song.
2. Each block is L2-normalized per song, so each block contributes equally regardless of its dimensionality.
3. Blocks are multiplied by `BLOCK_WEIGHTS` (default 1/1/1, stored in `meta.json`), concatenated, and the full row L2-normalized.

Per-song normalization (min-max or z-scoring each song's own timeline) was rejected: it stretches every song to the same range and erases exactly the between-song differences we want to compare (this is what Reeled In's `NORMALIZATION_DECISION.md` found for clips). The z-scores depend on the catalog, so adding songs changes everyone's embedding slightly; the notebook recomputes all embeddings at the end of every run, which is cheap.

### `embeddings_audio.npy`

`float32 [N, 2048]`, the baseline. Wav2Vec-BERT 2.0 features taken from TRIBE's own audio extractor (`[2 layer groups, 1024, T @ 2 Hz]`, layers 0.5/0.75/1.0 grouped as TRIBE does) **(verified)**, mean-pooled over the same trimmed time span as the timeline, then catalog z-scored and row L2-normalized. Same row order as `songs.json`.

### `networks.json`

```json
{
  "atlas": "Schaefer2018_400Parcels_7Networks_order (fsaverage5, CBIG)",
  "auditory_source": "Destrieux 2009 (nilearn fetch_atlas_surf_destrieux, fsaverage5)",
  "n_parcels": 400,
  "networks": [
    { "id": "auditory", "name": "Auditory", "yeo": null, "description": "...", "parcels": [ ... ] },
    { "id": "visual",   "name": "Visual",   "yeo": "Vis", "description": "...", "parcels": [ ... ] }
  ],
  "parcels": [ { "index": 0, "name": "7Networks_LH_Vis_1", "hemi": "L", "network": "visual", "n_vertices": 47 } ],
  "assumptions": [ "..." ]
}
```

Networks, in this order: `auditory`, `visual`, `somatomotor`, `dorsal_attention`, `ventral_attention`, `limbic`, `control`, `default`. Each parcel belongs to exactly one network. A parcel is reassigned from SomMot to `auditory` when at least 30% of its vertices fall in the Destrieux auditory regions (`G_temp_sup-G_T_transv`, `S_temporal_transverse`, `G_temp_sup-Plan_tempo`, `G_temp_sup-Lateral`). There is no language network: Yeo-7 has none and the spec says drop rather than fake. A per-network value is always the plain mean of its parcels' values.

### `meta.json`

```json
{
  "mock": false,
  "created_utc": "2026-10-04T23:00:00Z",
  "tribe": { "repo": "facebookresearch/tribev2", "commit": "af58661…", "weights": "facebook/tribev2" },
  "mesh_space": "fsaverage5",
  "parcellation": "Schaefer2018_400Parcels_7Networks_order",
  "n_parcels": 400,
  "tr_s": 1.0,
  "max_seconds": 300, "window_mode": false, "window_seconds": 90,
  "trim_start_s": 3, "trim_end_s": 2,
  "n_bins": 16,
  "brain_embedding": { "blocks": [ {"name": "mean", "dim": 400, "weight": 1.0}, … ], "dim": 928, "normalization": "catalog z-score per feature, L2 per block, weighted concat, L2 row" },
  "audio_embedding": { "source": "tribev2 Wav2VecBert extractor (facebook/w2v-bert-2.0)", "dim": 2048, "normalization": "catalog z-score, L2 row" },
  "n_songs": 100,
  "failures": [ { "filename": "…", "error": "…" } ]
}
```

### `mesh/cortex.json` + `mesh/cortex.bin`

Static mesh exported once by `scripts/export_mesh.py` (nilearn fsaverage5 **inflated** surface, both hemispheres, the right one shifted along +x so they don't overlap; coordinates in mm).

`cortex.json`:

```json
{
  "space": "fsaverage5", "surface": "inflated",
  "n_vertices": 20484, "n_faces": 40960,
  "hemispheres": { "left": {"vertex_start": 0, "vertex_count": 10242}, "right": {"vertex_start": 10242, "vertex_count": 10242} },
  "n_parcels": 400,
  "buffers": {
    "positions": { "offset": 0,      "count": 61452, "dtype": "float32", "item_size": 3 },
    "faces":     { "offset": 245808, "count": 122880, "dtype": "uint32", "item_size": 3 },
    "parcel":    { "offset": 737328, "count": 20484, "dtype": "int16",  "item_size": 1 }
  }
}
```

`offset` is in bytes inside `cortex.bin`. Face indices are global (right-hemisphere faces already offset by 10242). `parcel[v]` is the parcel index of vertex `v` (`-1` = medial wall, drawn grey).

## API notes

- `GET /api/mesh` returns `cortex.json`; `GET /api/mesh/bin` returns `cortex.bin` (split so the browser can fetch the binary as an `ArrayBuffer`).
- `GET /api/meta` adds `network_range` (min/max of all songs' per-network means), so network bars share one scale across songs, and `explain_available`.
- `GET /api/networks` returns `networks.json` (the frontend needs the parcel lists to compute live per-network values).
- `data/mesh/` is produced locally by `scripts/export_mesh.py`, not by the notebook.

## API wire format for timelines

`GET /api/songs/{id}/timeline` returns `application/octet-stream`:

| Bytes | Type | Meaning |
|---|---|---|
| 0–3 | uint32 | T |
| 4–7 | uint32 | P |
| 8–11 | float32 | `timeline_start_s` |
| 12–15 | float32 | sample rate in Hz (1.0) |
| 16… | float32 × T·P | row-major timeline |
