"""Generate a synthetic data/ + audio/ pair with exactly the schema of docs/DATA_CONTRACT.md.

Uses the real fsaverage5 mesh + Schaefer/Yeo networks when nilearn and the network are
available, otherwise a two-sphere mesh with fake parcels. Audio is short generated WAV.

    uv run --with numpy --with nilearn --with nibabel scripts/make_mock_data.py            # -> mock/data, mock/audio
    uv run --with numpy scripts/make_mock_data.py --no-real-mesh                          # offline sphere
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.embedding import audio_embeddings, brain_embeddings  # noqa: E402
from atlas import N_PARCELS, NET_ORDER, networks_doc  # noqa: E402
from export_mesh import write_mesh  # noqa: E402

SR = 16000
N_BINS = 16
TRIM_START_S, TRIM_END_S = 3, 2
WEIGHTS = {"mean": 1.0, "std": 1.0, "shape": 1.0}
STYLES = [  # name, root Hz, tempo BPM, network profile (one value per NET_ORDER entry)
    ("Drone", 110.0, 60, [0.9, 0.1, -0.2, -0.3, 0.0, 0.2, -0.4, 0.6]),
    ("Pulse", 146.8, 128, [0.6, 0.5, 0.7, 0.3, 0.4, -0.3, 0.1, -0.5]),
    ("Chime", 220.0, 90, [0.7, 0.6, -0.1, 0.5, -0.2, 0.0, 0.4, 0.1]),
    ("Noise", 98.0, 100, [0.4, -0.2, 0.2, -0.1, 0.8, 0.5, -0.3, -0.2]),
]
ADJ = ["Velvet", "Static", "Hollow", "Golden", "Paper", "Neon", "Quiet", "Glass", "Lunar", "Salt"]
NOUN = ["Harbor", "Engine", "Garden", "Signal", "Mirror", "Tide", "Atlas", "Ember", "Orbit", "Field"]
ARTISTS = ["Mock Ensemble", "Test Pattern", "The Placeholders", "Synthetic Choir"]


def smooth_noise(rng, n: int, k: int, width: float) -> np.ndarray:
    """[n, k] Gaussian-smoothed noise, unit-ish variance."""
    x = rng.standard_normal((n + 60, k))
    t = np.arange(-30, 31)
    kern = np.exp(-0.5 * (t / width) ** 2)
    kern /= kern.sum()
    out = np.stack([np.convolve(x[:, j], kern, mode="same") for j in range(k)], 1)[30:30 + n]
    return out / max(out.std(), 1e-6)


def synth_audio(rng, path: Path, seconds: float, root: float, bpm: int, noisy: bool) -> None:
    t = np.arange(int(seconds * SR)) / SR
    chord = sum(np.sin(2 * np.pi * root * r * t + rng.uniform(0, 6)) for r in (1.0, 1.25, 1.5))
    beat = 0.5 + 0.5 * (np.sin(2 * np.pi * (bpm / 60) * t) > 0.6)
    sig = chord / 3 * (0.55 + 0.45 * beat)
    if noisy:
        sig += 0.25 * rng.standard_normal(t.size) * beat
    fade = np.minimum(1, np.minimum(t, seconds - t) / 1.5)
    pcm = np.int16(np.clip(sig * fade * 0.5, -1, 1) * 32767)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def sphere_mesh(rng, n_lat: int = 48, n_lon: int = 96):
    """Two UV spheres (left/right) with 200 nearest-centroid parcels each."""
    lat = np.linspace(0, np.pi, n_lat)
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    unit = np.stack([np.sin(la) * np.cos(lo), np.sin(la) * np.sin(lo), np.cos(la)], -1).reshape(-1, 3)
    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            a, b = i * n_lon + j, i * n_lon + (j + 1) % n_lon
            c, d = a + n_lon, b + n_lon
            faces += [(a, c, b), (b, c, d)]
    faces = np.array(faces, np.uint32)
    nv = unit.shape[0]
    positions, all_faces, parcel = [], [], []
    for h, (dx, off) in enumerate(((-75.0, 0), (75.0, 200))):
        pos = unit * np.array([65.0, 100.0, 75.0]) + np.array([dx, 0, 0])
        cent = rng.standard_normal((200, 3))
        cent /= np.linalg.norm(cent, axis=1, keepdims=True)
        lab = np.argmax(unit @ cent.T, axis=1) + off
        lab[unit[:, 0] * (1 if h == 0 else -1) > 0.85] = -1  # fake medial wall
        positions.append(pos)
        all_faces.append(faces + h * nv)
        parcel.append(lab)
    names = [f"Mock_{'LH' if p < 200 else 'RH'}_{p % 200 + 1}" for p in range(N_PARCELS)]
    return (np.concatenate(positions).astype(np.float32), np.concatenate(all_faces),
            np.concatenate(parcel).astype(np.int16), names)


def build_mesh_and_networks(rng, mesh_dir: Path, use_real: bool, cache: Path):
    if use_real:
        try:
            from atlas import build_networks, load_schaefer
            from export_mesh import load_inflated

            positions, faces = load_inflated()
            vert_parcel, names = load_schaefer(cache)
            networks = build_networks(vert_parcel, names)
            write_mesh(mesh_dir, positions, faces, vert_parcel, "fsaverage5", "inflated", N_PARCELS)
            return networks, "fsaverage5", "Schaefer2018_400Parcels_7Networks_order"
        except Exception as e:  # offline or nilearn missing
            print(f"real mesh unavailable ({e!r}); falling back to sphere")
    positions, faces, vert_parcel, names = sphere_mesh(rng)
    net_of = [NET_ORDER[int(i)] for i in rng.integers(0, len(NET_ORDER), N_PARCELS)]
    counts = np.bincount(vert_parcel[vert_parcel >= 0], minlength=N_PARCELS)
    write_mesh(mesh_dir, positions, faces, vert_parcel, "mock-sphere", "sphere", N_PARCELS)
    return networks_doc(names, net_of, counts, {}, mock=True), "mock-sphere", "mock"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate mock Spotimind data")
    ap.add_argument("--out", type=Path, default=ROOT / "mock")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-real-mesh", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    data, audio = args.out / "data", args.out / "audio"
    for d in (data, audio):
        if d.exists():
            shutil.rmtree(d)
    (data / "timelines").mkdir(parents=True)
    audio.mkdir(parents=True)

    networks, space, parcellation = build_mesh_and_networks(rng, data / "mesh", not args.no_real_mesh, ROOT / ".cache/atlas")
    (data / "networks.json").write_text(json.dumps(networks, indent=1))
    netmat = np.zeros((N_PARCELS, len(NET_ORDER)), np.float32)
    parcel_net = np.zeros(N_PARCELS, int)
    for j, net in enumerate(networks["networks"]):
        netmat[net["parcels"], j] = 1.0 / max(len(net["parcels"]), 1)
        parcel_net[net["parcels"]] = j

    songs, timelines, audio_vecs = [], [], []
    style_audio_dirs = rng.standard_normal((len(STYLES), 2048))
    for i in range(args.n):
        s = i % len(STYLES)
        style, root, bpm, profile = STYLES[s]
        title = f"{ADJ[i % len(ADJ)]} {NOUN[(i + 3 * (i // len(ADJ))) % len(NOUN)]}"  # unique for n <= 30
        sid = f"mock-{i:02d}-{title.lower().replace(' ', '-')}"
        dur = float(rng.uniform(35, 70))
        fname = f"{sid}.wav"
        synth_audio(rng, audio / fname, dur, root * rng.uniform(0.9, 1.1), bpm, noisy=style == "Noise")

        T = int(np.ceil(dur)) - TRIM_START_S - TRIM_END_S
        latent = np.array(profile) + 0.35 * rng.standard_normal(len(NET_ORDER))       # this song's network levels
        net_ts = latent + 0.5 * smooth_noise(rng, T, len(NET_ORDER), width=4.0)        # [T, K]
        loading = 0.6 + 0.8 * rng.random(N_PARCELS)
        tl = net_ts[:, parcel_net] * loading * 0.4 + 0.15 * smooth_noise(rng, T, N_PARCELS, width=2.0)
        tl = tl.astype(np.float32)
        np.save(data / "timelines" / f"{sid}.npy", tl)
        timelines.append(tl)
        # Audio baseline: mostly style-driven, but every 3rd song leans to the next style, so the
        # two spaces agree only partly (makes the side-by-side comparison non-trivial).
        a_style = (s + 1) % len(STYLES) if i % 3 == 0 else s
        audio_vecs.append(style_audio_dirs[a_style] + 0.8 * rng.standard_normal(2048))
        songs.append({
            "id": sid, "title": title, "artist": ARTISTS[s % len(ARTISTS)] if i % 5 else None,
            "filename": fname, "duration_s": round(dur, 3), "processed_duration_s": float(T),
            "bin_count": T, "timeline_start_s": float(TRIM_START_S),
        })

    emb_brain, blocks = brain_embeddings(timelines, netmat, N_BINS, WEIGHTS)
    emb_audio = audio_embeddings(audio_vecs)
    np.save(data / "embeddings_brain.npy", emb_brain)
    np.save(data / "embeddings_audio.npy", emb_audio)
    (data / "songs.json").write_text(json.dumps(songs, indent=1, ensure_ascii=False))
    meta = {
        "mock": True,
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tribe": {"repo": "facebookresearch/tribev2", "commit": None, "weights": None},
        "mesh_space": space,
        "parcellation": parcellation,
        "n_parcels": N_PARCELS,
        "tr_s": 1.0,
        "max_seconds": 300, "window_mode": False, "window_seconds": 90,
        "trim_start_s": TRIM_START_S, "trim_end_s": TRIM_END_S,
        "n_bins": N_BINS,
        "brain_embedding": {"blocks": blocks, "dim": int(emb_brain.shape[1]),
                            "normalization": "catalog z-score per feature, L2 per block, weighted concat, L2 row"},
        "audio_embedding": {"source": "MOCK random vectors", "dim": int(emb_audio.shape[1]),
                            "normalization": "catalog z-score, L2 row"},
        "n_songs": len(songs),
        "failures": [],
    }
    (data / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"mock data: {len(songs)} songs -> {data} (mesh: {space}), audio -> {audio}")


if __name__ == "__main__":
    main()
