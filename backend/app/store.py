"""Loads data/ into memory once at startup (see docs/DATA_CONTRACT.md)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .similarity import cosine_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
TIMELINE_HEADER = np.dtype([("t", "<u4"), ("p", "<u4"), ("start", "<f4"), ("rate", "<f4")])


@dataclass(eq=False)  # identity hash, so lru_cache works on methods
class Store:
    data_dir: Path
    audio_dir: Path
    meta: dict
    songs: list[dict]
    networks: dict
    emb: dict[str, np.ndarray]
    sims: dict[str, np.ndarray] = field(init=False)
    index: dict[str, int] = field(init=False)
    netmat: np.ndarray = field(init=False)
    net_means: np.ndarray = field(init=False)  # [N, K] per-song mean activation per network

    def __post_init__(self) -> None:
        self.index = {s["id"]: i for i, s in enumerate(self.songs)}
        self.sims = {space: cosine_matrix(e) for space, e in self.emb.items()}
        n_parcels = self.networks["n_parcels"]
        nets = self.networks["networks"]
        self.netmat = np.zeros((n_parcels, len(nets)), np.float32)
        for j, net in enumerate(nets):
            self.netmat[net["parcels"], j] = 1.0 / len(net["parcels"])
        self.net_means = np.stack([self.timeline(s["id"]).mean(0) @ self.netmat for s in self.songs])

    @property
    def network_ids(self) -> list[str]:
        return [n["id"] for n in self.networks["networks"]]

    def timeline(self, song_id: str) -> np.ndarray:
        return np.load(self.data_dir / "timelines" / f"{song_id}.npy", mmap_mode="r")

    def timeline_bytes(self, song_id: str) -> bytes:
        tl = np.ascontiguousarray(self.timeline(song_id), "<f4")
        song = self.songs[self.index[song_id]]
        header = np.array([(tl.shape[0], tl.shape[1], song["timeline_start_s"], 1.0 / self.meta.get("tr_s", 1.0))],
                          TIMELINE_HEADER)
        return header.tobytes() + tl.tobytes()

    def network_values(self, i: int) -> dict[str, float]:
        return {nid: float(v) for nid, v in zip(self.network_ids, self.net_means[i])}

    def network_percentiles(self, i: int) -> dict[str, float]:
        """Where song i sits within the catalog for each network (0 = lowest, 100 = highest)."""
        n = len(self.songs)
        ranks = (self.net_means < self.net_means[i]).sum(0)
        return {nid: float(100 * r / max(n - 1, 1)) for nid, r in zip(self.network_ids, ranks)}

    @lru_cache(maxsize=256)
    def cover(self, filename: str) -> tuple[bytes, str] | None:
        """Embedded cover art of an audio file as (bytes, mime), or None."""
        try:
            from mutagen import File as MutagenFile

            tags = MutagenFile(self.audio_dir / filename)
            if tags is None or tags.tags is None:
                return None
            pics = [v for k, v in tags.tags.items() if k.startswith("APIC")]
            return (pics[0].data, pics[0].mime or "image/jpeg") if pics else None
        except Exception:
            return None


def load_store(data_dir: Path | None = None, audio_dir: Path | None = None) -> Store:
    data_dir = Path(data_dir or os.environ.get("SPOTIMIND_DATA_DIR") or REPO_ROOT / "data")
    audio_dir = Path(audio_dir or os.environ.get("SPOTIMIND_AUDIO_DIR") or REPO_ROOT / "audio")
    if not (data_dir / "songs.json").exists():
        raise FileNotFoundError(
            f"No songs.json in {data_dir}. Copy the notebook output there, or run "
            "`uv run --with numpy scripts/make_mock_data.py` and set SPOTIMIND_DATA_DIR=mock/data."
        )
    songs = json.loads((data_dir / "songs.json").read_text())
    meta = json.loads((data_dir / "meta.json").read_text())
    networks = json.loads((data_dir / "networks.json").read_text())
    emb = {
        "brain": np.load(data_dir / "embeddings_brain.npy"),
        "audio": np.load(data_dir / "embeddings_audio.npy"),
    }
    for space, e in emb.items():
        if e.shape[0] != len(songs):
            raise ValueError(f"embeddings_{space}.npy has {e.shape[0]} rows but songs.json has {len(songs)} songs")
    return Store(data_dir=data_dir, audio_dir=audio_dir, meta=meta, songs=songs, networks=networks, emb=emb)
