"""Export the static cortex mesh used by the 3D viewer: data/mesh/cortex.{json,bin}.

fsaverage5 inflated surface from nilearn (both hemispheres, pulled apart along x) and the
Schaefer-400 parcel index of every vertex. Layout is specified in docs/DATA_CONTRACT.md.

    uv run --with nilearn --with nibabel scripts/export_mesh.py --out data/mesh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from atlas import N_PARCELS, N_PER_HEMI, load_schaefer

GAP_MM = 8.0


def load_inflated() -> tuple[np.ndarray, np.ndarray]:
    from nilearn import datasets, surface

    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    coords, faces = [], []
    for hemi in ("left", "right"):
        mesh = surface.load_surf_mesh(fs[f"infl_{hemi}"])
        c, f = np.asarray(mesh.coordinates, np.float32), np.asarray(mesh.faces, np.uint32)
        assert c.shape[0] == N_PER_HEMI
        coords.append(c)
        faces.append(f)
    # Inflated hemispheres overlap at the midline: push them apart so their inner edges are GAP_MM apart.
    left, right = coords
    left[:, 0] -= left[:, 0].max() + GAP_MM / 2
    right[:, 0] -= right[:, 0].min() - GAP_MM / 2
    positions = np.concatenate([left, right])
    positions -= (positions.max(0) + positions.min(0)) / 2  # centre the bounding box
    faces = np.concatenate([faces[0], faces[1] + N_PER_HEMI])
    return positions, faces


def write_mesh(out: Path, positions: np.ndarray, faces: np.ndarray, parcel: np.ndarray,
               space: str, surface_name: str, n_parcels: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    blobs = {
        "positions": np.ascontiguousarray(positions, "<f4"),
        "faces": np.ascontiguousarray(faces, "<u4"),
        "parcel": np.ascontiguousarray(parcel, "<i2"),
    }
    item = {"positions": 3, "faces": 3, "parcel": 1}
    buffers, offset = {}, 0
    with open(out / "cortex.bin", "wb") as f:
        for name, arr in blobs.items():
            data = arr.tobytes()
            buffers[name] = {"offset": offset, "count": int(arr.size), "dtype": str(arr.dtype.newbyteorder("=").name), "item_size": item[name]}
            f.write(data)
            offset += len(data)
            pad = (-offset) % 4  # keep every buffer 4-byte aligned for typed-array views
            f.write(b"\0" * pad)
            offset += pad
    n_v = positions.shape[0]
    half = n_v // 2
    desc = {
        "space": space,
        "surface": surface_name,
        "n_vertices": int(n_v),
        "n_faces": int(faces.shape[0]),
        "hemispheres": {"left": {"vertex_start": 0, "vertex_count": half},
                        "right": {"vertex_start": half, "vertex_count": n_v - half}},
        "n_parcels": n_parcels,
        "buffers": buffers,
    }
    (out / "cortex.json").write_text(json.dumps(desc, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/mesh"))
    ap.add_argument("--cache", type=Path, default=Path(".cache/atlas"))
    args = ap.parse_args()

    positions, faces = load_inflated()
    vert_parcel, _ = load_schaefer(args.cache)
    assert vert_parcel.shape[0] == positions.shape[0]
    write_mesh(args.out, positions, faces, vert_parcel, "fsaverage5", "inflated", N_PARCELS)
    print(f"wrote {args.out}/cortex.json + cortex.bin: {positions.shape[0]} vertices, {faces.shape[0]} faces, "
          f"{(vert_parcel < 0).sum()} medial-wall vertices")


if __name__ == "__main__":
    main()
