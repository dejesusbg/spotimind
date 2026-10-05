"""Atlas helpers shared by export_mesh.py and make_mock_data.py.

Mirrors cell [C3] of notebooks/01_embed_catalog.ipynb (the notebook keeps its own copy so it
stays self-contained in Colab). Keep the two in sync; the contract is docs/DATA_CONTRACT.md.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

SCHAEFER = "Schaefer2018_400Parcels_7Networks_order"
SCHAEFER_BASE = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/"
    "Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5/label"
)
N_PER_HEMI = 10242
N_PARCELS = 400
AUDITORY_MIN_FRACTION = 0.30
AUD_DESTRIEUX = ["G_temp_sup-G_T_transv", "S_temporal_transverse", "G_temp_sup-Plan_tempo", "G_temp_sup-Lateral"]
YEO = {
    "Vis": "visual",
    "SomMot": "somatomotor",
    "DorsAttn": "dorsal_attention",
    "SalVentAttn": "ventral_attention",
    "Limbic": "limbic",
    "Cont": "control",
    "Default": "default",
}
NET_ORDER = [
    "auditory", "visual", "somatomotor", "dorsal_attention",
    "ventral_attention", "limbic", "control", "default",
]
NET_INFO = {
    "auditory": ("Auditory", None, "Schaefer SomMot parcels overlapping Destrieux auditory cortex (Heschl's gyrus, transverse sulcus, planum temporale, lateral superior temporal gyrus)."),
    "visual": ("Visual", "Vis", "Yeo-7 visual network."),
    "somatomotor": ("Somatomotor", "SomMot", "Yeo-7 somatomotor network, minus the parcels moved to Auditory."),
    "dorsal_attention": ("Dorsal attention", "DorsAttn", "Yeo-7 dorsal attention network (goal-directed attention)."),
    "ventral_attention": ("Salience / ventral attention", "SalVentAttn", "Yeo-7 salience / ventral attention network."),
    "limbic": ("Limbic", "Limbic", "Yeo-7 limbic network (orbitofrontal cortex, temporal pole). Low fMRI signal quality in these areas."),
    "control": ("Control", "Cont", "Yeo-7 frontoparietal control network."),
    "default": ("Default mode", "Default", "Yeo-7 default mode network."),
}


def load_schaefer(cache_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Per-vertex parcel index [20484] (-1 = medial wall) and the 400 parcel names."""
    import nibabel as nib

    cache_dir.mkdir(parents=True, exist_ok=True)
    vert_parcel, names_all = [], []
    for hemi, offset in (("lh", 0), ("rh", 200)):
        path = cache_dir / f"{hemi}.{SCHAEFER}.annot"
        if not path.exists():
            urllib.request.urlretrieve(f"{SCHAEFER_BASE}/{hemi}.{SCHAEFER}.annot", path)
        labels, _, names = nib.freesurfer.read_annot(str(path))
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        assert labels.shape[0] == N_PER_HEMI and len(names) == 201
        vert_parcel.append(np.where(labels > 0, labels - 1 + offset, -1))
        names_all += names[1:]
    return np.concatenate(vert_parcel).astype(np.int16), names_all


def build_networks(vert_parcel: np.ndarray, parcel_names: list[str]) -> dict:
    """networks.json content: Yeo-7 from Schaefer names plus the Destrieux auditory carve-out."""
    from nilearn import datasets

    des = datasets.fetch_atlas_surf_destrieux()
    dlabels = [l.decode() if isinstance(l, bytes) else str(l) for l in des["labels"]]
    dmap = np.concatenate([des["map_left"], des["map_right"]]).astype(int)
    aud_vert = np.isin(dmap, [dlabels.index(n) for n in AUD_DESTRIEUX])

    parcel_net = [YEO[n.split("_")[2]] for n in parcel_names]
    kept_elsewhere: dict[str, int] = {}
    for p in range(N_PARCELS):
        if aud_vert[vert_parcel == p].mean() >= AUDITORY_MIN_FRACTION:
            if parcel_net[p] == "somatomotor":
                parcel_net[p] = "auditory"
            else:
                kept_elsewhere[parcel_net[p]] = kept_elsewhere.get(parcel_net[p], 0) + 1
    counts = np.bincount(vert_parcel[vert_parcel >= 0], minlength=N_PARCELS)
    return networks_doc(parcel_names, parcel_net, counts, kept_elsewhere)


def networks_doc(parcel_names, parcel_net, counts, kept_elsewhere, mock=False) -> dict:
    networks = []
    for nid in NET_ORDER:
        name, yeo, desc = NET_INFO[nid]
        networks.append({
            "id": nid, "name": name, "yeo": yeo, "description": desc,
            "parcels": [p for p, n in enumerate(parcel_net) if n == nid],
        })
    parcels = [
        {"index": p, "name": parcel_names[p], "hemi": "L" if p < 200 else "R",
         "network": parcel_net[p], "n_vertices": int(counts[p])}
        for p in range(len(parcel_names))
    ]
    assumptions = [
        "Parcellation: Schaefer 2018, 400 parcels, 7-network order, native FreeSurfer fsaverage5 annotation (CBIG). No volume-to-surface projection.",
        "TRIBE v2 output vertex order assumed to be [left 10242, right 10242] fsaverage5; verified in Phase 0 by left/right regional symmetry (r=0.97).",
        "Each parcel's network comes from its Schaefer name (Yeo 2011, 7 networks).",
        f"Auditory: a SomMot parcel is relabelled 'auditory' when >= {int(AUDITORY_MIN_FRACTION * 100)}% of its vertices fall in Destrieux labels {', '.join(AUD_DESTRIEUX)}.",
        f"Parcels in other Yeo networks that also meet the auditory overlap rule are left in their Yeo network: {kept_elsewhere or 'none'}.",
        "No language network: Yeo-7 has none, so it is omitted rather than approximated.",
        "A network value is the plain mean of its parcels' values; a parcel value is the plain mean of its vertices' TRIBE predictions.",
    ]
    if mock:
        assumptions.insert(0, "MOCK: synthetic parcellation and network assignment, for app development only.")
    return {
        "atlas": f"{SCHAEFER} (fsaverage5, CBIG)" if not mock else "MOCK sphere parcellation",
        "auditory_source": "Destrieux 2009 (nilearn fetch_atlas_surf_destrieux, fsaverage5)",
        "n_parcels": len(parcel_names),
        "networks": networks,
        "parcels": parcels,
        "assumptions": assumptions,
    }
