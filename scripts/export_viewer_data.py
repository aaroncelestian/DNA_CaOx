#!/usr/bin/env python3
"""
Pack the DLS growth model into a compact helix-frame JSON for the 3D viewer.

DNA is reduced to backbone traces (P, C1', glycosidic N). CaOx is Ca sites,
oxalate C2O4 sticks, water oxygens (OW/HOH), and a smoothed occupancy envelope.
"""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dls_caox import find_oxalates  # noqa: E402
from oxalate_sticks import (  # noqa: E402
    oxalate_segments_by_residue,
    oxalate_segments_crystal_patch,
)
from find_symmetry import (  # noqa: E402
    COM_A,
    DNA_HOTSPOT_DP_MAX,
    HOTSPOT_CLUSTER_CUT,
    cluster_local,
    helix_axis_radius,
    load_growth_seed_positions,
    load_whw_ca,
    load_dna_heavy,
    load_phosphate_xyz,
    phosphate_surface_radius,
)
from grow_crystal_from_growth import dna_slab_frame  # noqa: E402
from grow_whewellite import parse_atoms  # noqa: E402
from plot_sleeve_diagram import helix_basis  # noqa: E402

SEED_PDB = ROOT / "DNA_CaOx_growth.pdb"
OUT = ROOT / "viewer" / "model-data.js"
DEFAULT_GEOM = "templating_gel"

# Viewer CUT options (export only these into model-data.js).
GEOMETRIES = {
    "templating_gel": {
        "pdb": ROOT / "DNA_CaOx_templating_gel_omm.pdb",
        "csv": ROOT
        / "figures/crystallinity/DNA_CaOx_templating_gel_omm_ca_metrics.csv",
        "seeds": ROOT / "DNA_CaOx_templating_gel_seeds.pdb",
        "title": "Gel",
        "cut": "220 CaOx (44 at P + 88 second-row + 88 third-row) and extra waters; FIRE with no COM targets, gel unfrozen",
        "seedRadius": 24.0,
        "cutKind": "spheres",
        "oxalate": True,
        "traj": "trajectories/DNA_CaOx_templating_gel_fire.trj.json",
    },
    "templating_gel_thick": {
        "pdb": ROOT / "DNA_CaOx_templating_gel_thick_omm.pdb",
        "csv": ROOT
        / "figures/crystallinity/DNA_CaOx_templating_gel_thick_omm_ca_metrics.csv",
        "seeds": ROOT / "DNA_CaOx_templating_gel_thick_seeds.pdb",
        "title": "5-row gel + MD",
        "cut": "396 CaOx (44 at P + 352 coat) and ~800 extra waters; honest FIRE, no COM targets",
        "seedRadius": 30.0,
        "cutKind": "spheres",
        "oxalate": True,
        "traj": "trajectories/DNA_CaOx_templating_gel_thick_fire.trj.json",
    },
    "templating_gel_10shell": {
        "pdb": ROOT / "DNA_CaOx_templating_gel_10shell_omm.pdb",
        "csv": ROOT
        / "figures/crystallinity/DNA_CaOx_templating_gel_10shell_omm_ca_metrics.csv",
        "seeds": ROOT / "DNA_CaOx_templating_gel_10shell_seeds.pdb",
        "title": "10-row gel +MD",
        "cut": "616 CaOx (44 at P + 572 coat, 10 rows) + ~1300 waters; FIRE + NVT MD",
        "seedRadius": 50.0,
        "cutKind": "spheres",
        "oxalate": True,
        "traj": "trajectories/DNA_CaOx_templating_gel_10shell_fire.trj.json",
    },
    "templating_gel_15shell": {
        "pdb": ROOT / "DNA_CaOx_templating_gel_15shell_omm.pdb",
        "csv": ROOT
        / "figures/crystallinity/DNA_CaOx_templating_gel_15shell_omm_ca_metrics.csv",
        "seeds": ROOT / "DNA_CaOx_templating_gel_15shell_seeds.pdb",
        "title": "15-row gel + MD",
        "cut": "836 CaOx (44 at P + 792 coat, 15 rows) + ~1800 waters; FIRE + 0.1 ns NVT MD",
        "seedRadius": 72.0,
        "cutKind": "spheres",
        "oxalate": True,
        "traj": "trajectories/DNA_CaOx_templating_gel_15shell_fire.trj.json",
    },
    "templating_nodna": {
        "pdb": ROOT / "DNA_CaOx_templating_gel_nodna_omm.pdb",
        "csv": ROOT
        / "figures/crystallinity/DNA_CaOx_templating_gel_nodna_omm_ca_metrics.csv",
        "seeds": None,
        "title": "Blob",
        "cut": "Same unit count as templating gel, random sphere, no DNA, no COM targets",
        "seedRadius": 24.0,
        "cutKind": "spheres",
        "oxalate": True,
        "traj": "trajectories/DNA_CaOx_templating_gel_nodna_fire.trj.json",
    },
}

DUPLEX_PAIRS = (("A", "B"), ("C", "D"))
PHASES = ("amorphous", "intermediate", "crystalline")
PHASE_INDEX = {p: i for i, p in enumerate(PHASES)}

# Cube split along the 0–6 diagonal; adjacent cubes share faces consistently.
TETS = (
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
    (0, 5, 1, 6),
)
CORNERS = (
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
)


def r3(v) -> list[float]:
    return [round(float(x), 3) for x in np.asarray(v, float)]


def glycosidic(atoms: list[dict]):
    """C1' = carbon 1.44–1.52 Å from a nitrogen and closest to P."""
    p = next(a["xyz"] for a in atoms if a["element"].upper() == "P")
    ns = [a["xyz"] for a in atoms if a["element"].upper() == "N"]
    cs = [a["xyz"] for a in atoms if a["element"].upper() == "C"]
    hits = []
    for c in cs:
        dn = min(float(np.linalg.norm(c - n)) for n in ns)
        if 1.44 <= dn <= 1.52:
            ng = min(ns, key=lambda n: float(np.linalg.norm(c - n)))
            hits.append((float(np.linalg.norm(c - p)), c, ng))
    if not hits:
        raise RuntimeError("no C1'/N glycosidic pair")
    hits.sort(key=lambda h: h[0])
    _, c1, ng = hits[0]
    c4 = c1
    best = 1e9
    for c in cs:
        d1 = float(np.linalg.norm(c - c1))
        if 2.15 <= d1 <= 2.60:
            dp = float(np.linalg.norm(c - p))
            if dp < best:
                best = dp
                c4 = c
    base_xyz = [a["xyz"] for a in atoms if a["element"].upper() in ("C", "N")]
    base = np.mean(np.stack(base_xyz, 0), axis=0)
    return p, c4, c1, ng, base


def to_helix(p, origin, e_x, axis, e_y):
    rel = np.asarray(p, float) - origin
    return np.array([rel @ e_x, rel @ axis, rel @ e_y], float)


def interp_edge(p0, p1, v0, v1, iso):
    t = (iso - v0) / (v1 - v0 + 1e-15)
    t = float(np.clip(t, 0.0, 1.0))
    return (1.0 - t) * p0 + t * p1


def marching_tets(dens, mins, pitch, iso):
    """Return (vertices, triangles) for dens >= iso using marching tetrahedra."""
    nx, ny, nz = dens.shape
    edge_cache: dict[tuple, int] = {}
    verts: list[np.ndarray] = []
    tris: list[tuple[int, int, int]] = []

    def vid(ia, ja, ka, ib, jb, kb):
        key = (ia, ja, ka, ib, jb, kb)
        if key[0] > key[3] or (key[0] == key[3] and key[1] > key[4]) or (
            key[0] == key[3] and key[1] == key[4] and key[2] > key[5]
        ):
            key = (ib, jb, kb, ia, ja, ka)
        if key in edge_cache:
            return edge_cache[key]
        p0 = mins + pitch * np.array([key[0], key[1], key[2]], float)
        p1 = mins + pitch * np.array([key[3], key[4], key[5]], float)
        v0 = dens[key[0], key[1], key[2]]
        v1 = dens[key[3], key[4], key[5]]
        p = interp_edge(p0, p1, v0, v1, iso)
        idx = len(verts)
        verts.append(p)
        edge_cache[key] = idx
        return idx

    def corner_ijk(i, j, k, c):
        dx, dy, dz = CORNERS[c]
        return i + dx, j + dy, k + dz

    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                vals = []
                inside = []
                for c in range(8):
                    ii, jj, kk = corner_ijk(i, j, k, c)
                    v = dens[ii, jj, kk]
                    vals.append(v)
                    inside.append(v >= iso)
                if all(inside) or not any(inside):
                    continue
                for tet in TETS:
                    bits = [inside[t] for t in tet]
                    n_in = sum(bits)
                    if n_in == 0 or n_in == 4:
                        continue
                    pts_idx = []
                    for a in range(4):
                        for b in range(a + 1, 4):
                            if bits[a] != bits[b]:
                                ca, cb = tet[a], tet[b]
                                ia = corner_ijk(i, j, k, ca)
                                ib = corner_ijk(i, j, k, cb)
                                pts_idx.append(vid(*ia, *ib))
                    if n_in in (1, 3) and len(pts_idx) == 3:
                        if n_in == 3:
                            pts_idx = [pts_idx[0], pts_idx[2], pts_idx[1]]
                        tris.append((pts_idx[0], pts_idx[1], pts_idx[2]))
                    elif n_in == 2 and len(pts_idx) == 4:
                        tris.append((pts_idx[0], pts_idx[1], pts_idx[2]))
                        tris.append((pts_idx[0], pts_idx[2], pts_idx[3]))

    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    return np.vstack(verts), np.array(tris, dtype=int)


ENVELOPE_PARAMS = {
    "amorphous": dict(pitch=2.8, sigma=1.2, iso_frac=0.28),
    "intermediate": dict(pitch=3.8, sigma=1.6, iso_frac=0.22),
    "crystalline": dict(pitch=2.0, sigma=1.1, iso_frac=0.32),
    "nucleation": dict(pitch=2.2, sigma=1.1, iso_frac=0.32),
    # Tight iso so occupancy is a coat, not a blob through the grooves.
    "shell": dict(pitch=2.6, sigma=1.15, iso_frac=0.28),
}


def _bin_hot_frac(x: np.ndarray, hotspot: np.ndarray, edges: np.ndarray) -> list[dict]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        n = int(m.sum())
        n_hot = int((m & hotspot).sum())
        out.append(
            {
                "lo": round(float(lo), 1),
                "hi": round(float(hi), 1),
                "mid": round(0.5 * float(lo + hi), 1),
                "n": n,
                "nHot": n_hot,
                "frac": round(n_hot / n, 3) if n else 0.0,
            }
        )
    return out


def nucleation_where_summary(
    d_p: np.ndarray,
    r_axis: np.ndarray,
    r_phosphate: float,
    hotspot: np.ndarray,
    phi: np.ndarray,
    n_p: int,
    n_p_with_hot: int,
    pair_corr: np.ndarray,
    score: np.ndarray,
    phase: np.ndarray,
    hit_384: np.ndarray,
    hit_629: np.ndarray,
    n_clusters: int,
) -> dict:
    """Summarize nucleation sites and local COM symmetry — not COM lattice positioning."""
    near = d_p < 8.0
    room = (d_p >= 12.0) & (d_p < 20.0)
    far = d_p >= 20.0
    inside = r_axis < (r_phosphate - 0.35)
    dr = r_axis - r_phosphate
    dp_bins = _bin_hot_frac(d_p, hotspot, np.arange(0.0, 32.1, 2.0))
    rad_bins = _bin_hot_frac(dr, hotspot, np.arange(-6.0, 24.1, 2.0))
    n_hot = int(hotspot.sum())
    inter = phase == PHASE_INDEX["intermediate"]

    def pack(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        n_hot_m = int((mask & hotspot).sum())
        return {
            "n": n,
            "nHot": n_hot_m,
            "frac": round(n_hot_m / n, 3) if n else 0.0,
            "medianDp": round(float(np.median(d_p[mask])), 1) if n else None,
        }

    def med(arr: np.ndarray, mask: np.ndarray) -> float | None:
        sub = arr[mask]
        return round(float(np.median(sub)), 2) if len(sub) else None

    hot_pc = med(pair_corr, hotspot)
    all_pc = med(pair_corr, np.ones(len(d_p), bool))
    hot_sc = med(score, hotspot)
    all_sc = med(score, np.ones(len(d_p), bool))
    hot_pc_mean = (
        round(float(pair_corr[hotspot].mean()), 2) if hotspot.any() else None
    )
    sym_line = (
        f"mean pair-corr {hot_pc_mean} vs {round(float(pair_corr.mean()), 2)} for all Ca"
        if hot_pc_mean is not None
        else "elevated COM pair-corr"
    )
    if hot_sc and all_sc and (hot_sc > 0 or all_sc > 0):
        sym_line += f"; crystallinity {hot_sc} vs {all_sc}"
    n_hot_near = int((hotspot & near).sum())
    n_hot_far = int((hotspot & ~near).sum())
    n_384 = int((hotspot & hit_384).sum())
    n_629 = int((hotspot & hit_629).sum())
    n_both = int((hotspot & hit_384 & hit_629).sum())
    n_inter_hot = int((hotspot & inter).sum())
    n_inter_all = int(inter.sum())

    answers = [
        (
            f"Symmetry pockets: {n_hot} Ca in {n_clusters} clusters with elevated COM pair-corr "
            f"({sym_line}). Hotspots are local nucleation sites, not a uniform ordered coat."
        ),
        (
            f"Shell growth from P-tethered gel: {n_hot_near}/{n_hot} hotspots are phosphate-bound "
            f"(d(P)<8 Å); {n_hot_far} sit in the outer gel. "
            f"Median hotspot d(P) {med(d_p, hotspot)} Å vs {med(d_p, np.ones(len(d_p), bool))} Å for all Ca. "
            f"{int((inside & hotspot).sum())} groove-interior Ca flagged — order nucleates on the gel shell, "
            "not packed in the phosphate cylinder."
        ),
        (
            f"Incipient COM registry: at hotspots, {n_384}/{n_hot} have a 3.84 Å neighbor, "
            f"{n_629}/{n_hot} at 6.29 Å, {n_both}/{n_hot} both (whewellite edge-share + a-chain). "
            f"{n_inter_hot}/{n_hot} hotspots are intermediate-phase ({n_inter_all} total). "
            f"{n_p_with_hot}/{n_p} phosphates within 8 Å of a hotspot — partial patches, not a locked P-register."
        ),
    ]
    return {
        "rPhosphate": round(float(r_phosphate), 2),
        "comA": COM_A,
        "nInside": int(inside.sum()),
        "nInsideHot": int((inside & hotspot).sum()),
        "nHot": n_hot,
        "nClusters": n_clusters,
        "medianPairCorrHot": hot_pc,
        "medianPairCorrAll": all_pc,
        "medianScoreHot": hot_sc,
        "medianScoreAll": all_sc,
        "medianDpAll": round(float(np.median(d_p)), 1) if len(d_p) else None,
        "medianDpHot": round(float(np.median(d_p[hotspot])), 1) if hotspot.any() else None,
        "medianRHot": round(float(np.median(r_axis[hotspot])), 1) if hotspot.any() else None,
        "nearP": pack(near),
        "room": pack(room),
        "far": pack(far),
        "dpBins": dp_bins,
        "radialBins": rad_bins,
        "nP": n_p,
        "nPWithHot8": n_p_with_hot,
        "answers": answers,
    }


def hotspot_clusters_for_viewer(
    hx_ca: np.ndarray,
    hotspot: np.ndarray,
    pair_corr: np.ndarray,
    d_p: np.ndarray,
) -> list[dict]:
    """Spatial clusters among symmetry-flagged hotspot Ca (for viewer rings)."""
    hot_ix = np.where(hotspot)[0]
    if len(hot_ix) == 0:
        return []
    sub_pts = hx_ca[hot_ix]
    clusters_out: list[dict] = []
    for cl in cluster_local(sub_pts, cutoff=HOTSPOT_CLUSTER_CUT, min_size=2):
        core = hot_ix[np.asarray(cl, int)]
        cent = hx_ca[core].mean(axis=0)
        clusters_out.append(
            {
                "center": [round(float(x), 2) for x in cent],
                "radius": round(
                    float(np.linalg.norm(hx_ca[core] - cent, axis=1).max()) + 1.8, 2
                ),
                "n": int(len(core)),
                "meanPairCorr": round(float(pair_corr[core].mean()), 3),
                "meanDP": round(float(d_p[core].mean()), 1),
            }
        )
    clusters_out.sort(key=lambda c: (-c["meanPairCorr"], c["meanDP"]))
    return clusters_out


def _cluster_bounding_sphere(pts: np.ndarray, margin: float = 2.0) -> tuple[np.ndarray, float]:
    cen = pts.mean(axis=0)
    rad = float(np.linalg.norm(pts - cen, axis=1).max()) + margin
    return cen, rad


def merge_overlapping_hotspot_groups(
    groups: list[np.ndarray],
    hx_ca: np.ndarray,
    envelope_bleed: float,
) -> list[np.ndarray]:
    """Union-find merge hotspot groups whose occupancy envelopes would overlap."""
    if len(groups) <= 1:
        return groups
    centers: list[np.ndarray] = []
    radii: list[float] = []
    for g in groups:
        cen, rad = _cluster_bounding_sphere(hx_ca[g])
        centers.append(cen)
        radii.append(rad)

    n = len(groups)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi

    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(centers[i] - centers[j]))
            if d <= radii[i] + radii[j] + envelope_bleed:
                union(i, j)

    merged: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        merged.setdefault(root, []).extend(groups[i].tolist())
    return [np.unique(np.asarray(ix, int)) for ix in merged.values()]


def hotspot_groups_for_envelope(hx_ca: np.ndarray, hotspot: np.ndarray) -> list[np.ndarray]:
    """Spatial hotspot Ca groups; singleton hotspots kept as one-point groups."""
    hot_ix = np.where(hotspot)[0]
    if len(hot_ix) == 0:
        return []
    sub_pts = hx_ca[hot_ix]
    clustered_local: set[int] = set()
    groups: list[np.ndarray] = []
    for cl in cluster_local(sub_pts, cutoff=HOTSPOT_CLUSTER_CUT, min_size=2):
        local = np.asarray(cl, int)
        clustered_local.update(local.tolist())
        groups.append(hot_ix[local])
    for local_i in range(len(hot_ix)):
        if local_i not in clustered_local:
            groups.append(hot_ix[local_i : local_i + 1])
    nucleation = ENVELOPE_PARAMS["nucleation"]
    bleed = float(nucleation["sigma"] * nucleation["pitch"] * 1.5)
    return merge_overlapping_hotspot_groups(groups, hx_ca, bleed)


def concat_envelopes(env_list: list[dict]) -> dict:
    verts: list[list[float]] = []
    indices: list[int] = []
    base = 0
    for env in env_list:
        if not env.get("vertices"):
            continue
        verts.extend(env["vertices"])
        for idx in env["indices"]:
            indices.append(int(idx) + base)
        base += len(env["vertices"])
    return {"vertices": verts, "indices": indices}


def nucleation_envelope_merged(hx_ca: np.ndarray, hotspot: np.ndarray) -> dict:
    """One occupancy shell per merged hotspot patch (overlapping clusters combined)."""
    groups = hotspot_groups_for_envelope(hx_ca, hotspot)
    if not groups:
        return {"vertices": [], "indices": []}
    params = ENVELOPE_PARAMS["nucleation"]
    env_list = []
    for g in groups:
        pts = hx_ca[g]
        if len(pts) < 1:
            continue
        env = phase_envelope(pts, **params)
        if env["vertices"]:
            env_list.append(env)
    if not env_list:
        return {"vertices": [], "indices": []}
    if len(env_list) == 1:
        return env_list[0]
    return concat_envelopes(env_list)


def clip_envelope_outside_cylinder(env: dict, r_min: float) -> dict:
    """Remove envelope triangles that cross inside the DNA phosphate cylinder."""
    if not env.get("vertices") or r_min <= 0:
        return env
    verts = np.asarray(env["vertices"], float)
    tris = np.asarray(env["indices"], int)
    if tris.size == 0:
        return env
    if tris.ndim == 1:
        tris = tris.reshape(-1, 3)
    inside = np.hypot(verts[:, 0], verts[:, 2]) < r_min
    keep = [
        [int(a), int(b), int(c)]
        for a, b, c in tris
        if not (inside[a] or inside[b] or inside[c])
    ]
    if not keep:
        return {"vertices": [], "indices": []}
    keep_tris = np.asarray(keep, int)
    used = np.unique(keep_tris.ravel())
    remap = {int(old): new for new, old in enumerate(used)}
    new_verts = verts[used]
    new_tris = np.array([[remap[a], remap[b], remap[c]] for a, b, c in keep_tris], int)
    return {
        "vertices": [[round(float(x), 2) for x in v] for v in new_verts],
        "indices": new_tris.reshape(-1).tolist(),
    }


def envelope_dna_r_min(r_phosphate: float, sigma: float, pitch: float) -> float:
    """Minimum axis radius for occupancy envelopes — keeps shell outside DNA."""
    if not np.isfinite(r_phosphate) or r_phosphate <= 0:
        return 0.0
    bleed = float(sigma * pitch)
    return float(r_phosphate) + max(0.5, bleed * 0.4)


def phase_envelope(
    pts,
    pitch=2.8,
    sigma=1.5,
    iso_frac=0.2,
    y_clip=None,
    r_min=None,
):
    if len(pts) < 4:
        return {"vertices": [], "indices": []}
    mins = pts.min(axis=0) - 3.0 * pitch
    maxs = pts.max(axis=0) + 3.0 * pitch
    if y_clip is not None:
        mins[1] = min(float(mins[1]), float(y_clip[0]))
        maxs[1] = max(float(maxs[1]), float(y_clip[1]))
    shape = np.ceil((maxs - mins) / pitch).astype(int) + 1
    dens = np.zeros(tuple(int(s) for s in shape), float)
    ix = np.clip(((pts - mins) / pitch).astype(int), 0, np.array(shape) - 1)
    np.add.at(dens, (ix[:, 0], ix[:, 1], ix[:, 2]), 1.0)
    dens = gaussian_filter(dens, sigma=sigma)
    if r_min is not None and r_min > 0:
        xs = mins[0] + np.arange(shape[0]) * pitch
        zs = mins[2] + np.arange(shape[2]) * pitch
        inside = np.hypot(xs[:, None, None], zs[None, None, :]) < r_min
        dens = np.where(inside, 0.0, dens)
    peak = float(dens.max())
    if peak <= 0:
        return {"vertices": [], "indices": []}
    verts, tris = marching_tets(dens, mins, pitch, iso_frac * peak)
    if len(tris) == 0:
        return {"vertices": [], "indices": []}
    env = {
        "vertices": [[round(float(x), 2) for x in v] for v in verts],
        "indices": tris.reshape(-1).tolist(),
    }
    if r_min is not None and r_min > 0:
        env = clip_envelope_outside_cylinder(env, r_min)
    return env


def load_dna(atoms):
    by = defaultdict(list)
    for a in atoms:
        if a["resname"] == "NUC":
            by[(a["chain"], a["resseq"])].append(a)
    strands = []
    for chain in sorted({ch for ch, _ in by}):
        residues = []
        for resseq in sorted(r for ch, r in by if ch == chain):
            p, c4, c1, ng, base = glycosidic(by[(chain, resseq)])
            residues.append(
                {
                    "resseq": int(resseq),
                    "P": r3(p),
                    "C4": r3(c4),
                    "C1": r3(c1),
                    "N": r3(ng),
                    "base": r3(base),
                }
            )
        strands.append({"chain": chain, "residues": residues})
    pairs = []
    by_chain = {s["chain"]: s["residues"] for s in strands}
    for ch_a, ch_b in DUPLEX_PAIRS:
        if ch_a not in by_chain or ch_b not in by_chain:
            continue
        bmap = {r["resseq"]: r for r in by_chain[ch_b]}
        for ra in by_chain[ch_a]:
            rb = bmap.get(ra["resseq"])
            if rb is None:
                continue
            d = float(np.linalg.norm(np.array(ra["C1"]) - np.array(rb["C1"])))
            if 9.0 <= d <= 13.0:
                pairs.append({"a": ra["C1"], "b": rb["C1"], "na": ra["N"], "nb": rb["N"]})
    return strands, pairs


def transform_strands(strands, pairs, seeds, origin, e_x, axis, e_y):
    def hx(p):
        return r3(to_helix(p, origin, e_x, axis, e_y))

    out_s = []
    for s in strands:
        res = []
        for r in s["residues"]:
            res.append(
                {
                    "resseq": r["resseq"],
                    "P": hx(r["P"]),
                    "C4": hx(r["C4"]),
                    "C1": hx(r["C1"]),
                    "N": hx(r["N"]),
                    "base": hx(r["base"]),
                }
            )
        out_s.append({"chain": s["chain"], "residues": res})
    out_p = [
        {"a": hx(p["a"]), "b": hx(p["b"]), "na": hx(p["na"]), "nb": hx(p["nb"])}
        for p in pairs
    ]
    out_seeds = [hx(xyz) for xyz in seeds.values()]
    return out_s, out_p, out_seeds


def water_helix(atoms, origin, e_x, axis, e_y):
    """Hydrate OW + extra HOH oxygens in helix frame (not oxalate O1–O4)."""
    xs, ys, zs = [], [], []
    for a in atoms:
        name = (a.get("name") or "").strip().upper()
        res = (a.get("resname") or "").strip().upper()
        el = (a.get("element") or "").strip().upper()
        is_ow = name.startswith("OW")
        is_hoh = res == "HOH" and el.startswith("O")
        if not (is_ow or is_hoh):
            continue
        hx = to_helix(a["xyz"], origin, e_x, axis, e_y)
        xs.append(round(float(hx[0]), 3))
        ys.append(round(float(hx[1]), 3))
        zs.append(round(float(hx[2]), 3))
    return {"x": xs, "y": ys, "z": zs}


def oxalate_segments(atoms, origin, e_x, axis, e_y):
    """C–C and C–O sticks: monomer C2O4 per shell residue + crystal seed patch."""
    to_hx = lambda p: r3(to_helix(p, origin, e_x, axis, e_y))
    segs, offsets = oxalate_segments_by_residue(atoms, to_hx)
    patch_segs = oxalate_segments_crystal_patch(atoms, to_hx)
    if patch_segs:
        segs = segs + patch_segs
    return segs, len(offsets), offsets


def build_model(geom: str) -> dict:
    spec = GEOMETRIES[geom]
    pdb, csv_path = spec["pdb"], spec["csv"]
    print(f"\n=== {geom}: {pdb.name} ===", flush=True)
    atoms, _ = parse_atoms(pdb)
    nuc = [a for a in atoms if a["resname"] == "NUC"]
    ca_pts, _ = load_whw_ca(pdb)
    if nuc:
    origin, axis, zmin, zmax, r_dna = dna_slab_frame(nuc, pad=0.5)
    else:
        origin = ca_pts.mean(axis=0) if len(ca_pts) else np.zeros(3)
        axis = np.array([0.0, 1.0, 0.0])
        zmin, zmax, r_dna = -20.0, 20.0, 12.0
    if spec.get("seeds") is None:
        seeds = {}
    else:
        seed_path = spec.get("seeds", SEED_PDB)
        try:
    seeds = load_growth_seed_positions(seed_path)
        except SystemExit:
            seeds = {i + 1: ca_pts[i] for i in range(min(4, len(ca_pts)))}
    basis_seeds = seeds if seeds else {
        i + 1: ca_pts[i] for i in range(min(4, len(ca_pts)))
    }
    e_x, e_y = helix_basis(origin, axis, basis_seeds)

    strands, pairs = load_dna(atoms)
    strands, pairs, seed_xyz = transform_strands(
        strands, pairs, seeds, origin, e_x, axis, e_y
    )

    rows = list(csv.DictReader(csv_path.open()))
    if len(rows) != len(ca_pts):
        raise SystemExit(f"{geom}: CSV/PDB Ca mismatch: {len(rows)} vs {len(ca_pts)}")

    hx_ca = np.array([to_helix(p, origin, e_x, axis, e_y) for p in ca_pts])
    phase = np.array([PHASE_INDEX[r["phase"]] for r in rows], dtype=int)
    d_p = np.array([float(r["d_p_A"]) for r in rows])
    score = np.array([float(r["crystallinity"]) for r in rows])
    com_reg = np.array([float(r.get("com_registry", 0) or 0) for r in rows])
    pair_corr = np.array([float(r.get("pair_corr", r.get("com_registry", 0)) or 0) for r in rows])
    hit_384 = np.array([int(r.get("hit_384", 0) or 0) for r in rows], dtype=bool)
    hit_629 = np.array([int(r.get("hit_629", 0) or 0) for r in rows], dtype=bool)
    hotspot = np.array([int(r.get("hotspot", 0) or 0) for r in rows], dtype=bool)
    hotspot_shell = hotspot & (d_p >= DNA_HOTSPOT_DP_MAX)
    hotspot = hotspot & (d_p < DNA_HOTSPOT_DP_MAX)
    dna_heavy = load_dna_heavy(pdb)
    pxyz = load_phosphate_xyz(pdb)
    r_axis = helix_axis_radius(ca_pts, dna_heavy)
    r_phosphate = phosphate_surface_radius(pxyz, dna_heavy)
    if len(dna_heavy) < 3 or not np.isfinite(r_phosphate):
        outward = np.ones(len(ca_pts), bool)
        r_phosphate = 0.0
    else:
    outward = r_axis >= (r_phosphate - 0.35)
    hotspot = hotspot & outward
    radial = np.hypot(hx_ca[:, 0], hx_ca[:, 2])
    phi = np.degrees(np.arctan2(hx_ca[:, 2], hx_ca[:, 0]))
    y_dna = (float(zmin) - 2.0, float(zmax) + 2.0)
    dna_len = (hx_ca[:, 1] >= y_dna[0]) & (hx_ca[:, 1] <= y_dna[1])

    n_p_with_hot = 0
    if len(pxyz) and hotspot.any():
        d_ph = np.linalg.norm(pxyz[:, None, :] - ca_pts[hotspot][None, :, :], axis=2)
        n_p_with_hot = int((d_ph.min(axis=1) < 8.0).sum())
    hotspot_clusters = hotspot_clusters_for_viewer(hx_ca, hotspot, pair_corr, d_p)
    where = nucleation_where_summary(
        d_p,
        r_axis,
        r_phosphate,
        hotspot,
        phi,
        len(pxyz),
        n_p_with_hot,
        pair_corr,
        score,
        phase,
        hit_384,
        hit_629,
        len(hotspot_clusters),
    )

    envelopes = {}
    amorphous_r_min = envelope_dna_r_min(
        r_phosphate,
        ENVELOPE_PARAMS["amorphous"]["sigma"],
        ENVELOPE_PARAMS["amorphous"]["pitch"],
    )
    for name, idx in PHASE_INDEX.items():
        pts = hx_ca[(phase == idx) & outward]
        print(f"envelope {name}: {len(pts)} Ca (outward of P cylinder) ...", flush=True)
        params = dict(ENVELOPE_PARAMS[name])
        if name == "amorphous" and amorphous_r_min > 0:
            params["r_min"] = amorphous_r_min
            print(f"  DNA cylinder clip r_min={amorphous_r_min:.2f} Å", flush=True)
        envelopes[name] = phase_envelope(pts, **params)
        nv = len(envelopes[name]["vertices"])
        nt = len(envelopes[name]["indices"]) // 3
        print(f"  {nv} verts, {nt} tris")
    hot_pts = hx_ca[hotspot]
    print(f"envelope nucleation: {len(hot_pts)} hotspot Ca ...", flush=True)
    hot_groups = hotspot_groups_for_envelope(hx_ca, hotspot)
    print(f"  {len(hot_groups)} merged hotspot patch(es) for envelope", flush=True)
    envelopes["nucleation"] = nucleation_envelope_merged(hx_ca, hotspot)
    nv = len(envelopes["nucleation"]["vertices"])
    nt = len(envelopes["nucleation"]["indices"]) // 3
    print(f"  {nv} verts, {nt} tris")
    shell_pts = hx_ca[outward & dna_len]
    print(
        f"envelope shell: {len(shell_pts)} outward Ca along DNA "
        f"(y {y_dna[0]:.1f}…{y_dna[1]:.1f}) ...",
        flush=True,
    )
    envelopes["shell"] = phase_envelope(shell_pts, **ENVELOPE_PARAMS["shell"])
    nv = len(envelopes["shell"]["vertices"])
    nt = len(envelopes["shell"]["indices"]) // 3
    print(f"  {nv} verts, {nt} tris")

    counts = {name: int((phase == i).sum()) for name, i in PHASE_INDEX.items()}
    counts["nucleationHotspots"] = int(hotspot.sum())
    counts["nucleationHotspotsShell"] = int(hotspot_shell.sum())
    counts["nucleationClusters"] = len(hotspot_clusters)
    counts["insidePhosphate"] = int((~outward).sum())
    print("counts", counts)
    oxalate = []
    oxalate_units = 0
    oxalate_unit_offsets = []
    if spec.get("oxalate", geom == "local"):
        oxalate, oxalate_units, oxalate_unit_offsets = oxalate_segments(
            atoms, origin, e_x, axis, e_y
        )
        print(
            f"oxalate sticks: {len(oxalate)} bonds, {oxalate_units} intact C2O4 units",
            flush=True,
        )
    water = water_helix(atoms, origin, e_x, axis, e_y)
    print(f"water O: {len(water['x'])}", flush=True)
    counts["nWater"] = len(water["x"])
    seed_r = float(spec.get("seedRadius", 30.0))
    r_ca = float(np.max(radial)) if len(radial) else r_dna
    out = {
        "geometry": geom,
        "title": spec["title"],
        "cut": spec["cut"],
        "source": pdb.name,
        "metrics": csv_path.name,
        "seedRadius": seed_r,
        "cutKind": spec.get("cutKind", "spheres"),
        "note": (
            "Helix frame: +Y along DNA. Ca sites colored by COM-net phase. "
            "Oxalate sticks are intact C2O4 from the same PDB. "
            "Alt-P coordinates are rigid-oxalate FIRE then OpenMM; "
            "the local wrap used Cartesian DLS."
        ),
        "helix": {
            "rDna": round(float(r_dna), 3),
            "rPhosphate": round(float(r_phosphate), 3),
            "zmin": round(float(hx_ca[:, 1].min()), 3),
            "zmax": round(float(hx_ca[:, 1].max()), 3),
            "dnaZmin": round(float(zmin), 3),
            "dnaZmax": round(float(zmax), 3),
            "rCoat": round(float(max(r_ca, r_dna + seed_r)), 3),
            "comA": COM_A,
        },
        "strands": strands,
        "pairs": pairs,
        "seeds": seed_xyz,
        "oxalate": oxalate,
        "oxalateUnits": oxalate_units,
        "oxalateUnitOffsets": oxalate_unit_offsets,
        "water": water,
        "ca": {
            "x": [round(float(v), 3) for v in hx_ca[:, 0]],
            "y": [round(float(v), 3) for v in hx_ca[:, 1]],
            "z": [round(float(v), 3) for v in hx_ca[:, 2]],
            "phase": phase.tolist(),
            "dP": [round(float(v), 2) for v in d_p],
            "score": [round(float(v), 3) for v in score],
            "comRegistry": [round(float(v), 3) for v in pair_corr],
            "hotspot": hotspot.astype(int).tolist(),
            "hotspotShell": hotspot_shell.astype(int).tolist(),
            "radial": [round(float(v), 2) for v in radial],
            "phi": [round(float(v), 1) for v in phi],
        },
        "envelopes": envelopes,
        "hotspotClusters": hotspot_clusters,
        "nucleationWhere": where,
        "counts": counts,
        "nCa": int(len(ca_pts)),
        "nDna": int(len(nuc)),
    }
    if spec.get("traj"):
        out["traj"] = spec["traj"]
    return out


def load_existing_models(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    key = "window.DNA_CAOX_MODELS = "
    i = text.find(key)
    if i < 0:
        return {}
    rest = text[i + len(key) :]
    j = rest.find(";\nwindow.DNA_CAOX_MODEL")
    if j < 0:
        j = rest.rfind(";")
    try:
        data = json.loads(rest[:j])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def main():
    ap = argparse.ArgumentParser(description="Export helix-frame JSON for the viewer")
    ap.add_argument(
        "--geom",
        nargs="*",
        default=None,
        help="Only rebuild these geometry keys; merge into existing model-data.js",
    )
    args = ap.parse_args()
    wanted = list(args.geom) if args.geom else list(GEOMETRIES)
    models = load_existing_models(OUT) if args.geom else {}
    for geom in wanted:
        if geom not in GEOMETRIES:
            print(f"skip unknown geom {geom}", flush=True)
            continue
        spec = GEOMETRIES[geom]
        if not spec["pdb"].exists() or not spec["csv"].exists():
            print(f"skip {geom}: missing {spec['pdb'].name} or metrics CSV", flush=True)
            continue
        models[geom] = build_model(geom)
    if not models:
        raise SystemExit("No geometries exported")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(models, separators=(",", ":"))
    default = DEFAULT_GEOM if DEFAULT_GEOM in models else next(iter(models))
    exported_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    OUT.write_text(
        "window.DNA_CAOX_MODELS = "
        + payload
        + f";\nwindow.DNA_CAOX_MODEL = window.DNA_CAOX_MODELS.{default};\n"
        + f'window.DNA_CAOX_EXPORTED_AT = "{exported_at}";\n',
        encoding="utf-8",
    )
    print(f"\nWrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
