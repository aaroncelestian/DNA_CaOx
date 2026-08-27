#!/usr/bin/env python3
"""
Pack the DLS growth model into a compact helix-frame JSON for the 3D viewer.

DNA is reduced to backbone traces (P, C1', glycosidic N). CaOx is reduced to
Ca sites plus a smoothed occupancy envelope per phase — no oxalate/water atoms.
"""

from __future__ import annotations

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
DEFAULT_GEOM = "shell_lattice"

# Sphere = union of 30 Å balls around the four seed Ca (1605 Ca, more crystal).
# The spherical DLS PDB was overwritten by the cylinder; coordinates live in
# the relaxed file. Phases are the strict COM-net labels from that DLS run.
# Slab = DNA-length cylinder (2468 Ca, almost all intermediate).
# AllP = 22 phosphate seeds, 30 Å (end-cap crystal is a merge artifact).
# Local = 22 phosphate seeds, 20 Å shells (~2 c-steps), oxalate/Ca DLS.
GEOMETRIES = {
    "sphere": {
        "pdb": ROOT / "DNA_CaOx_growth_whewellite30A_relaxed.pdb",
        "csv": ROOT
        / "figures/crystallinity/DNA_CaOx_growth_whewellite30A_dls_ca_metrics.csv",
        "title": "Spherical 30 Å (4 seed balloons)",
        "cut": "union of 30 Å spheres around the four COM seed Ca",
        "seedRadius": 30.0,
        "cutKind": "spheres",
    },
    "slab": {
        "pdb": ROOT / "DNA_CaOx_growth_whewellite30A_dls.pdb",
        "csv": ROOT
        / "figures/crystallinity/DNA_CaOx_growth_whewellite30A_slab_dls_ca_metrics.csv",
        "title": "Cylinder (DNA-length coating)",
        "cut": "cylinder along DNA, 30 Å from the duplex envelope",
        "seedRadius": 30.0,
        "cutKind": "cylinder",
    },
    "allp": {
        "pdb": ROOT / "DNA_CaOx_growth_whewellite30A_allP.pdb",
        "csv": ROOT
        / "figures/crystallinity/DNA_CaOx_growth_whewellite30A_allP_ca_metrics.csv",
        "seeds": ROOT / "DNA_CaOx_growth_allP_seeds.pdb",
        "title": "All phosphates (22 seeds, 30 Å)",
        "cut": "one COM seed at every P on both strands; union of 30 Å spheres",
        "seedRadius": 30.0,
        "cutKind": "spheres",
    },
    "local": {
        "pdb": ROOT / "DNA_CaOx_growth_whewellite20A_allP_dls.pdb",
        "csv": ROOT
        / "figures/crystallinity/DNA_CaOx_growth_whewellite20A_allP_dls_ca_metrics.csv",
        "seeds": ROOT / "DNA_CaOx_growth_allP_seeds.pdb",
        "title": "Local wrap (22 seeds, 20 Å, DLS)",
        "cut": "COM at every P; 20 Å spheres (~2 c-steps), then oxalate/Ca DLS",
        "seedRadius": 20.0,
        "cutKind": "spheres",
    },
    "altp": {
        "pdb": ROOT / "DNA_CaOx_growth_whewellite30A_altP_omm.pdb",
        "csv": ROOT
        / "figures/crystallinity/DNA_CaOx_growth_whewellite30A_altP_omm_ca_metrics.csv",
        "seeds": ROOT / "DNA_CaOx_growth_altP_seeds.pdb",
        "title": "Every other P (12 seeds, 30 Å, FIRE/OMM)",
        "cut": "COM at every other P on both strands; 30 Å spheres, then rigid-oxalate FIRE + OpenMM",
        "seedRadius": 30.0,
        "cutKind": "spheres",
        "oxalate": True,
    },
    "gel": {
        "pdb": ROOT / "DNA_CaOx_gel_first_omm.pdb",
        "csv": ROOT / "figures/crystallinity/DNA_CaOx_gel_first_omm_ca_metrics.csv",
        "seeds": ROOT / "DNA_CaOx_gel_first_seeds.pdb",
        "title": "Gel-first (44 P, FIRE/OMM)",
        "cut": "BV OP chelation at every phosphate; random oxalate orientations; FIRE+OpenMM, no COM Ca–Ca targets",
        "seedRadius": 12.0,
        "cutKind": "spheres",
        "oxalate": True,
    },
    "shell15": {
        "pdb": ROOT / "DNA_CaOx_gel_first_shell15A_omm.pdb",
        "csv": ROOT
        / "figures/crystallinity/DNA_CaOx_gel_first_shell15A_omm_ca_metrics.csv",
        "seeds": ROOT / "DNA_CaOx_gel_first_seeds.pdb",
        "title": "Gel + 15 Å shell (FIRE/OMM)",
        "cut": "Frozen gel (44 P) + random CaOx/water 2.25–15 Å from gel; shell relaxed, no COM targets",
        "seedRadius": 15.0,
        "cutKind": "spheres",
        "oxalate": True,
    },
    "gel_altp_geom": {
        "pdb": ROOT / "DNA_CaOx_gel_altP_geom_omm.pdb",
        "csv": ROOT / "figures/crystallinity/DNA_CaOx_gel_altP_geom_omm_ca_metrics.csv",
        "seeds": ROOT / "DNA_CaOx_gel_altP_geom_seeds.pdb",
        "title": "Gel alt-P + geometry (22 P, honest FIRE)",
        "cut": "Every other P; BV Ca; geometry-oriented oxalate; FIRE+OpenMM, no COM targets",
        "seedRadius": 12.0,
        "cutKind": "spheres",
        "oxalate": True,
        "traj": "trajectories/DNA_CaOx_gel_altP_geom_fire.trj.json",
    },
    "templating_gel": {
        "pdb": ROOT / "DNA_CaOx_templating_gel_omm.pdb",
        "csv": ROOT
        / "figures/crystallinity/DNA_CaOx_templating_gel_omm_ca_metrics.csv",
        "seeds": ROOT / "DNA_CaOx_templating_gel_seeds.pdb",
        "title": "Templating gel (all P, honest FIRE)",
        "cut": "CaOx at every P plus a disordered second row and extra waters; FIRE with no COM targets, gel unfrozen",
        "seedRadius": 12.0,
        "cutKind": "spheres",
        "oxalate": True,
    },
    "templating_nodna": {
        "pdb": ROOT / "DNA_CaOx_templating_gel_nodna_omm.pdb",
        "csv": ROOT
        / "figures/crystallinity/DNA_CaOx_templating_gel_nodna_omm_ca_metrics.csv",
        "seeds": ROOT / "DNA_CaOx_templating_gel_seeds.pdb",
        "title": "No-DNA CaOx blob (honest FIRE)",
        "cut": "Same unit count as templating gel, random sphere, no DNA, no COM targets",
        "seedRadius": 18.0,
        "cutKind": "spheres",
        "oxalate": True,
    },
    "shell_lattice": {
        "pdb": ROOT / "DNA_CaOx_gel_altP_geom_shell_lattice_omm.pdb",
        "csv": ROOT
        / "figures/crystallinity/DNA_CaOx_gel_altP_geom_shell_lattice_omm_ca_metrics.csv",
        "seeds": ROOT / "DNA_CaOx_gel_altP_geom_seeds.pdb",
        "title": "Gel + saturated CaOx shell",
        "cut": "Frozen gel; saturated CaOx shell to 30 Å (gel coat → intermediate → pre-crystal → bulk)",
        "seedRadius": 28.0,
        "cutKind": "spheres",
        "oxalate": True,
        "traj": "trajectories/DNA_CaOx_gel_altP_geom_shell_lattice_fire.trj.json",
    },
    "shell_lattice_seeded": {
        "pdb": ROOT / "DNA_CaOx_gel_altP_geom_shell_lattice_seeded_omm.pdb",
        "csv": ROOT
        / "figures/crystallinity/DNA_CaOx_gel_altP_geom_shell_lattice_seeded_omm_ca_metrics.csv",
        "seeds": ROOT / "DNA_CaOx_gel_altP_geom_seeds.pdb",
        "title": "Gel + lattice shell + whewellite seed",
        "cut": "Frozen gel; lattice shell 6–28 Å; authentic whewellite patch at 22–28 Å",
        "seedRadius": 28.0,
        "cutKind": "spheres",
        "oxalate": True,
        "traj": "trajectories/DNA_CaOx_gel_altP_geom_shell_lattice_seeded_fire.trj.json",
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
) -> dict:
    """Answer: backbone vs radial room vs helical nodes (hotspot *rate*, not raw count)."""
    near = (d_p < 8.0)
    room = (d_p >= 12.0) & (d_p < 20.0)
    far = d_p >= 20.0
    inside = r_axis < (r_phosphate - 0.35)
    dr = r_axis - r_phosphate
    dp_bins = _bin_hot_frac(d_p, hotspot, np.arange(0.0, 32.1, 2.0))
    rad_bins = _bin_hot_frac(dr, hotspot, np.arange(-6.0, 24.1, 2.0))
    peak = max((b for b in dp_bins if b["n"] >= 20), key=lambda b: b["frac"], default=None)
    peak_rad = max((b for b in rad_bins if b["n"] >= 20 and b["lo"] >= 0), key=lambda b: b["frac"], default=None)

    def pack(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        n_hot = int((mask & hotspot).sum())
        return {
            "n": n,
            "nHot": n_hot,
            "frac": round(n_hot / n, 3) if n else 0.0,
            "medianDp": round(float(np.median(d_p[mask])), 1) if n else None,
        }

    answers = [
        (
            f"Not along the phosphate backbone: {pack(near)['nHot']}/{pack(near)['n']} "
            f"Ca within 8 Å of P are hotspots ({100 * pack(near)['frac']:.1f}%). "
            f"{int(inside.sum())} Ca sit inside the phosphate cylinder (grooves); "
            f"{int((inside & hotspot).sum())} of those are hotspots."
        ),
        (
            "Needs radial room: hotspot fraction peaks at "
            + (
                f"d(P) {peak['lo']:.0f}–{peak['hi']:.0f} Å ({100 * peak['frac']:.0f}% of Ca there)"
                if peak
                else "larger d(P)"
            )
            + (
                f", ~{peak_rad['lo']:.0f}–{peak_rad['hi']:.0f} Å outside the phosphate surface."
                if peak_rad
                else "."
            )
            + f" COM a = {COM_A:.2f} Å."
        ),
        (
            f"No strong helical nodes: only {n_p_with_hot}/{n_p} phosphates have a hotspot "
            f"within 8 Å. Azimuth of hotspots is scattered, not a repeating P-register."
        ),
    ]
    return {
        "rPhosphate": round(float(r_phosphate), 2),
        "comA": COM_A,
        "nInside": int(inside.sum()),
        "nInsideHot": int((inside & hotspot).sum()),
        "nHot": int(hotspot.sum()),
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


def phase_envelope(pts, pitch=2.8, sigma=1.5, iso_frac=0.2, y_clip=None):
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
    peak = float(dens.max())
    if peak <= 0:
        return {"vertices": [], "indices": []}
    verts, tris = marching_tets(dens, mins, pitch, iso_frac * peak)
    if len(tris) == 0:
        return {"vertices": [], "indices": []}
    return {
        "vertices": [[round(float(x), 2) for x in v] for v in verts],
        "indices": tris.reshape(-1).tolist(),
    }


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
    seed_path = spec.get("seeds", SEED_PDB)
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
    try:
        seeds = load_growth_seed_positions(seed_path)
    except SystemExit:
        seeds = {i + 1: ca_pts[i] for i in range(min(4, len(ca_pts)))}
    e_x, e_y = helix_basis(origin, axis, seeds)

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
    hotspot = np.array([int(r.get("hotspot", 0) or 0) for r in rows], dtype=bool)
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
    where = nucleation_where_summary(
        d_p, r_axis, r_phosphate, hotspot, phi, len(pxyz), n_p_with_hot
    )

    envelopes = {}
    for name, idx in PHASE_INDEX.items():
        pts = hx_ca[(phase == idx) & outward]
        print(f"envelope {name}: {len(pts)} Ca (outward of P cylinder) ...", flush=True)
        envelopes[name] = phase_envelope(pts, **ENVELOPE_PARAMS[name])
        nv = len(envelopes[name]["vertices"])
        nt = len(envelopes[name]["indices"]) // 3
        print(f"  {nv} verts, {nt} tris")
    hot_pts = hx_ca[hotspot]
    print(f"envelope nucleation: {len(hot_pts)} hotspot Ca ...", flush=True)
    envelopes["nucleation"] = phase_envelope(hot_pts, **ENVELOPE_PARAMS["nucleation"])
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
    hotspot_clusters = hotspot_clusters_for_viewer(hx_ca, hotspot, pair_corr, d_p)
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
        "ca": {
            "x": [round(float(v), 3) for v in hx_ca[:, 0]],
            "y": [round(float(v), 3) for v in hx_ca[:, 1]],
            "z": [round(float(v), 3) for v in hx_ca[:, 2]],
            "phase": phase.tolist(),
            "dP": [round(float(v), 2) for v in d_p],
            "score": [round(float(v), 3) for v in score],
            "comRegistry": [round(float(v), 3) for v in pair_corr],
            "hotspot": hotspot.astype(int).tolist(),
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
    OUT.write_text(
        "window.DNA_CAOX_MODELS = "
        + payload
        + f";\nwindow.DNA_CAOX_MODEL = window.DNA_CAOX_MODELS.{default};\n",
        encoding="utf-8",
    )
    print(f"\nWrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
