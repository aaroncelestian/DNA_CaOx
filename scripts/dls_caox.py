#!/usr/bin/env python3
"""
Distance least-squares (DLS) on CaOx / whewellite atoms only.

Prefer scripts/fire_openmm_caox.py for production (rigid-oxalate FIRE,
then OpenMM minimization). This Cartesian L-BFGS path is kept as the
reference restraint potential.

DNA (and any other non-WHW residue) is copied unchanged. WHW atoms move
to satisfy:
  * intra-oxalate distances (C2O4 geometry from the starting model)
  * Ca–O coordination to the starting ligands
  * one-sided O···O ≥ OO_TARGET (clears CrystalMaker shorts)
  * one-sided Ca···Ca ≥ CA_MIN, plus soft COM Ca–Ca targets
  * weak positional anchors so the lattice does not explode
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from caox_units import find_oxalates_strict  # noqa: E402
from geom_constraints import DNA_HEAVY, is_ca, is_oxygen  # noqa: E402
from grow_whewellite import parse_atoms  # noqa: E402
from relax_whewellite_units import write_pdb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

OO_TARGET = 2.40
OO_CUT = 2.50
CA_MIN = 3.50
CA_CUT = 4.00
COM_CA = (3.843, 6.273, 6.290)
COM_CA_WIN = 0.40
W_OO = 80.0
W_CA = 20.0
W_COM = 6.0
W_GEL_COM = 16.0
W_EPITAX = 22.0
W_INTRA = 80.0
W_CAO = 15.0
W_POS = 8.0
W_DNA = 40.0
MAX_SHIFT = 1.60
MERGE_SAME = 1.15
SEED_CRYSTAL_BFAC = 12.0
EPITAX_RADIUS = 28.0
EPITAX_WIN = 5.0
EPITAX_DECAY_LEN = 6.0
GEL_COM_DECAY_LEN = 8.0
GEL_COM_INNER_FULL = 8.0
SHELL_COM_MIN_DGEL = 14.0


def el_of(a) -> str:
    return str(a.get("element", "")).upper() or a["name"].strip()[0].upper()


def union_find(n: int):
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    return find, union


def query_pairs(xyz: np.ndarray, cutoff: float) -> np.ndarray:
    if len(xyz) < 2:
        return np.zeros((0, 2), dtype=int)
    tree = cKDTree(xyz)
    raw = tree.query_pairs(cutoff)
    if len(raw) == 0:
        return np.zeros((0, 2), dtype=int)
    if isinstance(raw, np.ndarray):
        return raw
    return np.array(list(raw), dtype=int)


def merge_duplicate_atoms(atoms: list[dict], cutoff: float) -> list[dict]:
    """Average same-element WHW atoms closer than cutoff (overlapping images)."""
    if cutoff <= 0 or len(atoms) < 2:
        return atoms
    xyz = np.array([a["xyz"] for a in atoms], float)
    els = np.array([el_of(a) for a in atoms])
    find, union = union_find(len(atoms))
    for el in np.unique(els):
        idx = np.where(els == el)[0]
        if len(idx) < 2:
            continue
        pairs = query_pairs(xyz[idx], cutoff)
        for a, b in pairs:
            union(int(idx[a]), int(idx[b]))
    groups = defaultdict(list)
    for i in range(len(atoms)):
        groups[find(i)].append(i)
    out = []
    for members in groups.values():
        a = dict(atoms[members[0]])
        if len(members) > 1:
            a["xyz"] = np.mean([atoms[k]["xyz"] for k in members], axis=0)
        out.append(a)
    return out


def find_oxalates(atoms: list[dict]):
    """Intact C2O4 only (residue groups or exclusive C–C pairing)."""
    return find_oxalates_strict(atoms)


def pair_terms(xyz, i, j, d0, w, e, g):
    rij = xyz[i] - xyz[j]
    d = np.linalg.norm(rij, axis=1)
    ok = d > 1e-8
    i, j, rij, d = i[ok], j[ok], rij[ok], d[ok]
    if d0.ndim == 0 or d0.shape != d.shape:
        diff = d - float(d0)
    else:
        diff = d - d0[ok]
    e[0] += w * float(np.square(diff).sum())
    fac = (2.0 * w * diff / d)[:, None] * rij
    np.add.at(g, i, fac)
    np.add.at(g, j, -fac)


def pair_terms_weighted(xyz, i, j, d0, w_arr, e, g):
    """Harmonic distance springs with per-pair weights."""
    if len(i) == 0:
        return
    rij = xyz[i] - xyz[j]
    d = np.linalg.norm(rij, axis=1)
    ok = d > 1e-8
    i, j, rij, d, w_arr = i[ok], j[ok], rij[ok], d[ok], w_arr[ok]
    diff = d - d0[ok]
    e[0] += float(np.sum(w_arr * np.square(diff)))
    fac = (2.0 * w_arr * diff / d)[:, None] * rij
    np.add.at(g, i, fac)
    np.add.at(g, j, -fac)


def is_seed_atom(atom) -> bool:
    return abs(float(atom.get("bfac", 0.0)) - SEED_CRYSTAL_BFAC) < 0.5


def seed_ca_indices(atoms, ca_idx) -> np.ndarray:
    return np.array(
        [int(i) for i in ca_idx if is_seed_atom(atoms[int(i)])],
        int,
    )


def epitax_zone_mask(
    xyz0: np.ndarray,
    atoms,
    seed_ca: np.ndarray,
    epitax_radius: float,
    com_min_resseq: int,
) -> np.ndarray:
    """Shell atoms within epitax_radius of any seed Ca (excludes gel + seed)."""
    mask = np.zeros(len(atoms), bool)
    if len(seed_ca) == 0:
        return mask
    seed_xyz = xyz0[seed_ca]
    tree = cKDTree(seed_xyz)
    for i, a in enumerate(atoms):
        if int(a.get("resseq", 0)) <= com_min_resseq:
            continue
        if is_seed_atom(a):
            continue
        if float(tree.query(xyz0[i])[0]) <= epitax_radius:
            mask[i] = True
    return mask


def build_epitax_com_pairs(
    xyz0: np.ndarray,
    shell_ca: list[int],
    seed_ca: np.ndarray,
    epitax_radius: float,
    epitax_win: float,
    w_epitax: float,
    decay_len: float,
    dna_xyz: np.ndarray | None = None,
    outward_only: bool = False,
    outward_margin: float = 1.5,
):
    """
    Seed-local COM pulls: shell Ca within epitax_radius of a seed Ca are
    attracted to whewellite COM spacings (3.84 / 6.27 / 6.29 Å) vs the
    nearest seed Ca. Weight decays with distance from the seed cloud.
    """
    if len(shell_ca) == 0 or len(seed_ca) == 0:
        return (
            np.zeros(0, int),
            np.zeros(0, int),
            np.zeros(0, float),
            np.zeros(0, float),
        )
    seed_xyz = xyz0[seed_ca]
    tree = cKDTree(seed_xyz)
    dna_tree = cKDTree(dna_xyz) if outward_only and len(dna_xyz) else None
    epi_i, epi_j, epi_t, epi_w = [], [], [], []
    for si in shell_ca:
        d_near, _ = tree.query(xyz0[si])
        if float(d_near) > epitax_radius:
            continue
        d_dna_si = (
            float(dna_tree.query(xyz0[si])[0]) if dna_tree is not None else 0.0
        )
        decay = math.exp(-max(0.0, float(d_near) - 3.84) / max(decay_len, 0.5))
        for sj in seed_ca:
            sj = int(sj)
            if dna_tree is not None and d_dna_si < float(dna_tree.query(xyz0[sj])[0]) - outward_margin:
                continue
            d = float(np.linalg.norm(xyz0[si] - xyz0[sj]))
            if d < 0.5 or d > 11.5:
                continue
            tgt = min(COM_CA, key=lambda t: abs(d - t))
            w = w_epitax * decay
            mismatch = abs(d - tgt)
            if epitax_win > 0:
                w *= max(0.35, 1.0 - mismatch / epitax_win)
            epi_i.append(int(si))
            epi_j.append(sj)
            epi_t.append(float(tgt))
            epi_w.append(float(w))
    return (
        np.array(epi_i, int),
        np.array(epi_j, int),
        np.array(epi_t, float),
        np.array(epi_w, float),
    )


def onesided_terms(xyz, i, j, dmin, w, e, g):
    rij = xyz[i] - xyz[j]
    d = np.linalg.norm(rij, axis=1)
    mask = (d < dmin) & (d > 1e-8)
    if not np.any(mask):
        return
    diff = dmin - d[mask]
    e[0] += w * float(np.square(diff).sum())
    fac = (-2.0 * w * diff / d[mask])[:, None] * rij[mask]
    np.add.at(g, i[mask], fac)
    np.add.at(g, j[mask], -fac)


def inter_oo_stats(xyz, o_idx, gid):
    pairs = query_pairs(xyz[o_idx], 2.0)
    if len(pairs) == 0:
        return 0, None
    a = o_idx[pairs[:, 0]]
    b = o_idx[pairs[:, 1]]
    same = gid[a] == gid[b]
    keep = ~same
    if not np.any(keep):
        return 0, None
    d = np.linalg.norm(xyz[a[keep]] - xyz[b[keep]], axis=1)
    return int(len(d)), float(d.min())


def build_tables(
    atoms,
    oxalates,
    ca_idx,
    water,
    dna_xyz,
    com_min_resseq: int = 0,
    shell_pos_weight: float = 0.18,
    shell_max_shift: float = 4.5,
    seed_epitax: bool = False,
    epitax_radius: float = EPITAX_RADIUS,
    epitax_win: float = EPITAX_WIN,
    w_epitax: float = W_EPITAX,
    epitax_decay_len: float = EPITAX_DECAY_LEN,
    shell_epitax_pos_weight: float = 0.06,
    shell_epitax_max_shift: float = 10.0,
    seed_epitax_outward_only: bool = True,
    gel_outward_com: bool = False,
    w_gel_com: float = W_GEL_COM,
    gel_com_decay_len: float = GEL_COM_DECAY_LEN,
    shell_com_min_dgel: float = SHELL_COM_MIN_DGEL,
):
    xyz0 = np.array([a["xyz"] for a in atoms], float)
    n = len(atoms)
    gid = np.full(n, -1, int)
    intra_i, intra_j, intra_d = [], [], []
    for gi, grp in enumerate(oxalates):
        for i in grp:
            gid[i] = gi
        for a in range(len(grp)):
            for b in range(a + 1, len(grp)):
                i, j = grp[a], grp[b]
                d = float(np.linalg.norm(xyz0[i] - xyz0[j]))
                if 0.5 < d < 4.5:
                    intra_i.append(i)
                    intra_j.append(j)
                    intra_d.append(d)
    w0 = len(oxalates)
    for k, i in enumerate(water):
        gid[i] = w0 + k

    o_idx = np.array([i for i, a in enumerate(atoms) if is_oxygen(a)], int)
    ca = np.array(ca_idx, int)

    cao_i, cao_j, cao_d = [], [], []
    if len(ca) and len(o_idx):
        tree = cKDTree(xyz0[o_idx])
        for i in ca:
            hits = tree.query_ball_point(xyz0[i], 2.80)
            for k in hits:
                d = float(np.linalg.norm(xyz0[i] - xyz0[o_idx[k]]))
                if 1.90 < d < 2.80:
                    cao_i.append(int(i))
                    cao_j.append(int(o_idx[k]))
                    cao_d.append(d)

    com_i, com_j, com_t, com_w, short_i, short_j = [], [], [], [], [], []
    if len(ca) >= 2:
        pairs = query_pairs(xyz0[ca], max(CA_CUT, max(COM_CA) + COM_CA_WIN))
        for a, b in pairs:
            ia, ib = int(ca[a]), int(ca[b])
            d = float(np.linalg.norm(xyz0[ia] - xyz0[ib]))
            if d < CA_CUT:
                short_i.append(ia)
                short_j.append(ib)

        if com_min_resseq > 0:
            shell_atoms = [
                int(i)
                for i in ca_idx
                if int(atoms[int(i)].get("resseq", 0)) > com_min_resseq
            ]
            gel_ca = [
                int(i)
                for i in ca_idx
                if int(atoms[int(i)].get("resseq", 0)) <= com_min_resseq
            ]
            pull_cut = 30.0
            pull_win = 5.5
            gel_tree_ca = cKDTree(xyz0[gel_ca]) if gel_ca else None
            shell_d_gel = {
                int(si): float(gel_tree_ca.query(xyz0[int(si)])[0])
                if gel_tree_ca is not None
                else 0.0
                for si in shell_atoms
            }
            outer_shell = [
                int(si)
                for si in shell_atoms
                if shell_d_gel[int(si)] >= shell_com_min_dgel
            ]
            outer_set = set(outer_shell)
            decay_denom = max(float(gel_com_decay_len), 1e-6)
            for si in shell_atoms:
                si = int(si)
                d_near_gel = shell_d_gel[si]
                if gel_outward_com and gel_ca:
                    w_gel = w_gel_com * math.exp(
                        -max(0.0, d_near_gel - GEL_COM_INNER_FULL) / decay_denom
                    )
                    for gj in gel_ca:
                        d = float(np.linalg.norm(xyz0[si] - xyz0[gj]))
                        if d > pull_cut or d < 0.5:
                            continue
                        tgt = min(COM_CA, key=lambda t: abs(d - t))
                        mismatch = abs(d - tgt)
                        if mismatch <= pull_win or gel_outward_com:
                            com_i.append(si)
                            com_j.append(int(gj))
                            com_t.append(float(tgt))
                            com_w.append(w_gel)
                elif not gel_outward_com:
                    for oi in ca_idx:
                        oi = int(oi)
                        if oi == si:
                            continue
                        d = float(np.linalg.norm(xyz0[si] - xyz0[oi]))
                        if d > pull_cut or d < 0.5:
                            continue
                        tgt = min(COM_CA, key=lambda t: abs(d - t))
                        if abs(d - tgt) <= pull_win:
                            com_i.append(si)
                            com_j.append(oi)
                            com_t.append(float(tgt))
            if gel_outward_com and shell_com_min_dgel > 0 and outer_shell:
                for si in outer_shell:
                    for oi in shell_atoms:
                        oi = int(oi)
                        if oi == si or oi not in outer_set:
                            continue
                        d = float(np.linalg.norm(xyz0[si] - xyz0[oi]))
                        if d > pull_cut or d < 0.5:
                            continue
                        tgt = min(COM_CA, key=lambda t: abs(d - t))
                        if abs(d - tgt) <= pull_win:
                            com_i.append(si)
                            com_j.append(oi)
                            com_t.append(float(tgt))
                            com_w.append(W_COM)
        else:
            for a, b in pairs:
                ia, ib = int(ca[a]), int(ca[b])
                d = float(np.linalg.norm(xyz0[ia] - xyz0[ib]))
                for tgt in COM_CA:
                    if abs(d - tgt) <= COM_CA_WIN:
                        com_i.append(ia)
                        com_j.append(ib)
                        com_t.append(tgt)
                        break

    near_dna = np.array([], int)
    if len(dna_xyz):
        dtree = cKDTree(dna_xyz)
        dmin, _ = dtree.query(xyz0, k=1)
        near_dna = np.where(dmin < 6.0)[0]

    n = len(atoms)
    pos_w = np.ones(n, float)
    max_shift_arr = np.full(n, MAX_SHIFT, float)
    if com_min_resseq > 0:
        for i, a in enumerate(atoms):
            if int(a.get("resseq", 0)) > com_min_resseq:
                pos_w[i] = shell_pos_weight
                bfac = float(a.get("bfac", 20.0))
                if bfac >= 26.0:
                    pos_w[i] *= 0.35
                elif bfac >= 24.0:
                    pos_w[i] *= 0.55
                elif bfac >= 22.0:
                    pos_w[i] *= 0.75
                max_shift_arr[i] = shell_max_shift

    seed_ca = seed_ca_indices(atoms, ca_idx)
    epi_i = epi_j = epi_t = epi_w = np.zeros(0)
    if seed_epitax and len(seed_ca) and com_min_resseq > 0:
        shell_ca = [
            int(i)
            for i in ca_idx
            if int(atoms[int(i)].get("resseq", 0)) > com_min_resseq
            and not is_seed_atom(atoms[int(i)])
        ]
        epi_i, epi_j, epi_t, epi_w = build_epitax_com_pairs(
            xyz0,
            shell_ca,
            seed_ca,
            epitax_radius,
            epitax_win,
            w_epitax,
            epitax_decay_len,
            dna_xyz=dna_xyz,
            outward_only=seed_epitax_outward_only,
        )
        zone = epitax_zone_mask(xyz0, atoms, seed_ca, epitax_radius, com_min_resseq)
        pos_w[zone] = shell_epitax_pos_weight
        max_shift_arr[zone] = shell_epitax_max_shift

    return {
        "xyz0": xyz0,
        "gid": gid,
        "o_idx": o_idx,
        "intra": (np.array(intra_i, int), np.array(intra_j, int), np.array(intra_d, float)),
        "cao": (np.array(cao_i, int), np.array(cao_j, int), np.array(cao_d, float)),
        "com": (
            np.array(com_i, int),
            np.array(com_j, int),
            np.array(com_t, float),
            np.array(com_w, float),
        ),
        "epitax": (epi_i, epi_j, epi_t, epi_w),
        "seed_ca": seed_ca,
        "gel_outward_com": gel_outward_com,
        "w_gel_com": w_gel_com,
        "cashort": (np.array(short_i, int), np.array(short_j, int)),
        "near_dna": near_dna,
        "dna_xyz": dna_xyz,
        "pos_w": pos_w,
        "max_shift": max_shift_arr,
    }


def energy_grad(
    x,
    tbl,
    oo_pairs=None,
    skip_intra=False,
    w_com_scale: float = 1.0,
    w_pos_scale: float = 1.0,
    w_intra_scale: float = 1.0,
):
    xyz = x.reshape(-1, 3)
    g = np.zeros_like(xyz)
    e = [0.0]
    xyz0 = tbl["xyz0"]

    dpos = xyz - xyz0
    pw = tbl.get("pos_w")
    if pw is None:
        pw = np.ones(len(xyz), float)
    e[0] += W_POS * w_pos_scale * float(np.sum(pw[:, None] * np.square(dpos)))
    g += 2.0 * W_POS * w_pos_scale * (pw[:, None] * dpos)

    ii, jj, d0 = tbl["intra"]
    if len(ii) and not skip_intra:
        pair_terms(xyz, ii, jj, d0, W_INTRA * w_intra_scale, e, g)
    ci, cj, cd = tbl["cao"]
    if len(ci):
        pair_terms(xyz, ci, cj, cd, W_CAO, e, g)
    si, sj = tbl["cashort"]
    if len(si):
        onesided_terms(xyz, si, sj, CA_MIN, W_CA, e, g)
    com_data = tbl["com"]
    mi, mj, mt = com_data[0], com_data[1], com_data[2]
    mw = com_data[3] if len(com_data) > 3 else np.array([], float)
    if len(mi) and w_com_scale > 0:
        if len(mw):
            pair_terms_weighted(xyz, mi, mj, mt, mw * w_com_scale, e, g)
        else:
            w_use = (
                float(tbl.get("w_gel_com", W_COM))
                if tbl.get("gel_outward_com")
                else W_COM
            )
            pair_terms(xyz, mi, mj, mt, w_use * w_com_scale, e, g)
    ei, ej, et, ew = tbl.get("epitax", (np.zeros(0, int),) * 4)
    if len(ei) and w_com_scale > 0:
        pair_terms_weighted(xyz, ei, ej, et, ew * w_com_scale, e, g)

    o_idx = tbl["o_idx"]
    gid = tbl["gid"]
    if oo_pairs is not None:
        a, b = oo_pairs
        if len(a):
            onesided_terms(xyz, a, b, OO_TARGET, W_OO, e, g)
    elif len(o_idx) >= 2:
        pairs = query_pairs(xyz[o_idx], OO_CUT)
        if len(pairs):
            a = o_idx[pairs[:, 0]]
            b = o_idx[pairs[:, 1]]
            keep = gid[a] != gid[b]
            if np.any(keep):
                onesided_terms(xyz, a[keep], b[keep], OO_TARGET, W_OO, e, g)

    dna = tbl["dna_xyz"]
    near = tbl["near_dna"]
    if len(dna) and len(near):
        tree = cKDTree(dna)
        dmin, jn = tree.query(xyz[near], k=1)
        hit = dmin < DNA_HEAVY
        if np.any(hit):
            idx = near[hit]
            rij = xyz[idx] - dna[jn[hit]]
            d = dmin[hit]
            diff = DNA_HEAVY - d
            e[0] += W_DNA * float(np.square(diff).sum())
            fac = (-2.0 * W_DNA * diff / d)[:, None] * rij
            # f = -dE/dx; onesided_terms stored +dE/dx in g
            np.add.at(g, idx, fac)

    return e[0], g.ravel()


def run_dls(atoms, dna_xyz, steps: int):
    oxalates, ca_idx, water = find_oxalates(atoms)
    xyz0 = np.array([a["xyz"] for a in atoms], float)
    o_idx = np.array([i for i, a in enumerate(atoms) if is_oxygen(a)], int)
    gid = np.full(len(atoms), -1, int)
    for gi, grp in enumerate(oxalates):
        for i in grp:
            gid[i] = gi
    w0 = len(oxalates)
    for k, i in enumerate(water):
        gid[i] = w0 + k

    n0, min0 = inter_oo_stats(xyz0, o_idx, gid)
    print(
        f"DLS start: {len(oxalates)} oxalate groups, {len(ca_idx)} Ca, "
        f"{len(water)} water O, inter-group O-O < 2.0 Å: n={n0} min={min0}",
        flush=True,
    )

    tbl = build_tables(atoms, oxalates, ca_idx, water, dna_xyz)
    x0 = xyz0.ravel().copy()
    lo = x0 - MAX_SHIFT
    hi = x0 + MAX_SHIFT
    n_eval = [0]

    def fun(x):
        n_eval[0] += 1
        e, g = energy_grad(x, tbl)
        if n_eval[0] == 1 or n_eval[0] % 20 == 0:
            print(f"  eval {n_eval[0]:4d}  E={e:.3e}", flush=True)
        return e, g

    res = minimize(
        fun,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(lo, hi)),
        options={"maxiter": steps, "ftol": 1e-6, "gtol": 1e-4, "maxls": 20},
    )
    xyz = res.x.reshape(-1, 3)
    for i, a in enumerate(atoms):
        a["xyz"] = xyz[i]
    n1, min1 = inter_oo_stats(xyz, o_idx, gid)
    print(
        f"  L-BFGS {res.message}  nit={res.nit}  nfev={res.nfev}  "
        f"O-O<2Å n={n1} min={min1}",
        flush=True,
    )
    return {
        "n_oxalate": len(oxalates),
        "n_ca": len(ca_idx),
        "n_water": len(water),
        "n_oo_before": n0,
        "min_oo_before": min0,
        "n_oo_after": n1,
        "min_oo_after": min1,
        "n_atoms": len(atoms),
        "success": bool(res.success),
        "message": str(res.message),
        "nit": int(res.nit),
    }


def main():
    ap = argparse.ArgumentParser(description="DLS refinement of WHW / CaOx, DNA fixed")
    ap.add_argument(
        "pdb",
        nargs="?",
        default=str(ROOT / "DNA_CaOx_growth_whewellite30A.pdb"),
        type=Path,
    )
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument(
        "--merge", type=float, default=MERGE_SAME, help="same-element merge (Å); 0 disables"
    )
    args = ap.parse_args()

    in_pdb = args.pdb
    out_pdb = args.output
    if out_pdb is None:
        out_pdb = in_pdb.with_name(in_pdb.stem.replace("_relaxed", "") + "_dls.pdb")
        if out_pdb == in_pdb:
            out_pdb = in_pdb.with_name(in_pdb.stem + "_dls.pdb")
    report = out_pdb.with_name(out_pdb.stem + "_report.txt")

    atoms, _ = parse_atoms(in_pdb)
    dna = [a for a in atoms if a["resname"] == "NUC"]
    other = [a for a in atoms if a["resname"] not in ("WHW", "NUC")]
    whw = [dict(a) for a in atoms if a["resname"] == "WHW"]
    n_whw0 = len(whw)
    if args.merge > 0:
        whw = merge_duplicate_atoms(whw, args.merge)
        print(
            f"Merged same-element WHW atoms < {args.merge:.2f} Å: {n_whw0} -> {len(whw)}",
            flush=True,
        )

    dna_xyz = (
        np.array([a["xyz"] for a in dna if el_of(a) != "H"], float)
        if dna
        else np.zeros((0, 3))
    )

    stats = run_dls(whw, dna_xyz, args.steps)

    remarks = [
        "HEADER    DLS-REFINED WHEWELLITE ON GROWTH MODEL\n",
        "TITLE     DISTANCE LEAST-SQUARES ON CAOX ONLY; DNA FIXED\n",
        f"REMARK   1 O-O < 2.0 A: {stats['n_oo_before']} -> {stats['n_oo_after']}\n",
        f"REMARK   1 min O-O: {stats['min_oo_before']} -> {stats['min_oo_after']}\n",
    ]
    write_pdb(out_pdb, dna + other + whw, remarks)

    lines = [
        "Distance least-squares — CaOx / WHW only (DNA fixed)",
        "=" * 62,
        f"Input     : {in_pdb.name}",
        f"Output    : {out_pdb.name}",
        f"WHW atoms : {n_whw0} -> {stats['n_atoms']} (after same-element merge {args.merge} Å)",
        f"Groups    : {stats['n_oxalate']} oxalate (C2O4 DLS), {stats['n_ca']} Ca, {stats['n_water']} water O",
        f"L-BFGS    : nit={stats['nit']}  {stats['message']}",
        "",
        "Restraints:",
        f"  O···O one-sided target {OO_TARGET:.2f} Å  (weight {W_OO})",
        f"  Ca···Ca one-sided min {CA_MIN:.2f} Å, COM targets {COM_CA}",
        "  Intra-oxalate distances from the starting C2O4 geometry",
        f"  Ca–O keep starting coordination",
        f"  Positional anchors ±{MAX_SHIFT:.2f} Å from start",
        "  DNA coordinates unchanged; WHW–DNA exclusion 2.20 Å",
        "",
        "Inter-group O···O < 2.00 Å:",
        f"  before: n={stats['n_oo_before']}  min={stats['min_oo_before']}",
        f"  after : n={stats['n_oo_after']}  min={stats['min_oo_after']}",
        "",
        "Most of the CrystalMaker shorts were overlapping lattice images from",
        "the four DNA-seed orientations. Same-element merge collapses those;",
        "DLS uncrosses the rest under a displacement cap so DNA stays put.",
    ]
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
