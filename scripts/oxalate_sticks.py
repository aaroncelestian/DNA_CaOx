"""Intact C2O4 stick geometry for viewer + trajectories (per WHW residue)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from caox_units import atom_el

CC_MIN, CC_MAX = 1.40, 1.70
CO_MIN, CO_MAX = 1.18, 1.45


def _best_cc_pair(xyz: np.ndarray, carbons: list[int]):
    best = None
    best_d = None
    for i in range(len(carbons)):
        for j in range(i + 1, len(carbons)):
            d = float(np.linalg.norm(xyz[carbons[i]] - xyz[carbons[j]]))
            if CC_MIN <= d <= CC_MAX and (best_d is None or abs(d - 1.54) < abs(best_d - 1.54)):
                best = (carbons[i], carbons[j])
                best_d = d
    return best


def _sticks_for_group(group, xyz, to_hx):
    """Five sticks (C–C + 4× C–O) for one intact oxalate, or []."""
    n = len(group)
    carbons = [i for i in range(n) if atom_el(group[i]) == "C"]
    if len(carbons) < 2:
        return []
    pair = _best_cc_pair(xyz, carbons)
    if pair is None:
        return []
    c1, c2 = pair
    oxygens = [i for i in range(n) if atom_el(group[i]) == "O"]
    used_o: set[int] = set()
    unit = [[to_hx(xyz[c1]), to_hx(xyz[c2])]]
    for ci in (c1, c2):
        ds = sorted(
            (float(np.linalg.norm(xyz[ci] - xyz[oi])), oi)
            for oi in oxygens
            if oi not in used_o
        )
        added = 0
        for d, oi in ds:
            if CO_MIN <= d <= CO_MAX:
                unit.append([to_hx(xyz[ci]), to_hx(xyz[oi])])
                used_o.add(oi)
                added += 1
                if added == 2:
                    break
    return unit if len(unit) == 5 else []


SEED_CRYSTAL_BFAC = 12.0


def oxalate_segments_crystal_patch(
    atoms,
    to_hx,
    bfac_target: float = SEED_CRYSTAL_BFAC,
    bfac_tol: float = 0.5,
):
    """
    C–C and C–O bonds for an embedded whewellite patch (bfac ≈ 12).

    Crystal oxalate is shared across Ca sites, so per-residue C2O4 grouping
  misses most seed ligands; bond by distance within the patch instead.
    """
    patch = [
        a
        for a in atoms
        if a.get("resname") == "WHW"
        and abs(float(a.get("bfac", 0.0)) - bfac_target) < bfac_tol
    ]
    if len(patch) < 4:
        return []
    xyz = np.array([a["xyz"] for a in patch], float)
    els = [atom_el(a) for a in patch]
    segs: list = []
    seen: set[tuple[int, int]] = set()
    for i in range(len(patch)):
        for j in range(i + 1, len(patch)):
            ei, ej = els[i], els[j]
            d = float(np.linalg.norm(xyz[i] - xyz[j]))
            bond = False
            if ei == "C" and ej == "C" and CC_MIN <= d <= CC_MAX:
                bond = True
            elif "C" in (ei, ej) and "O" in (ei, ej) and CO_MIN <= d <= CO_MAX:
                bond = True
            if bond:
                key = (i, j)
                if key not in seen:
                    seen.add(key)
                    segs.append([to_hx(xyz[i]), to_hx(xyz[j])])
    return segs


def oxalate_segments_by_residue(atoms, to_hx):
    """
    Intact C2O4 sticks per WHW residue (skips water-only OW residues).
    Returns (flat segment list, unit_offsets).
    """
    whw = [a for a in atoms if a.get("resname") == "WHW"]
    by_res: dict[tuple, list] = defaultdict(list)
    for a in whw:
        by_res[(a.get("chain", "X"), int(a.get("resseq", 0)))].append(a)

    segs: list = []
    unit_offsets: list[int] = []
    for key in sorted(by_res):
        group = by_res[key]
        els = [atom_el(a) for a in group]
        if sum(e == "C" for e in els) < 2:
            continue
        xyz = np.array([a["xyz"] for a in group], float)
        n0 = len(segs)
        unit_segs = _sticks_for_group(group, xyz, to_hx)
        if len(unit_segs) == 5:
            unit_offsets.append(n0)
            segs.extend(unit_segs)
    return segs, unit_offsets
