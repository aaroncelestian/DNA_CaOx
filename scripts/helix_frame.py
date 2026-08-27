"""Helix-frame coordinates for trajectory export (matches viewer model-data)."""

from __future__ import annotations

import numpy as np

from oxalate_sticks import oxalate_segments_by_residue  # noqa: E402


def basis_from_dna(dna_xyz: np.ndarray, seed_xyz: np.ndarray | None = None):
    origin = dna_xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(dna_xyz - origin)
    axis = vt[0] / max(float(np.linalg.norm(vt[0])), 1e-8)
    if seed_xyz is not None and len(seed_xyz):
        rel = np.asarray(seed_xyz, float) - origin
        perp = rel - np.outer(rel @ axis, axis)
        mean_perp = perp.mean(axis=0)
    else:
        mean_perp = np.zeros(3)
    n = float(np.linalg.norm(mean_perp))
    if n < 1e-6:
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(float(tmp @ axis)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        mean_perp = np.cross(axis, tmp)
        n = float(np.linalg.norm(mean_perp))
    e_x = mean_perp / n
    e_y = np.cross(axis, e_x)
    e_y = e_y / max(float(np.linalg.norm(e_y)), 1e-8)
    return origin, axis, e_x, e_y


def to_helix(p, origin, e_x, axis, e_y):
    rel = np.asarray(p, float) - origin
    return [float(rel @ e_x), float(rel @ axis), float(rel @ e_y)]


def oxalate_segments_helix(atoms, origin, e_x, axis, e_y):
    to_hx = lambda p: to_helix(p, origin, e_x, axis, e_y)
    segs, _offsets = oxalate_segments_by_residue(atoms, to_hx)
    return segs


def ca_positions_helix(atoms, ca_idx, origin, e_x, axis, e_y):
    out = {"x": [], "y": [], "z": []}
    for i in ca_idx:
        hx = to_helix(atoms[i]["xyz"], origin, e_x, axis, e_y)
        out["x"].append(hx[0])
        out["y"].append(hx[1])
        out["z"].append(hx[2])
    return out
