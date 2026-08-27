"""Shared steric rules: rigid CaOx units, min inter-unit O–O 2 Å.

Ca–Ca floors are split on purpose:

  MIN_CA_CA       — bulk / non-neighbor packing (do not overlap random units)
  MIN_CA_CA_COM   — allow whewellite’s 3.84 Å edge-share (phosphate layer,
                    no-DNA blob). Never push DNA-bound neighbors to 6 Å;
                    that forbids COM at the backbone by construction.
"""

from __future__ import annotations

import numpy as np

MIN_CA_CA = 6.0
MIN_CA_CA_COM = 3.70  # keep COM 3.843; drop unphysical <3.7 Å
MIN_O_O = 2.0
DNA_HEAVY = 2.20


def is_ca(atom) -> bool:
    el = str(atom.get("element", "")).upper()
    name = str(atom.get("name", "")).strip().upper()
    return el == "CA" or name == "CA"


def is_oxygen(atom) -> bool:
    el = str(atom.get("element", "")).upper()
    name = str(atom.get("name", "")).strip().upper()
    return el.startswith("O") or name.startswith("O")


def xyz_of(atoms, pred=None) -> np.ndarray:
    sel = atoms if pred is None else [a for a in atoms if pred(a)]
    if not sel:
        return np.zeros((0, 3), float)
    return np.asarray([a["xyz"] for a in sel], float)


def min_pair(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 1e9
    return float(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2).min())


def ca_clear(ca: np.ndarray, existing: np.ndarray, cutoff: float = MIN_CA_CA) -> bool:
    if len(existing) == 0:
        return True
    return float(np.linalg.norm(existing - ca, axis=1).min()) >= cutoff


def oo_clear(new_o: np.ndarray, existing_o: np.ndarray, cutoff: float = MIN_O_O) -> bool:
    return min_pair(new_o, existing_o) >= cutoff


def separate_ca(points, min_d: float = MIN_CA_CA, niter: int = 60) -> np.ndarray:
    """Push Ca pairs apart until every intermolecular Ca–Ca is ≥ min_d."""
    pts = np.asarray(points, float).copy()
    n = len(pts)
    if n < 2:
        return pts
    floors = np.full((n, n), float(min_d))
    np.fill_diagonal(floors, 0.0)
    return separate_ca_pairwise(pts, floors, niter=niter)


def strand_ca_floor_matrix(
    n: int,
    sequential_pairs: list[tuple[int, int]],
    *,
    strand_min: float = MIN_CA_CA_COM,
    bulk_min: float = MIN_CA_CA,
) -> np.ndarray:
    """Per-pair Ca–Ca floor: COM-allowed along a strand, bulk elsewhere."""
    floors = np.full((n, n), float(bulk_min))
    np.fill_diagonal(floors, 0.0)
    for i, j in sequential_pairs:
        if 0 <= i < n and 0 <= j < n and i != j:
            floors[i, j] = floors[j, i] = float(strand_min)
    return floors


def separate_ca_pairwise(
    points, min_d_matrix: np.ndarray, niter: int = 80
) -> np.ndarray:
    """Push only pairs that violate their own minimum distance."""
    pts = np.asarray(points, float).copy()
    floors = np.asarray(min_d_matrix, float)
    n = len(pts)
    if n < 2:
        return pts
    for _ in range(niter):
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(d, 1e9)
        viol = floors - d
        np.fill_diagonal(viol, -1e9)
        i, j = np.unravel_index(int(np.argmax(viol)), viol.shape)
        if viol[i, j] <= 1e-6:
            break
        delta = pts[j] - pts[i]
        nrm = float(np.linalg.norm(delta))
        if nrm < 1e-8:
            delta = np.array([1.0, 0.0, 0.0])
            nrm = 1.0
        push = (floors[i, j] - nrm) / 2.0 + 0.02
        pts[i] -= push * delta / nrm
        pts[j] += push * delta / nrm
    return pts


def pair_distances(xyz: np.ndarray, cutoff: float) -> np.ndarray:
    if len(xyz) < 2:
        return np.array([])
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    iu = np.triu_indices(len(xyz), k=1)
    dd = d[iu]
    return dd[dd <= cutoff]


def short_contact_summary(atoms, min_ca=MIN_CA_CA, min_oo=MIN_O_O):
    """Inter-residue short contacts (intra-unit O–O ignored)."""
    cas = [a for a in atoms if is_ca(a)]
    oxs = [a for a in atoms if is_oxygen(a)]
    cxyz = xyz_of(cas)
    ca_d = pair_distances(cxyz, min_ca - 1e-6)
    oo_short = 0
    oo_min = 1e9
    if len(oxs) >= 2:
        oxyz = xyz_of(oxs)
        d = np.linalg.norm(oxyz[:, None, :] - oxyz[None, :, :], axis=2)
        n = len(oxs)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = oxs[i], oxs[j]
                if (
                    a.get("chain") == b.get("chain")
                    and a.get("resseq") == b.get("resseq")
                    and a.get("resname") == b.get("resname")
                ):
                    continue
                if d[i, j] < min_oo:
                    oo_short += 1
                    oo_min = min(oo_min, float(d[i, j]))
    return {
        "n_ca": len(cas),
        "n_ca_short": int(len(ca_d)),
        "ca_min": float(ca_d.min()) if len(ca_d) else None,
        "n_oo_short": oo_short,
        "oo_min": None if oo_min >= 1e8 else oo_min,
    }
