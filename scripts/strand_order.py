#!/usr/bin/env python3
"""
Hypothesis observables for DNA templating of COM — not bulk pair-corr.

  * Sequential Ca–Ca along each phosphate strand (~6.3 Å if P templates a)
  * 3.84 Å contacts among phosphate-bound Ca (COM edge-share)
  * Oxalate oxygens bridging two DNA-bound Ca
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

COM_384 = 3.843
COM_A = 6.290
WIN_384 = 0.25
WIN_A = 0.40
P_BOUND = 8.0  # Å — Ca assigned to a phosphate
OX_CA = 2.70  # Å — Ca–O for an oxalate bridge


def load_phosphates_labeled(path: Path) -> list[dict]:
    out = []
    for line in Path(path).open():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[17:20].strip() != "NUC":
            continue
        name = line[12:16].strip().upper()
        el = (line[76:78].strip() or name[:1]).upper()
        if name != "P" and el != "P":
            continue
        out.append(
            {
                "chain": line[21],
                "resseq": int(line[22:26]),
                "xyz": np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                ),
            }
        )
    return out


def strand_groups(phosphates: list[dict]) -> list[list[int]]:
    """Indices into `phosphates`, one list per chain sorted 5'→3' by resseq."""
    by: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(phosphates):
        by[p["chain"]].append(i)
    strands = []
    for ch in sorted(by):
        ix = sorted(by[ch], key=lambda i: phosphates[i]["resseq"])
        if len(ix) >= 2:
            strands.append(ix)
    return strands


def oxalate_bridge_count(pdb_path: Path, ca_pts: np.ndarray, p_bound: np.ndarray) -> int:
    """Count C2O4 residues with O near two different phosphate-bound Ca."""
    if not np.any(p_bound) or len(ca_pts) < 2:
        return 0
    units: dict[tuple, list[np.ndarray]] = defaultdict(list)
    for line in Path(pdb_path).open():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        res = line[17:20].strip()
        if res not in ("COM", "WHW"):
            continue
        name = line[12:16].strip().upper()
        el = (line[76:78].strip() or name[:1]).upper()
        if not (el.startswith("O") or name.startswith("O")):
            continue
        key = (line[21], int(line[22:26]), res)
        xyz = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        )
        units[key].append(xyz)
    hot = ca_pts[p_bound]
    n_br = 0
    for olist in units.values():
        if len(olist) < 2:
            continue
        oxyz = np.array(olist)
        d = np.linalg.norm(hot[:, None, :] - oxyz[None, :, :], axis=2)
        near = np.any(d <= OX_CA, axis=1)
        if int(near.sum()) >= 2:
            n_br += 1
    return n_br


def strand_order_metrics(
    pdb_path: Path,
    ca_pts: np.ndarray,
    d_p: np.ndarray | None = None,
) -> dict:
    """Per-Ca strand labels plus sequential / 3.84 / 6.29 summaries."""
    n = len(ca_pts)
    phosphates = load_phosphates_labeled(pdb_path)
    strand_id = np.full(n, -1, int)
    seq_index = np.full(n, -1, int)
    seq_next_d = np.full(n, np.nan)
    hit_384 = np.zeros(n, int)
    hit_629 = np.zeros(n, int)
    p_bound = np.zeros(n, bool)

    empty = {
        "n_p_ca": 0,
        "n_p": len(phosphates),
        "seq_d": np.array([]),
        "n_seq": 0,
        "n_seq_384": 0,
        "n_seq_629": 0,
        "frac_seq_384": 0.0,
        "frac_seq_629": 0.0,
        "median_seq": None,
        "n_pair_384": 0,
        "n_pair_629": 0,
        "n_ox_bridge": 0,
        "strand_id": strand_id,
        "seq_index": seq_index,
        "seq_next_d": seq_next_d,
        "hit_384": hit_384,
        "hit_629": hit_629,
        "p_bound": p_bound,
    }
    if n == 0 or not phosphates:
        return empty

    pxyz = np.array([p["xyz"] for p in phosphates])
    if d_p is None or len(d_p) != n:
        d_p = np.linalg.norm(pxyz[:, None, :] - ca_pts[None, :, :], axis=2).min(axis=0)
    p_bound = d_p < P_BOUND
    # Nearest phosphate index for each Ca.
    dmat = np.linalg.norm(ca_pts[:, None, :] - pxyz[None, :, :], axis=2)
    nearest_p = dmat.argmin(axis=1)

    # One Ca per phosphate (closest among those bound to that P).
    p_to_ca: dict[int, int] = {}
    for i in np.where(p_bound)[0]:
        pi = int(nearest_p[i])
        if pi not in p_to_ca:
            p_to_ca[pi] = int(i)
        elif d_p[i] < d_p[p_to_ca[pi]]:
            p_to_ca[pi] = int(i)

    strands = strand_groups(phosphates)
    seq_d = []
    for sid, pix in enumerate(strands):
        ordered_ca = []
        for k, pi in enumerate(pix):
            if pi not in p_to_ca:
                continue
            ci = p_to_ca[pi]
            strand_id[ci] = sid
            seq_index[ci] = len(ordered_ca)
            ordered_ca.append(ci)
        for a, b in zip(ordered_ca, ordered_ca[1:]):
            d = float(np.linalg.norm(ca_pts[a] - ca_pts[b]))
            seq_d.append(d)
            seq_next_d[a] = d

    seq_d = np.array(seq_d, float)
    p_ix = np.where(p_bound)[0]
    n_pair_384 = n_pair_629 = 0
    if len(p_ix) >= 2:
        sub = ca_pts[p_ix]
        d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
        iu = np.triu_indices(len(p_ix), k=1)
        dd = d[iu]
        n_pair_384 = int(np.sum(np.abs(dd - COM_384) <= WIN_384))
        n_pair_629 = int(np.sum(np.abs(dd - COM_A) <= WIN_A))
        for k, i in enumerate(p_ix):
            row = d[k]
            row[k] = 1e9
            if np.any(np.abs(row - COM_384) <= WIN_384):
                hit_384[i] = 1
            if np.any(np.abs(row - COM_A) <= WIN_A):
                hit_629[i] = 1

    n_seq_384 = int(np.sum(np.abs(seq_d - COM_384) <= WIN_384)) if len(seq_d) else 0
    n_seq_629 = int(np.sum(np.abs(seq_d - COM_A) <= WIN_A)) if len(seq_d) else 0
    n_br = oxalate_bridge_count(pdb_path, ca_pts, p_bound)
    return {
        "n_p_ca": int(p_bound.sum()),
        "n_p": len(phosphates),
        "seq_d": seq_d,
        "n_seq": int(len(seq_d)),
        "n_seq_384": n_seq_384,
        "n_seq_629": n_seq_629,
        "frac_seq_384": n_seq_384 / len(seq_d) if len(seq_d) else 0.0,
        "frac_seq_629": n_seq_629 / len(seq_d) if len(seq_d) else 0.0,
        "median_seq": float(np.median(seq_d)) if len(seq_d) else None,
        "n_pair_384": n_pair_384,
        "n_pair_629": n_pair_629,
        "n_ox_bridge": n_br,
        "strand_id": strand_id,
        "seq_index": seq_index,
        "seq_next_d": seq_next_d,
        "hit_384": hit_384,
        "hit_629": hit_629,
        "p_bound": p_bound,
    }


def format_strand_report(sm: dict) -> list[str]:
    lines = [
        "Strand / phosphate-layer order (hypothesis test)",
        f"  Phosphate-bound Ca (d(P)<{P_BOUND:.0f} Å): {sm['n_p_ca']} / {sm['n_p']} P",
        f"  Sequential Ca–Ca along strands: n={sm['n_seq']}"
        + (
            f"  median={sm['median_seq']:.2f} Å"
            if sm["median_seq"] is not None
            else ""
        ),
        f"    within {COM_A:.2f}±{WIN_A:.2f} Å (COM a / P-P): "
        f"{sm['n_seq_629']} ({100 * sm['frac_seq_629']:.0f}%)",
        f"    within {COM_384:.3f}±{WIN_384:.2f} Å (edge-share): "
        f"{sm['n_seq_384']} ({100 * sm['frac_seq_384']:.0f}%)",
        f"  Any P-bound Ca–Ca pair at 3.84 Å: {sm['n_pair_384']}",
        f"  Any P-bound Ca–Ca pair at 6.29 Å: {sm['n_pair_629']}",
        f"  Oxalate units bridging two P-bound Ca: {sm['n_ox_bridge']}",
    ]
    if sm["n_seq"]:
        arr = sm["seq_d"]
        lines.append("  Sequential distances (Å):")
        for d in arr:
            mark = ""
            if abs(d - COM_384) <= WIN_384:
                mark = "  3.84"
            elif abs(d - COM_A) <= WIN_A:
                mark = "  6.29"
            lines.append(f"    {d:6.3f}{mark}")
    return lines
