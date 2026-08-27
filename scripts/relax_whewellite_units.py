#!/usr/bin/env python3
"""
Relax inter-unit O–O clashes in a whewellite growth model by rigid-body
moves of WHW (CaOx) residues.

Each WHW Ca site is one rigid body (Ca + ligands within bonding distance).
Units rotate about Ca and may translate slightly to clear O–O < MIN_O_O.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geom_constraints import (  # noqa: E402
    DNA_HEAVY,
    MIN_O_O,
    is_ca,
    is_oxygen,
    short_contact_summary,
    xyz_of,
)
from grow_whewellite import format_atom, parse_atoms, rotation_around  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IN_PDB = ROOT / "DNA_CaOx_growth_whewellite10A.pdb"
OUT_PDB = ROOT / "DNA_CaOx_growth_whewellite10A_relaxed.pdb"
REPORT = ROOT / "DNA_CaOx_growth_whewellite10A_relaxed_report.txt"

CA_ANCHOR_W = 3.0
DNA_W = 30.0
OO_W = 80.0
MAX_TRANS = 1.50
LIGAND_CA_C = 4.6
LIGAND_CA_O = 3.3
NEIGHBOR_CA = 9.5
N_ANGLES = 36
MAX_SWEEPS = 20
PAIRWISE_ANGLES = 48
MAX_PASSES = 3
FAST_UNIT_THRESHOLD = 250


def configure_speed(n_units: int, fast: bool):
    global N_ANGLES, MAX_SWEEPS, MAX_PASSES, PAIRWISE_ANGLES, NEIGHBOR_CA
    if fast or n_units >= FAST_UNIT_THRESHOLD:
        N_ANGLES = 12
        MAX_SWEEPS = 6
        MAX_PASSES = 1
        PAIRWISE_ANGLES = 24
        NEIGHBOR_CA = 8.0


def norm(v):
    n = float(np.linalg.norm(v))
    return v * 0.0 if n < 1e-10 else v / n


class Unit:
    def __init__(self, resseq: int, atoms: list[dict]):
        self.resseq = resseq
        self.atoms = atoms
        self.ca_idx = next(i for i, a in enumerate(atoms) if is_ca(a))
        self.ca0 = atoms[self.ca_idx]["xyz"].copy()
        self.o_idx = [i for i, a in enumerate(atoms) if is_oxygen(a)]
        self.heavy_idx = [i for i, a in enumerate(atoms) if a["element"].upper() != "H"]
        self.sync_rel()

    def sync_rel(self):
        ca = self.atoms[self.ca_idx]["xyz"]
        self.rel = np.array([a["xyz"] - ca for a in self.atoms], float)

    @property
    def ca(self) -> np.ndarray:
        return self.atoms[self.ca_idx]["xyz"]

    def set_pose(self, ca_xyz: np.ndarray, R: np.ndarray | None = None):
        ca_xyz = np.asarray(ca_xyz, float)
        R = np.eye(3) if R is None else R
        for i, a in enumerate(self.atoms):
            a["xyz"] = self.rel[i] @ R.T + ca_xyz
        self.atoms[self.ca_idx]["xyz"] = ca_xyz

    def oxyz(self) -> np.ndarray:
        return np.array([self.atoms[i]["xyz"] for i in self.o_idx], float)

    def heavy_xyz(self) -> np.ndarray:
        return np.array([self.atoms[i]["xyz"] for i in self.heavy_idx], float)

    def axes(self) -> list[np.ndarray]:
        out = []
        cs = [a for a in self.atoms if a["element"].upper() == "C"]
        if len(cs) >= 2:
            v = cs[1]["xyz"] - cs[0]["xyz"]
            if np.linalg.norm(v) > 0.15:
                out.append(norm(v))
        if self.o_idx:
            v = self.oxyz().mean(axis=0) - self.ca
            if np.linalg.norm(v) > 0.15:
                out.append(norm(v))
        for j in range(3):
            out.append(np.eye(3)[j])
        uniq = []
        for a in out:
            if not any(abs(float(np.dot(a, b))) > 0.92 for b in uniq):
                uniq.append(a)
        return uniq


def repartition_units(whw_atoms: list[dict]) -> list[Unit]:
    cas = sorted([a for a in whw_atoms if is_ca(a)], key=lambda a: a["resseq"])
    cxyz = xyz_of(cas)
    buckets = {c["resseq"]: [c] for c in cas}
    for a in whw_atoms:
        if is_ca(a):
            continue
        d = np.linalg.norm(cxyz - a["xyz"], axis=1)
        j = int(np.argmin(d))
        lim = LIGAND_CA_C if a["element"].upper() == "C" else LIGAND_CA_O
        if float(d[j]) <= lim:
            buckets[cas[j]["resseq"]].append(a)
    return [Unit(r, bs) for r, bs in sorted(buckets.items())]


def neighbor_pairs(units: list[Unit], cutoff=NEIGHBOR_CA):
    cxyz = np.array([u.ca for u in units], float)
    pairs = set()
    for i in range(len(units)):
        d = np.linalg.norm(cxyz - cxyz[i], axis=1)
        for j in np.where((d > 0.05) & (d <= cutoff))[0]:
            pairs.add((i, int(j)) if i < int(j) else (int(j), i))
    return sorted(pairs)


def oo_clash_list(units, pairs):
    hits = []
    for i, j in pairs:
        oi, oj = units[i].oxyz(), units[j].oxyz()
        if len(oi) == 0 or len(oj) == 0:
            continue
        d = np.linalg.norm(oi[:, None, :] - oj[None, :, :], axis=2)
        ii, jj = np.where(d < MIN_O_O)
        for a, b in zip(ii, jj):
            hits.append((float(d[a, b]), i, j, int(a), int(b)))
    hits.sort()
    return hits


def unit_score(idx, units, nbrs, dna_heavy):
    u = units[idx]
    sc = CA_ANCHOR_W * float(np.linalg.norm(u.ca - u.ca0) ** 2)
    heavy = u.heavy_xyz()
    if len(dna_heavy) and len(heavy):
        d = float(np.linalg.norm(dna_heavy - heavy[:, None, :], axis=2).min())
        if d < DNA_HEAVY:
            sc += DNA_W * (DNA_HEAVY - d) ** 2
    oi = u.oxyz()
    if len(oi) == 0:
        return sc
    for j in nbrs[idx]:
        oj = units[j].oxyz()
        if len(oj) == 0:
            continue
        d = np.linalg.norm(oi[:, None, :] - oj[None, :, :], axis=2)
        short = d[d < MIN_O_O]
        if short.size:
            sc += OO_W * float(np.sum((MIN_O_O - short) ** 2))
    return sc


def eval_pose(rel, o_idx, heavy_idx, ca_xyz, R, nbrs, units, idx, dna_heavy, ca0):
    oxyz = rel[o_idx] @ R.T + ca_xyz
    heavy = rel[heavy_idx] @ R.T + ca_xyz
    sc = CA_ANCHOR_W * float(np.linalg.norm(ca_xyz - ca0) ** 2)
    if len(dna_heavy) and len(heavy):
        d = float(np.linalg.norm(dna_heavy - heavy[:, None, :], axis=2).min())
        if d < DNA_HEAVY:
            sc += DNA_W * (DNA_HEAVY - d) ** 2
    for j in nbrs[idx]:
        oj = units[j].oxyz()
        if len(oj) == 0 or len(oxyz) == 0:
            continue
        d = np.linalg.norm(oxyz[:, None, :] - oj[None, :, :], axis=2)
        short = d[d < MIN_O_O]
        if short.size:
            sc += OO_W * float(np.sum((MIN_O_O - short) ** 2))
    return sc


def pairwise_relax(i, j, units, nbrs, dna_heavy):
    """Rotate one partner about its Ca (plus small shift) to clear O-O with the other."""
    ui, uj = units[i], units[j]
    mover, fixed, mi = (uj, ui, j) if j > i else (ui, uj, i)
    base = unit_score(mi, units, nbrs, dna_heavy)
    best_sc, best_ca, best_R = base, mover.ca.copy(), np.eye(3)

    axis = norm(mover.ca - fixed.ca)
    if np.linalg.norm(axis) < 0.1:
        axis = mover.axes()[0]
    oj = fixed.oxyz()
    for ang in np.linspace(0, 2 * math.pi, PAIRWISE_ANGLES, endpoint=False):
        R = rotation_around(axis, ang)
        for ca_try in (mover.ca0, mover.ca0 + axis * 0.15, mover.ca0 - axis * 0.15):
            sc = eval_pose(
                mover.rel, mover.o_idx, mover.heavy_idx, ca_try, R,
                nbrs, units, mi, dna_heavy, mover.ca0,
            )
            if len(mover.o_idx) and len(oj):
                oxyz = mover.rel[mover.o_idx] @ R.T + ca_try
                d = np.linalg.norm(oxyz[:, None, :] - oj[None, :, :], axis=2)
                short = d[d < MIN_O_O]
                if short.size:
                    sc += OO_W * float(np.sum((MIN_O_O - short) ** 2))
            if sc < best_sc:
                best_sc, best_ca, best_R = sc, ca_try.copy(), R

    if best_sc < base - 1e-5:
        mover.set_pose(best_ca, best_R)
        return True
    return False


def optimize_unit(idx, units, nbrs, dna_heavy, clash_hints):
    u = units[idx]
    base = unit_score(idx, units, nbrs, dna_heavy)
    best_sc, best_ca, best_R = base, u.ca.copy(), np.eye(3)
    angles = np.linspace(0, 2 * math.pi, N_ANGLES, endpoint=False)

    axes = u.axes()
    for other_idx in clash_hints.get(idx, [])[:6]:
        v = units[other_idx].ca - u.ca0
        if np.linalg.norm(v) > 0.1:
            axes.append(norm(v))
            axes.append(norm(np.cross(v, u.axes()[0])))

    for axis in axes:
        axis = norm(axis)
        for ang in angles:
            R = rotation_around(axis, ang)
            sc = eval_pose(
                u.rel, u.o_idx, u.heavy_idx, u.ca0, R, nbrs, units, idx, dna_heavy, u.ca0
            )
            if sc < best_sc:
                best_sc, best_ca, best_R = sc, u.ca0.copy(), R

    push = np.zeros(3)
    for _d, i, j, _oa, _ob in clash_hints.get("pairs", []):
        if idx not in (i, j):
            continue
        other = units[j if idx == i else i].ca
        push += norm(u.ca - other)

    dirs = [norm(push)] if np.linalg.norm(push) > 0.05 else []
    dirs.extend(axes)
    for direction in dirs:
        direction = norm(direction)
        for step in np.linspace(0.08, MAX_TRANS, 10):
            ca_try = best_ca + direction * step
            sc = eval_pose(
                u.rel, u.o_idx, u.heavy_idx, ca_try, best_R, nbrs, units, idx, dna_heavy, u.ca0
            )
            if sc < best_sc:
                best_sc, best_ca = sc, ca_try

    if best_sc < base - 1e-5:
        u.set_pose(best_ca, best_R)
        return True
    return False


def write_pdb(path, atoms, remarks):
    out = list(remarks)
    serial = 0
    prev = None
    for a in sorted(
        atoms,
        key=lambda t: (
            0 if t["resname"] == "NUC" else 1,
            t.get("chain", ""),
            t.get("resseq", 0),
            t.get("serial", 0),
        ),
    ):
        if prev is not None and a.get("chain") != prev and a["resname"] == "NUC":
            out.append("TER\n")
        serial += 1
        rec = "ATOM" if a["resname"] == "NUC" else "HETATM"
        out.append(
            format_atom(
                rec,
                serial,
                a["name"],
                a["resname"],
                a["chain"],
                a["resseq"],
                a["xyz"],
                1.0,
                a.get("bfac", 0.0),
                a["element"],
            )
        )
        prev = a.get("chain")
    out.append("END\n")
    path.write_text("".join(out))


def main():
    fast = "--fast" in sys.argv
    argv = [a for a in sys.argv[1:] if a != "--fast"]

    in_pdb = IN_PDB
    out_pdb = OUT_PDB
    if len(argv) > 0:
        in_pdb = Path(argv[0])
    if len(argv) > 1:
        out_pdb = Path(argv[1])
    elif in_pdb != IN_PDB:
        out_pdb = in_pdb.with_name(in_pdb.stem + "_relaxed.pdb")
    report = out_pdb.with_name(out_pdb.stem + "_report.txt")

    atoms, _ = parse_atoms(in_pdb)
    dna_heavy = xyz_of(
        [a for a in atoms if a["resname"] == "NUC" and a["element"].upper() != "H"]
    )
    other = [a for a in atoms if a["resname"] != "WHW"]
    whw = [a for a in atoms if a["resname"] == "WHW"]
    units = repartition_units(whw)
    configure_speed(len(units), fast)
    if fast or len(units) >= FAST_UNIT_THRESHOLD:
        print(f"Fast relax: {len(units)} WHW units")
    pairs = neighbor_pairs(units)
    nbrs = defaultdict(list)
    for i, j in pairs:
        nbrs[i].append(j)
        nbrs[j].append(i)

    ster0 = short_contact_summary([a for u in units for a in u.atoms])
    hits0 = oo_clash_list(units, pairs)
    clash_units = {i for _d, i, j, _oa, _ob in hits0 for i in (i, j)}
    if fast or len(units) >= FAST_UNIT_THRESHOLD:
        print(f"Units in O-O clash: {len(clash_units)} / {len(units)}")

    moves = 0
    for pass_no in range(MAX_PASSES):
        pass_moves = 0
        for sweep in range(MAX_SWEEPS):
            hits = oo_clash_list(units, pairs)
            if not hits:
                break
            clash_units.update(idx for _d, i, j, _oa, _ob in hits for idx in (i, j))
            clash_hints = defaultdict(list)
            clash_hints["pairs"] = hits[:200]
            for _d, i, j, _oa, _ob in hits[:200]:
                clash_hints[i].append(j)
                clash_hints[j].append(i)
            order = sorted(
                (i for i in clash_units if i in clash_hints),
                key=lambda k: len(clash_hints.get(k, [])),
                reverse=True,
            )
            improved = False
            seen_pairs = set()
            n_pair = 10 if len(units) >= FAST_UNIT_THRESHOLD else 30
            for _d, i, j, _oa, _ob in hits[:n_pair]:
                key = (min(i, j), max(i, j))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                if pairwise_relax(i, j, units, nbrs, dna_heavy):
                    pass_moves += 1
                    moves += 1
                    improved = True
            for idx in order:
                if optimize_unit(idx, units, nbrs, dna_heavy, clash_hints):
                    pass_moves += 1
                    moves += 1
                    improved = True
            if not improved:
                break
        if pass_moves == 0:
            break
        for u in units:
            u.ca0 = u.ca.copy()
            u.sync_rel()

    whw_after = [a for u in units for a in u.atoms]
    ster1 = short_contact_summary(whw_after)
    hits1 = oo_clash_list(units, pairs)

    remarks = [
        "HEADER    RELAXED WHEWELLITE ON GROWTH MODEL\n",
        "TITLE     RIGID WHW UNIT RELAXATION FOR O-O CLEARANCE\n",
        f"REMARK   1 {moves} rigid moves; O-O < {MIN_O_O:.1f} A: "
        f"{ster0['n_oo_short']} -> {ster1['n_oo_short']}.\n",
    ]
    write_pdb(out_pdb, other + whw_after, remarks)

    lines = [
        "Rigid-body WHW relaxation",
        "=" * 50,
        f"Input  : {in_pdb.name}",
        f"Output : {out_pdb.name}",
        f"Units  : {len(units)} WHW Ca sites (repartitioned ligands)",
        f"Moves  : {moves} accepted (up to {MAX_PASSES} passes)",
        "",
        "Inter-residue O-O (WHW only):",
        f"  before: n={ster0['n_oo_short']}  min={ster0['oo_min']:.3f} A",
        f"  after : n={ster1['n_oo_short']}  min={ster1['oo_min']:.3f} A",
        "",
        "Worst pairs remaining (<= 15):",
    ]
    for d, i, j, _oa, _ob in hits1[:15]:
        lines.append(f"  {d:.3f} A  WHW {units[i].resseq} -- WHW {units[j].resseq}")
    if not hits1:
        lines.append("  none")
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
