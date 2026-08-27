#!/usr/bin/env python3
"""Isolate one local COM-like patch for small MD / QM jobs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geom_constraints import MIN_CA_CA, MIN_O_O, is_ca, short_contact_summary  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "CaOx_whewellite_noDNA.pdb"
SEED_RES = None  # pick the seed with the most 6.3 Å neighbors
RADIUS = 8.0
OUT_PDB = ROOT / "COM_patch.pdb"
OUT_XYZ = ROOT / "COM_patch.xyz"
OUT_CORE = ROOT / "COM_patch_core.pdb"
REPORT = ROOT / "COM_patch_report.txt"


def parse(path: Path):
    atoms = []
    for line in path.open():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atoms.append(
            {
                "rec": line[:6].strip(),
                "name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21],
                "resseq": int(line[22:26]),
                "xyz": np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                ),
                "element": line[76:78].strip() or line[12:16].strip()[0],
                "bfac": float(line[60:66]) if line[60:66].strip() else 0.0,
            }
        )
    return atoms


def format_atom(serial, a):
    el = a["element"]
    name = a["name"]
    if el.upper() == "CA":
        aname, el = "CA  ", "Ca"
    elif name.startswith("OW"):
        aname = f"{name:<4}"[:4]
    elif len(name) >= 4:
        aname = name[:4]
    else:
        aname = f" {name:<3}"[:4]
    x, y, z = a["xyz"]
    return (
        f"HETATM{serial:5d} {aname:4s} {a['resname']:<3s} {a['chain']}{a['resseq']:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{a['bfac']:6.2f}"
        f"          {el:>2s}\n"
    )


def isolate(atoms, seed_xyz, radius, seed_res, complete_residues=True):
    """Keep whole rigid CaOx residues that enter the sphere."""
    keys_hit = set()
    for a in atoms:
        if np.linalg.norm(a["xyz"] - seed_xyz) <= radius:
            keys_hit.add((a["chain"], a["resseq"], a["resname"]))
    keys_hit.add(("X", seed_res, "COM"))
    if complete_residues:
        return [
            a
            for a in atoms
            if (a["chain"], a["resseq"], a["resname"]) in keys_hit
        ]
    return [a for a in atoms if np.linalg.norm(a["xyz"] - seed_xyz) <= radius]


def write_pdb(path, atoms, remarks):
    out = list(remarks)
    serial = 0
    prev = None
    for a in sorted(atoms, key=lambda t: (t["chain"], t["resseq"], t["name"])):
        if prev is not None and a["chain"] != prev:
            out.append("TER\n")
        serial += 1
        out.append(format_atom(serial, a))
        prev = a["chain"]
    out.append("TER\nEND\n")
    path.write_text("".join(out))


def write_xyz(path, atoms, title):
    lines = [f"{len(atoms)}\n", title + "\n"]
    for a in atoms:
        el = "Ca" if a["element"].upper() == "CA" else a["element"]
        x, y, z = a["xyz"]
        lines.append(f"{el:<2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")
    path.write_text("".join(lines))


def main():
    atoms = parse(SRC)
    seeds = [
        a
        for a in atoms
        if a["chain"] == "X"
        and a["name"].upper() == "CA"
        and a["resname"] == "COM"
    ]
    all_ca = [a for a in atoms if is_ca(a)]
    cxyz = np.array([a["xyz"] for a in all_ca])
    best_seed, best_n = seeds[0], -1
    for s in seeds:
        d = np.linalg.norm(cxyz - s["xyz"], axis=1)
        n = int(np.sum((d >= MIN_CA_CA) & (d <= 7.20) & (d > 0.1)))
        if n > best_n:
            best_n, best_seed = n, s
    seed = best_seed
    seed_res = seed["resseq"]
    patch = isolate(atoms, seed["xyz"], RADIUS, seed_res)
    core = isolate(atoms, seed["xyz"], 6.5, seed_res)

    cas = [a for a in patch if a["element"].upper() == "CA" or a["name"].upper() == "CA"]
    cxyz = np.array([a["xyz"] for a in cas])
    pair = []
    for i in range(len(cxyz)):
        for j in range(i + 1, len(cxyz)):
            d = float(np.linalg.norm(cxyz[i] - cxyz[j]))
            if d <= 11:
                pair.append(d)
    pair = np.array(pair) if pair else np.array([])

    def nnear(target, tol=0.30):
        return int(np.sum(np.abs(pair - target) <= tol)) if pair.size else 0

    counts = defaultdict(int)
    for a in patch:
        counts[a["resname"]] += 1

    ster = short_contact_summary(patch)
    remarks = [
        "HEADER    SINGLE COM-LIKE PATCH\n",
        f"TITLE     8 A RIGID CAOX PATCH AROUND DNA-BOUND SEED {seed_res}\n",
        "REMARK   1 Isolated from CaOx_whewellite_noDNA.pdb. No DNA.\n",
        f"REMARK   2 Center = COM chain X residue {seed_res} Ca. Radius {RADIUS:.1f} A.\n",
        "REMARK   2 Whole rigid CaOx units; Ca-Ca >= 6 A, inter-unit O-O >= 2 A.\n",
        f"REMARK   3 {len(patch)} atoms, {len(cas)} Ca. For small MD / cluster QM.\n",
    ]
    write_pdb(OUT_PDB, patch, remarks)
    write_xyz(OUT_XYZ, patch, f"COM patch seed {seed_res} R={RADIUS} A, no DNA")
    write_pdb(
        OUT_CORE,
        core,
        [
            "HEADER    COM PATCH CORE\n",
            f"TITLE     6.5 A CORE AROUND SEED {seed_res} (SMALLER)\n",
            f"REMARK   1 {len(core)} atoms. Use if 8 A is still too large.\n",
        ],
    )

    lines = [
        "Isolated COM-like patch",
        "=" * 50,
        f"Source : {SRC.name}",
        f"Center : COM X residue {seed_res} ({best_n} neighbors at 6.0–7.2 Å)",
        f"Keep   : whole rigid CaOx residues that enter {RADIUS:.1f} A",
        f"Atoms  : {len(patch)}   Ca: {len(cas)}   {dict(counts)}",
        f"Core 6.5 A: {len(core)} atoms",
        "",
        f"Steric: Ca–Ca < {MIN_CA_CA:.1f} Å = {ster['n_ca_short']}; "
        f"O–O < {MIN_O_O:.1f} Å = {ster['n_oo_short']}",
        "",
        "Ca–Ca in the 8 A patch (≤ 11 A):",
        f"  6.29 Å ±0.30 : {nnear(6.29)} pairs",
        f"  10.12 Å ±0.30: {nnear(10.116)} pairs",
        f"  n pairs={len(pair)}"
        + (f"  min={pair.min():.2f}  max={pair.max():.2f}" if pair.size else ""),
        "",
        "Files:",
        f"  {OUT_PDB.name}   {len(patch)} atoms  (recommended MD start)",
        f"  {OUT_XYZ.name}   same, XYZ",
        f"  {OUT_CORE.name}  {len(core)} atoms (tighter cut)",
        "",
        "Solvate and add H on the cluster (tleap / gmx pdb2gmx).",
        "This is a geometric extract, not minimized.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
