#!/usr/bin/env python3
"""
One authentic whewellite (COM) crystallite for small MD.

Unlike COM_patch.pdb (rigid monomers, Ca–Ca ≥ 6 Å), this cut is a
single crystal fragment: 3.84, 6.29, and 10.1 Å Ca–Ca are all present.
Only one seed is used, so neighboring DNA patches cannot overlap.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geom_constraints import is_ca, is_oxygen, pair_distances, xyz_of  # noqa: E402
from grow_whewellite import (  # noqa: E402
    BACKBONE,
    XTL,
    cell_vectors,
    dna_ca_by_strand,
    expand_crystal,
    format_atom,
    local_frame,
    parse_atoms,
    rotation_around,
    rotation_from_to,
    unique_ca_distances,
)

ROOT = Path(__file__).resolve().parents[1]
SEED_RES = 11
RADIUS = 10.0
MIN_CA_CA = 3.50  # keep COM 3.84; drop unphysical <3.5 Å
MIN_O_O = 2.0
OUT_PDB = ROOT / "COM_patch_whewellite.pdb"
OUT_XYZ = ROOT / "COM_patch_whewellite.xyz"
OUT_DNA = ROOT / "COM_patch_whewellite_withDNA.pdb"
REPORT = ROOT / "COM_patch_whewellite_report.txt"

COM_TARGETS = (3.843, 6.290, 6.681, 6.881, 7.277, 10.116)


def complete_oxalate(kept, crystal):
    """If any C is kept, add its partner C and carboxylate O from the crystal."""
    have = {id(a) for a in kept}
    carbons = [a for a in kept if a["element"].upper() == "C"]
    extra = []
    for c in carbons:
        for a in crystal:
            if a is c or id(a) in have:
                continue
            d = float(np.linalg.norm(a["xyz"] - c["xyz"]))
            el = a["element"].upper()
            if el == "C" and d < 1.70:
                extra.append(a)
                have.add(id(a))
            elif el.startswith("O") and d < 1.55:
                extra.append(a)
                have.add(id(a))
    # also add waters coordinated to a kept Ca (OW not bound to C)
    cas = [a for a in kept if is_ca(a)]
    cxyz = xyz_of([a for a in crystal if a["element"].upper() == "C"])
    for a in crystal:
        if not a["element"].upper().startswith("O"):
            continue
        if id(a) in have:
            continue
        if len(cxyz) and float(np.linalg.norm(cxyz - a["xyz"], axis=1).min()) < 1.55:
            continue
        if cas and float(np.linalg.norm(xyz_of(cas) - a["xyz"], axis=1).min()) < 2.70:
            extra.append(a)
            have.add(id(a))
    return kept + extra


def assign_residues(atoms):
    cas = [a for a in atoms if is_ca(a)]
    cas.sort(key=lambda a: float(np.linalg.norm(a["xyz"])))
    cxyz = xyz_of(cas)
    out = []
    for a in atoms:
        b = dict(a)
        if is_ca(a):
            b["resseq"] = next(i for i, c in enumerate(cas, start=1) if c is a)
        else:
            b["resseq"] = int(np.argmin(np.linalg.norm(cxyz - a["xyz"], axis=1))) + 1
        b["resname"] = "WHW"
        b["chain"] = "Z"
        out.append(b)
    # seed Ca (closest to origin after centering? we keep seed as res 1)
    return out


def drop_clash_ca(atoms, min_ca=MIN_CA_CA):
    cas = [a for a in atoms if is_ca(a)]
    keep = []
    for a in cas:
        if keep and float(np.linalg.norm(xyz_of(keep) - a["xyz"], axis=1).min()) < min_ca:
            continue
        keep.append(a)
    keep_set = {id(a) for a in keep}
    # drop ligands whose nearest Ca was removed
    kept_ca_xyz = xyz_of(keep)
    out = []
    for a in atoms:
        if is_ca(a):
            if id(a) in keep_set:
                out.append(a)
            continue
        if len(kept_ca_xyz) and float(np.linalg.norm(kept_ca_xyz - a["xyz"], axis=1).min()) < 5.5:
            out.append(a)
    return out


def drop_oo_clashes(atoms: list[dict]) -> list[dict]:
    """Remove inter-residue O–O pairs closer than MIN_O_O."""
    cas = {a["resseq"]: a for a in atoms if is_ca(a)}
    oxs = [a for a in atoms if is_oxygen(a)]
    drop = set()
    for i, a in enumerate(oxs):
        for b in oxs[i + 1 :]:
            if a["resseq"] == b["resseq"]:
                continue
            if float(np.linalg.norm(a["xyz"] - b["xyz"])) < MIN_O_O:
                da = float(np.linalg.norm(a["xyz"] - cas[a["resseq"]]["xyz"]))
                db = float(np.linalg.norm(b["xyz"] - cas[b["resseq"]]["xyz"]))
                drop.add(id(a if da >= db else b))
    return [a for a in atoms if id(a) not in drop]


def cut_crystallite_patch(
    center_xyz,
    tangent,
    outward,
    radius: float = RADIUS,
    nmax: int = 2,
) -> list[dict]:
    """
    Authentic whewellite fragment aligned to a local DNA frame.

    ``center_xyz`` is the patch center (typically a Ca site in the bulk shell).
    Returns atom dicts with xyz set; no resseq/chain assigned yet.
    """
    from grow_crystal_from_growth import alignment_matrix

    xtl_atoms, cryst = parse_atoms(XTL)
    if cryst is None:
        raise ValueError("No CRYST1 in whewellite xtl.pdb")
    expanded, (av, bv, cv) = expand_crystal(xtl_atoms, cryst, nmax=nmax)
    ref = next(a for a in xtl_atoms if a["element"].upper() == "CA")
    center = np.asarray(center_xyz, float)
    R = alignment_matrix(av, cv, np.asarray(tangent, float), np.asarray(outward, float))

    crystal = []
    for a in expanded:
        if a["element"].upper() == "H":
            continue
        b = dict(a)
        b["xyz"] = (a["xyz"] - ref["xyz"]) @ R.T + center
        crystal.append(b)

    kept = [
        a
        for a in crystal
        if float(np.linalg.norm(a["xyz"] - center)) <= radius + 0.05
    ]
    kept = complete_oxalate(kept, crystal)
    kept = drop_clash_ca(kept, MIN_CA_CA)
    kept = assign_residues(kept)
    return drop_oo_clashes(kept)


def write_pdb(path, atoms, remarks, extra=None):
    out = list(remarks)
    serial = 0
    extra = extra or []
    prev = None
    for a in extra:
        if prev is not None and a.get("chain") != prev:
            out.append("TER\n")
        serial += 1
        out.append(
            format_atom(
                "ATOM",
                serial,
                a["name"],
                a.get("resname", "NUC"),
                a.get("chain", "A"),
                a.get("resseq", 1),
                a["xyz"],
                1.0,
                0.0,
                a["element"],
            )
        )
        prev = a.get("chain")
    if extra:
        out.append("TER\n")
    for a in sorted(atoms, key=lambda t: (t["resseq"], t["name"])):
        serial += 1
        el = "Ca" if is_ca(a) else a["element"]
        name = "CA" if is_ca(a) else a["name"]
        out.append(
            format_atom(
                "HETATM", serial, name, "WHW", "Z", a["resseq"], a["xyz"], 1.0, 20.0, el
            )
        )
    out.append("END\n")
    path.write_text("".join(out))


def write_xyz(path, atoms, title):
    lines = [f"{len(atoms)}\n", title + "\n"]
    for a in atoms:
        el = "Ca" if is_ca(a) else a["element"]
        x, y, z = a["xyz"]
        lines.append(f"{el:<2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")
    path.write_text("".join(lines))


def main():
    dna_atoms, _ = parse_atoms(BACKBONE)
    xtl_atoms, cryst = parse_atoms(XTL)
    if cryst is None:
        raise SystemExit("No CRYST1 in whewellite xtl.pdb")
    expanded, (av, bv, cv) = expand_crystal(xtl_atoms, cryst, nmax=2)
    xtl_ca = xyz_of([a for a in xtl_atoms if a["element"].upper() == "CA"])
    uniq_d = unique_ca_distances(xtl_ca, av, bv, cv, cutoff=11.0)
    ref = next(a for a in xtl_atoms if a["element"].upper() == "CA")

    strands = dna_ca_by_strand(dna_atoms)
    seed = None
    tangent = outward = None
    for strand in strands:
        helix_o = np.mean([a["xyz"] for a in strand], axis=0)
        for i, ca in enumerate(strand):
            if ca["resseq"] == SEED_RES:
                seed = ca
                tangent, outward = local_frame(strand, i, helix_o)
                break
        if seed is not None:
            break
    if seed is None:
        raise SystemExit(f"COM residue {SEED_RES} not found")

    kept = cut_crystallite_patch(seed["xyz"], tangent, outward, RADIUS, nmax=2)

    cas = [a for a in kept if is_ca(a)]
    cxyz = xyz_of(cas)
    pair = pair_distances(cxyz, 11.0)

    def nnear(target, tol=0.30):
        return int(np.sum(np.abs(pair - target) <= tol)) if pair.size else 0

    nuc = [
        a
        for a in dna_atoms
        if a["resname"] == "NUC"
        and float(np.linalg.norm(a["xyz"] - seed["xyz"])) <= RADIUS + 2.0
    ]

    lines = [
        "Whewellite crystallite patch (single seed)",
        "=" * 56,
        f"Source crystal : {XTL.name}",
        f"Center         : DNA-bound COM residue {SEED_RES} (same site as COM_patch)",
        f"Cut            : {RADIUS:.1f} Å COM lattice, one seed only (no patch overlap)",
        f"Atoms          : {len(kept)}   Ca: {len(cas)}",
        "",
        "This is a real COM fragment, not rigid CaOx monomers.",
        "Ca–Ca therefore includes the crystallographic 3.84 Å contact.",
        "Unphysical Ca–Ca < 3.50 Å and inter-residue O–O < 2.0 Å are removed.",
        "",
        "COM reference Ca–Ca ≤ 11 Å:",
    ]
    for d, n in uniq_d:
        lines.append(f"   {d:6.3f} Å   (x{n})")
    lines.append("")
    lines.append("Ca–Ca in this patch (≤ 11 Å):")
    for tgt in COM_TARGETS:
        lines.append(f"  {tgt:5.2f} Å ±0.30 : {nnear(tgt)} pairs")
    lines.append(
        f"  n pairs={len(pair)}"
        + (f"  min={pair.min():.2f}  max={pair.max():.2f}" if pair.size else "")
    )
    lines.append("")
    lines.append("Files:")
    lines.append(f"  {OUT_PDB.name}   crystallite only (MD / QM)")
    lines.append(f"  {OUT_XYZ.name}   same, XYZ")
    lines.append(f"  {OUT_DNA.name}   crystallite + nearby DNA atoms")
    lines.append("")
    lines.append("COM_patch.pdb is the 6 Å rigid-monomer alternative.")
    report = "\n".join(lines) + "\n"
    REPORT.write_text(report)
    print(report)

    remarks = [
        "HEADER    WHEWELLITE CRYSTALLITE PATCH\n",
        f"TITLE     COM LATTICE 10 A AROUND DNA-BOUND SEED {SEED_RES}\n",
        "REMARK   1 Single-seed whewellite fragment. Ca-Ca includes 3.84 A.\n",
        "REMARK   1 a-axis || local DNA backbone; c points off the helix.\n",
        f"REMARK   2 {len(kept)} atoms, {len(cas)} Ca. No overlapping patches.\n",
        f"REMARK   3 COM cell a={cryst['a']:.3f} b={cryst['b']:.3f} c={cryst['c']:.3f} "
        f"beta={cryst['beta']:.2f}\n",
    ]
    write_pdb(OUT_PDB, kept, remarks)
    write_xyz(OUT_XYZ, kept, f"Whewellite crystallite seed {SEED_RES} R={RADIUS} A")
    write_pdb(
        OUT_DNA,
        kept,
        [
            "HEADER    WHEWELLITE PATCH WITH LOCAL DNA\n",
            f"TITLE     SEED {SEED_RES} CRYSTALLITE + NEARBY NUCLEOTIDE ATOMS\n",
        ],
        extra=nuc,
    )
    print(f"Wrote {OUT_PDB}")
    print(f"Wrote {OUT_XYZ}")
    print(f"Wrote {OUT_DNA}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
