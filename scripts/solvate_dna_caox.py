#!/usr/bin/env python3
"""
Add a compact water droplet and free (solution) CaOx units around one DNA duplex.

Designed to stay small enough for cluster MD / Gaussian setup:
one 1BNA duplex + backbone COM + first-shell waters + a few floating CaOx.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geom_constraints import MIN_CA_CA, MIN_O_O, is_oxygen, min_pair, xyz_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IN_PDB = ROOT / "DNA_CaOx_backbone.pdb"
CAOX_PDB = ROOT / "DOCS" / "Whewellite ca_ox.pdb"
OUT_PDB = ROOT / "DNA_CaOx_solvated.pdb"
OUT_XYZ = ROOT / "DNA_CaOx_solvated.xyz"
REPORT = ROOT / "DNA_CaOx_solvated_report.txt"

# Keep the system small: one duplex, first hydration shell, few free CaOx.
CHAINS = "AB"
N_WATER = 110
N_FLOAT = 8
SHELL_MIN = 2.40  # Å, O vs solute heavy atoms
SHELL_MAX = 3.90
WATER_SPACING = 2.80
FLOAT_DNA_MIN = 4.60
FLOAT_CA_MIN = MIN_CA_CA
WATER_FLOAT_MIN = MIN_O_O
RNG = np.random.default_rng(7)

# TIP3P
OH = 0.9572
HOH = math.radians(104.52)
TIP3P = np.array(
    [
        [0.0, 0.0, 0.0],
        [OH, 0.0, 0.0],
        [OH * math.cos(HOH), OH * math.sin(HOH), 0.0],
    ]
)


def parse_atoms(path: Path):
    atoms = []
    for line in path.open():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atoms.append(
            {
                "rec": line[:6].strip(),
                "serial": int(line[6:11]),
                "name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21],
                "resseq": int(line[22:26]),
                "xyz": np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                ),
                "element": (line[76:78].strip() or line[12:16].strip()[0]),
            }
        )
    return atoms


def format_atom(rec, serial, name, resname, chain, resseq, xyz, occ, bfac, element):
    if element.upper() == "CA":
        aname, element = "CA  ", "Ca"
    elif name.startswith("OW"):
        aname = f"{name:<4}"[:4]
    elif name in ("O", "H1", "H2"):
        aname = f" {name:<3}"[:4]
    elif len(name) >= 4:
        aname = name[:4]
    else:
        aname = f" {name:<3}"[:4]
    return (
        f"{rec:<6}{serial:5d} {aname:4s} {resname:>3s} {chain:1s}{resseq:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{occ:6.2f}{bfac:6.2f}"
        f"          {element:>2s}\n"
    )


def random_rotation():
    # Uniform rotation (Shoemake)
    u1, u2, u3 = RNG.random(3)
    q = np.array(
        [
            math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
            math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
            math.sqrt(u1) * math.sin(2 * math.pi * u3),
            math.sqrt(u1) * math.cos(2 * math.pi * u3),
        ]
    )
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def load_caox_solute(path: Path):
    """Ca + oxalate only (no crystal waters) for solution-phase units."""
    raw = parse_atoms(path)
    ca = next(a for a in raw if a["element"].upper() == "CA")
    carbons = [a for a in raw if a["element"].upper() == "C"]
    carbon_ids = {a["serial"] for a in carbons}
    # oxalate O: remaining O bonded conceptually — take the four O nearest the C2 pair
    others = [a for a in raw if a["serial"] != ca["serial"] and a["element"].upper() != "C"]
    ccent = np.mean([a["xyz"] for a in carbons], axis=0)
    others.sort(key=lambda a: np.linalg.norm(a["xyz"] - ccent))
    oxo = others[:4]
    atoms = [ca] + carbons + oxo
    origin = ca["xyz"].copy()
    return [{"name": a["name"], "element": a["element"], "xyz": a["xyz"] - origin} for a in atoms]


def min_dist(point, cloud):
    return float(np.linalg.norm(cloud - point, axis=1).min())


def select_duplex(atoms):
    dna = [a for a in atoms if a["resname"] == "NUC" and a["chain"] in CHAINS]
    p_dna = np.array([a["xyz"] for a in dna if a["element"].upper() == "P"])
    com = [a for a in atoms if a["resname"] == "COM"]
    keep_res = set()
    for a in com:
        if a["name"].strip().upper() != "CA":
            continue
        if min_dist(a["xyz"], p_dna) <= 5.0:
            keep_res.add(a["resseq"])
    bound = [a for a in com if a["resseq"] in keep_res]
    return dna, bound, keep_res


def place_floating(frag, solute_xyz, solute_o, n=N_FLOAT):
    lo = solute_xyz.min(axis=0) - 2.5
    hi = solute_xyz.max(axis=0) + 2.5
    placed = []
    ca_xyz = []
    placed_o = []
    for _ in range(8000):
        if len(placed) >= n:
            break
        trial = RNG.uniform(lo, hi)
        d_sol = min_dist(trial, solute_xyz)
        if d_sol < FLOAT_DNA_MIN or d_sol > SHELL_MAX + 2.8:
            continue
        if ca_xyz and min_dist(trial, np.array(ca_xyz)) < FLOAT_CA_MIN:
            continue
        R = random_rotation()
        atoms = []
        for a in frag:
            b = dict(a)
            b["xyz"] = a["xyz"] @ R.T + trial
            atoms.append(b)
        heavy = np.array([a["xyz"] for a in atoms])
        oxyz = np.array([a["xyz"] for a in atoms if is_oxygen(a)])
        dmat = np.linalg.norm(heavy[:, None, :] - solute_xyz[None, :, :], axis=2)
        if dmat.min() < 2.30:
            continue
        exist_o = solute_o
        if placed_o:
            exist_o = np.vstack([exist_o, np.array(placed_o)]) if len(exist_o) else np.array(placed_o)
        if len(oxyz) and len(exist_o) and min_pair(oxyz, exist_o) < MIN_O_O:
            continue
        placed.append(atoms)
        ca_xyz.append(trial)
        placed_o.extend([p for p in oxyz])
    return placed


def candidate_water_sites(solute_xyz):
    pad = SHELL_MAX + 0.4
    lo = solute_xyz.min(axis=0) - pad
    hi = solute_xyz.max(axis=0) + pad
    xs = np.arange(lo[0], hi[0], WATER_SPACING)
    ys = np.arange(lo[1], hi[1], WATER_SPACING * math.sqrt(3) / 2)
    sites = []
    for iy, y in enumerate(ys):
        xoff = 0.5 * WATER_SPACING if iy % 2 else 0.0
        for x in xs:
            for iz, z in enumerate(np.arange(lo[2], hi[2], WATER_SPACING * 0.82)):
                pt = np.array([x + xoff, y, z + (0.4 * WATER_SPACING if iz % 2 else 0.0)])
                d = min_dist(pt, solute_xyz)
                if SHELL_MIN <= d <= SHELL_MAX:
                    sites.append((d, pt))
    sites.sort(key=lambda t: t[0])
    # greedy spacing
    kept = []
    for d, pt in sites:
        if kept and min_dist(pt, np.array(kept)) < WATER_SPACING * 0.92:
            continue
        kept.append(pt)
        if len(kept) >= N_WATER * 3:
            break
    return kept


def pack_waters(solute_xyz, extra_xyz, n=N_WATER):
    cloud = solute_xyz if extra_xyz.size == 0 else np.vstack([solute_xyz, extra_xyz])
    sites = candidate_water_sites(cloud)
    waters = []
    for pt in sites:
        if extra_xyz.size and min_dist(pt, extra_xyz) < WATER_FLOAT_MIN:
            continue
        R = random_rotation()
        coords = TIP3P @ R.T + pt
        # hydrogens should not sit inside solute
        h_ok = True
        for h in coords[1:]:
            if min_dist(h, cloud) < 1.70:
                h_ok = False
                break
        if not h_ok:
            continue
        waters.append(coords)
        if len(waters) >= n:
            break
    return waters


def main():
    all_atoms = parse_atoms(IN_PDB)
    dna, bound, keep_res = select_duplex(all_atoms)
    solute = dna + bound
    solute_xyz = np.array([a["xyz"] for a in solute])

    frag = load_caox_solute(CAOX_PDB)
    solute_o = xyz_of([a for a in solute if is_oxygen(a)])
    floating = place_floating(frag, solute_xyz, solute_o, N_FLOAT)
    extra = np.array([a["xyz"] for unit in floating for a in unit]) if floating else np.zeros((0, 3))
    waters = pack_waters(solute_xyz, extra, N_WATER)

    n_dna = len(dna)
    n_bound = len(bound)
    n_float = sum(len(u) for u in floating)
    n_wat_atoms = 3 * len(waters)
    n_total = n_dna + n_bound + n_float + n_wat_atoms

    lines = [
        "DNA + CaOx + compact water droplet",
        "=" * 50,
        f"Duplex chains     : {CHAINS}",
        f"DNA atoms         : {n_dna}",
        f"Bound COM atoms   : {n_bound}  (residues {sorted(keep_res)})",
        f"Floating CaOx     : {len(floating)} units, {n_float} atoms (Ca + C2O4)",
        f"Waters            : {len(waters)} molecules, {n_wat_atoms} atoms",
        f"TOTAL atoms       : {n_total}",
        "",
        "This is a first hydration shell, not a periodic box.",
        "DNA hydrogens are not added here; protonate before MD",
        "(reduce / tleap / OpenBabel). Waters are TIP3P-like.",
        "",
    ]
    report = "\n".join(lines)
    REPORT.write_text(report + "\n")
    print(report)

    remarks = [
        "REMARK   1 One 1BNA duplex (chains A+B) with backbone COM units,",
        "REMARK   1 a compact first-shell water droplet, and free CaOx in water.",
        "REMARK   2 Second crystal-packing duplex omitted to keep the atom count",
        "REMARK   2 small for cluster MD / Gaussian setup.",
        f"REMARK   3 Waters: {len(waters)} TIP3P-like H2O (chain W).",
        f"REMARK   3 Floating CaOx: {len(floating)} CaC2O4 units (chain Y, residue COX).",
        f"REMARK   4 Total atoms: {n_total}. Geometric packing, not equilibrated.",
        "REMARK   4 Protonate DNA before production MD.",
    ]

    out = [
        "HEADER    DNA CAOX DROPLET FOR MD\n",
        "TITLE     ONE DUPLEX + BACKBONE COM + WATER + FREE CAOX\n",
    ]
    for r in remarks:
        out.append(r + "\n")

    serial = 0

    def write_atom(rec, name, resname, chain, resseq, xyz, element, bfac=0.0):
        nonlocal serial
        serial += 1
        out.append(
            format_atom(rec, serial, name, resname, chain, resseq, xyz, 1.00, bfac, element)
        )

    prev_chain = None
    for a in sorted(dna, key=lambda t: (t["chain"], t["resseq"], t["serial"])):
        if prev_chain is not None and a["chain"] != prev_chain:
            out.append("TER\n")
        write_atom("ATOM", a["name"], "NUC", a["chain"], a["resseq"], a["xyz"], a["element"])
        prev_chain = a["chain"]
    out.append("TER\n")

    for a in sorted(bound, key=lambda t: (t["resseq"], t["serial"])):
        write_atom("HETATM", a["name"], "COM", "X", a["resseq"], a["xyz"], a["element"], 20.0)
    out.append("TER\n")

    name_map = {"CA": ("CA", "Ca")}
    for i, unit in enumerate(floating, start=1):
        c_i = o_i = 0
        for a in unit:
            el = a["element"]
            if el.upper() == "CA":
                name, el = "CA", "Ca"
            elif el.upper() == "C":
                c_i += 1
                name = f"C{c_i}"
            else:
                o_i += 1
                name = f"O{o_i}"
            write_atom("HETATM", name, "COX", "Y", i, a["xyz"], el, 30.0)
    if floating:
        out.append("TER\n")

    for i, coords in enumerate(waters, start=1):
        write_atom("HETATM", "O", "HOH", "W", i, coords[0], "O")
        write_atom("HETATM", "H1", "HOH", "W", i, coords[1], "H")
        write_atom("HETATM", "H2", "HOH", "W", i, coords[2], "H")

    out.append("END\n")
    OUT_PDB.write_text("".join(out))

    xyz_atoms = []
    for a in dna + bound:
        xyz_atoms.append((a["element"] if a["element"] != "CA" else "Ca", a["xyz"]))
    for unit in floating:
        for a in unit:
            el = "Ca" if a["element"].upper() == "CA" else a["element"]
            xyz_atoms.append((el, a["xyz"]))
    for coords in waters:
        xyz_atoms.append(("O", coords[0]))
        xyz_atoms.append(("H", coords[1]))
        xyz_atoms.append(("H", coords[2]))

    xyz = [f"{len(xyz_atoms)}\n", "DNA CaOx droplet (one duplex + water + free CaOx)\n"]
    for el, p in xyz_atoms:
        xyz.append(f"{el:<2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n")
    OUT_XYZ.write_text("".join(xyz))

    print(f"Wrote {OUT_PDB}")
    print(f"Wrote {OUT_XYZ}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
