#!/usr/bin/env python3
"""
Build starting models to test DNA templating of whewellite.

1) Assembly: bare DNA + Na+ + free Ca2+ + free oxalate + water
   (nucleation from solution — no pre-bound COM)

2) Growth: DNA + a short bound CaOx seed + free Ca2+/oxalate + water
   (elongation along the phosphate ladder)

3) QM cluster: one phosphate, bound CaOx, incoming CaOx, nearby waters
   (Gaussian association energy)
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geom_constraints import MIN_CA_CA, MIN_O_O, min_pair  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BACKBONE = ROOT / "DNA_CaOx_backbone.pdb"
SOLVATED = ROOT / "DNA_CaOx_solvated.pdb"
CAOX = ROOT / "DOCS" / "Whewellite ca_ox.pdb"

OUT_ASM = ROOT / "DNA_CaOx_assembly.pdb"
OUT_GRO = ROOT / "DNA_CaOx_growth.pdb"
OUT_QM_XYZ = ROOT / "QM_association_cluster.xyz"
OUT_QM_COM = ROOT / "QM_association_cluster.com"
PROTOCOL = ROOT / "DNA_CaOx_simulation_plan.txt"

CHAINS = "AB"
N_WATER = 90
N_FREE_PAIR = 8
N_SEED = 4
SHELL_MIN, SHELL_MAX = 2.40, 4.20
WATER_SPACING = 2.80
RNG = np.random.default_rng(11)

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
    elif element.upper() == "NA":
        aname, element = "NA  ", "Na"
    elif name.startswith("OW"):
        aname = f"{name:<4}"[:4]
    elif name in ("O", "H1", "H2"):
        aname = f" {name:<3}"[:4]
    elif len(name) >= 4:
        aname = name[:4]
    else:
        aname = f" {name:<3}"[:4]
    return (
        f"{rec:<6}{serial:5d} {aname:4s} {resname:<3s} {chain:1s}{resseq:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{occ:6.2f}{bfac:6.2f}"
        f"          {element:>2s}\n"
    )


def random_rotation():
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


def min_dist(pt, cloud):
    return float(np.linalg.norm(cloud - pt, axis=1).min())


def oxalate_template(path: Path):
    raw = parse_atoms(path)
    carbons = [a for a in raw if a["element"].upper() == "C"]
    ccent = np.mean([a["xyz"] for a in carbons], axis=0)
    others = [a for a in raw if a["element"].upper() == "O"]
    others.sort(key=lambda a: np.linalg.norm(a["xyz"] - ccent))
    atoms = carbons + others[:4]
    return [a["xyz"] - ccent for a in atoms]


def load_duplex_and_com():
    atoms = parse_atoms(BACKBONE)
    dna = [a for a in atoms if a["resname"] == "NUC" and a["chain"] in CHAINS]
    p_dna = np.array([a["xyz"] for a in dna if a["element"].upper() == "P"])
    com = [a for a in atoms if a["resname"] == "COM"]
    keep = set()
    for a in com:
        if a["name"].strip().upper() == "CA" and min_dist(a["xyz"], p_dna) <= 5.0:
            keep.add(a["resseq"])
    bound = [a for a in com if a["resseq"] in keep]
    phosphates = [a for a in dna if a["element"].upper() == "P"]
    return dna, bound, sorted(keep), phosphates


def place_near_phosphates(phosphates, n, radius=2.45, existing=None):
    """Place monatomic ions near unused phosphates."""
    existing = [] if existing is None else list(existing)
    pts = []
    for p in phosphates:
        if len(pts) >= n:
            break
        # several random directions; keep the most solvent-exposed (farthest from other P)
        others = np.array([q["xyz"] for q in phosphates if q is not p])
        best = None
        best_sc = -1
        for _ in range(40):
            vec = RNG.normal(size=3)
            vec /= np.linalg.norm(vec)
            trial = p["xyz"] + radius * vec
            if existing and min_dist(trial, np.array(existing + pts)) < 2.2:
                continue
            sc = min_dist(trial, others)
            if sc > best_sc:
                best_sc, best = sc, trial
        if best is not None:
            pts.append(best)
            existing.append(best)
    return pts


def place_in_shell(solute, n, dmin, dmax, sep, existing=None):
    lo = solute.min(0) - 2.0
    hi = solute.max(0) + 2.0
    existing = [] if existing is None else list(existing)
    pts = []
    for _ in range(12000):
        if len(pts) >= n:
            break
        trial = RNG.uniform(lo, hi)
        d = min_dist(trial, solute)
        if not (dmin <= d <= dmax):
            continue
        cloud = existing + pts
        if cloud and min_dist(trial, np.array(cloud)) < sep:
            continue
        pts.append(trial)
    return pts


def pack_waters(solute, n):
    pad = SHELL_MAX + 0.4
    lo, hi = solute.min(0) - pad, solute.max(0) + pad
    xs = np.arange(lo[0], hi[0], WATER_SPACING)
    ys = np.arange(lo[1], hi[1], WATER_SPACING * math.sqrt(3) / 2)
    cands = []
    for iy, y in enumerate(ys):
        xoff = 0.5 * WATER_SPACING if iy % 2 else 0.0
        for x in xs:
            for iz, z in enumerate(np.arange(lo[2], hi[2], WATER_SPACING * 0.82)):
                pt = np.array(
                    [x + xoff, y, z + (0.4 * WATER_SPACING if iz % 2 else 0.0)]
                )
                d = min_dist(pt, solute)
                if SHELL_MIN <= d <= SHELL_MAX:
                    cands.append((d, pt))
    cands.sort(key=lambda t: t[0])
    kept = []
    for _, pt in cands:
        if kept and min_dist(pt, np.array([w[0] for w in kept])) < WATER_SPACING * 0.92:
            continue
        placed = None
        for _ in range(12):
            coords = TIP3P @ random_rotation().T + pt
            if all(min_dist(h, solute) >= 1.55 for h in coords[1:]):
                placed = coords
                break
        if placed is None:
            continue
        kept.append(placed)
        if len(kept) >= n:
            break
    return kept


def write_pdb(path, remarks, groups):
    out = ["HEADER    DNA CAOX ASSEMBLY / GROWTH MODEL\n"]
    for r in remarks:
        line = r.rstrip("\n")
        out.append(line + "\n")
    serial = 0

    def add(rec, name, resname, chain, resseq, xyz, element, bfac=0.0):
        nonlocal serial
        serial += 1
        out.append(
            format_atom(rec, serial, name, resname, chain, resseq, xyz, 1.00, bfac, element)
        )

    for kind, atoms in groups:
        if kind == "TER":
            out.append("TER\n")
            continue
        for a in atoms:
            add(*a)
    out.append("END\n")
    path.write_text("".join(out))


def oxalate_at(center, template, existing_o=None):
    names = [("C1", "C"), ("C2", "C"), ("O1", "O"), ("O2", "O"), ("O3", "O"), ("O4", "O")]
    best, best_sc = None, 1e99
    for _ in range(48):
        R = random_rotation()
        atoms = [(n, el, xyz @ R.T + center) for (n, el), xyz in zip(names, template)]
        if existing_o is None or len(existing_o) == 0:
            return atoms
        oxyz = np.array([xyz for n, el, xyz in atoms if el == "O"])
        d = min_pair(oxyz, existing_o)
        sc = 0.0 if d >= MIN_O_O else (MIN_O_O - d) ** 2
        if sc < best_sc:
            best_sc, best = sc, atoms
        if sc == 0.0:
            return atoms
    return best


def build_ion_sets(dna, phosphates, ox_tmpl, bound_seed=None):
    dna_xyz = np.array([a["xyz"] for a in dna])
    seed_xyz = (
        np.array([a["xyz"] for a in bound_seed])
        if bound_seed
        else np.zeros((0, 3))
    )
    used_p = set()
    if bound_seed:
        seed_ca = [a for a in bound_seed if a["name"].strip().upper() == "CA"]
        for ca in seed_ca:
            nearest = min(phosphates, key=lambda p: np.linalg.norm(p["xyz"] - ca["xyz"]))
            used_p.add(nearest["serial"])
    free_p = [p for p in phosphates if p["serial"] not in used_p]

    n_na = len(phosphates)  # neutralize DNA
    # Only a few Na sit on phosphates so the first water shell survives.
    n_na_contact = min(8, len(free_p), n_na)
    na_phos = place_near_phosphates(free_p, n_na_contact, radius=2.50)
    solute0 = np.vstack([dna_xyz] + ([seed_xyz] if len(seed_xyz) else []))
    if na_phos:
        solute0 = np.vstack([solute0, np.array(na_phos)])
    na_extra = place_in_shell(
        solute0, n_na - len(na_phos), 4.6, 7.5, 3.4, existing=na_phos
    )
    na_pts = na_phos + na_extra

    solute1 = np.vstack([solute0, np.array(na_pts)]) if na_pts else solute0
    seed_ca_pts = (
        [a["xyz"] for a in bound_seed if str(a.get("name", "")).strip().upper() == "CA"]
        if bound_seed
        else []
    )
    ca_pts = place_in_shell(
        solute1, N_FREE_PAIR, 5.0, 8.5, MIN_CA_CA, existing=na_pts + seed_ca_pts
    )
    solute2 = np.vstack([solute1, np.array(ca_pts)]) if ca_pts else solute1
    ox_pts = place_in_shell(solute2, N_FREE_PAIR, 4.8, 8.5, 5.5, existing=ca_pts + na_pts)

    # pair each oxalate loosely with a Ca (contact ion pair, ~3.2–4.5 Å)
    ox_centers = []
    for i, ca in enumerate(ca_pts):
        if i < len(ox_pts):
            vec = ox_pts[i] - ca
            nrm = np.linalg.norm(vec)
            if nrm < 1e-6:
                vec = RNG.normal(size=3)
                nrm = np.linalg.norm(vec)
            # place oxalate centroid ~4.0 Å from Ca
            ox_centers.append(ca + 4.0 * vec / nrm)
        else:
            ox_centers.append(ox_pts[i] if i < len(ox_pts) else ca + np.array([4.0, 0, 0]))

    # First-shell waters around DNA only; ions are filtered out afterward.
    shell_core = dna_xyz if len(seed_xyz) == 0 else np.vstack([dna_xyz, seed_xyz])
    waters = pack_waters(shell_core, N_WATER)
    clash = []
    clash.extend(na_pts)
    clash.extend(ca_pts)
    seed_o = (
        [a["xyz"] for a in bound_seed if str(a.get("element", "")).upper().startswith("O") or str(a.get("name", "")).upper().startswith("O")]
        if bound_seed
        else []
    )
    exist_o = list(seed_o)
    ox_atoms = []
    if ox_centers:
        for center in ox_centers:
            atoms = oxalate_at(center, ox_tmpl, np.array(exist_o) if exist_o else None)
            ox_atoms.append(atoms)
            for _, el, xyz in atoms:
                clash.append(xyz)
                if el == "O":
                    exist_o.append(xyz)
    if clash:
        clash = np.array(clash)
        waters = [w for w in waters if min_dist(w[0], clash) >= MIN_O_O]
    return na_pts, ca_pts, ox_centers, waters, ox_atoms


def dna_group(dna):
    recs = []
    prev = None
    groups = []
    buf = []
    for a in sorted(dna, key=lambda t: (t["chain"], t["resseq"], t["serial"])):
        if prev is not None and a["chain"] != prev:
            groups.append(("atoms", buf))
            groups.append(("TER", []))
            buf = []
        buf.append(
            ("ATOM", a["name"], "NUC", a["chain"], a["resseq"], a["xyz"], a["element"], 0.0)
        )
        prev = a["chain"]
    if buf:
        groups.append(("atoms", buf))
        groups.append(("TER", []))
    return groups


def ion_groups(na_pts, ca_pts, ox_centers, ox_tmpl, waters, seed=None, ox_atoms=None):
    groups = []
    if seed:
        recs = [
            (
                "HETATM",
                a["name"],
                "COM",
                "X",
                a["resseq"],
                a["xyz"],
                a["element"],
                20.0,
            )
            for a in sorted(seed, key=lambda t: (t["resseq"], t["serial"]))
        ]
        groups.append(("atoms", recs))
        groups.append(("TER", []))

    recs = [
        ("HETATM", "NA", "NA", "N", i, xyz, "Na", 10.0)
        for i, xyz in enumerate(na_pts, start=1)
    ]
    if recs:
        groups.append(("atoms", recs))
        groups.append(("TER", []))

    recs = [
        ("HETATM", "CA", "CA", "C", i, xyz, "Ca", 15.0)
        for i, xyz in enumerate(ca_pts, start=1)
    ]
    if recs:
        groups.append(("atoms", recs))
        groups.append(("TER", []))

    recs = []
    for i, center in enumerate(ox_centers, start=1):
        atoms = ox_atoms[i - 1] if ox_atoms and i - 1 < len(ox_atoms) else oxalate_at(center, ox_tmpl)
        for name, el, xyz in atoms:
            recs.append(("HETATM", name, "OXL", "O", i, xyz, el, 15.0))
    if recs:
        groups.append(("atoms", recs))
        groups.append(("TER", []))

    recs = []
    for i, coords in enumerate(waters, start=1):
        recs.append(("HETATM", "O", "HOH", "W", i, coords[0], "O", 0.0))
        recs.append(("HETATM", "H1", "HOH", "W", i, coords[1], "H", 0.0))
        recs.append(("HETATM", "H2", "HOH", "W", i, coords[2], "H", 0.0))
    groups.append(("atoms", recs))
    return groups


def count_atoms(groups):
    n = 0
    for kind, atoms in groups:
        if kind == "atoms":
            n += len(atoms)
    return n


def build_qm_cluster():
    """One bound COM + nearest DNA P/O + nearest free COX + nearby waters."""
    if not SOLVATED.exists():
        return 0
    atoms = parse_atoms(SOLVATED)
    dna = [a for a in atoms if a["resname"] == "NUC"]
    com = [a for a in atoms if a["resname"] == "COM"]
    cox = [a for a in atoms if a["resname"] == "COX"]
    hoh = [a for a in atoms if a["resname"] == "HOH"]
    if not (dna and com and cox):
        return 0

    com_ca = [a for a in com if a["name"].strip().upper() == "CA"]
    p_atoms = [a for a in dna if a["element"].upper() == "P"]
    # pick a bound Ca with a close phosphate and a nearby free Ca
    cox_ca = [a for a in cox if a["name"].strip().upper() == "CA"]
    best = None
    best_sc = 1e9
    for bca in com_ca:
        p = min(p_atoms, key=lambda x: np.linalg.norm(x["xyz"] - bca["xyz"]))
        fca = min(cox_ca, key=lambda x: np.linalg.norm(x["xyz"] - bca["xyz"]))
        dpf = np.linalg.norm(p["xyz"] - bca["xyz"])
        dff = np.linalg.norm(fca["xyz"] - bca["xyz"])
        if dff < 5.5 or dff > 12.0:
            continue
        sc = abs(dff - 8.0) + 0.3 * dpf
        if sc < best_sc:
            best_sc = sc
            best = (bca, p, fca)
    if best is None:
        bca = com_ca[0]
        p = min(p_atoms, key=lambda x: np.linalg.norm(x["xyz"] - bca["xyz"]))
        fca = min(cox_ca, key=lambda x: np.linalg.norm(x["xyz"] - bca["xyz"]))
        best = (bca, p, fca)
    bca, p, fca = best

    bound_res = [a for a in com if a["resseq"] == bca["resseq"]]
    free_res = [a for a in cox if a["resseq"] == fca["resseq"]]
    # DNA atoms within 4.0 Å of the phosphate (one nucleotide-ish)
    core = np.array([p["xyz"], bca["xyz"], fca["xyz"]])
    dna_near = [
        a
        for a in dna
        if min_dist(a["xyz"], np.array([p["xyz"]])) <= 3.6
        or (a["chain"] == p["chain"] and abs(a["resseq"] - p["resseq"]) == 0)
    ]
    # limit DNA atoms
    dna_near.sort(key=lambda a: np.linalg.norm(a["xyz"] - p["xyz"]))
    dna_near = dna_near[:18]

    centers = np.array(
        [a["xyz"] for a in dna_near + bound_res + free_res]
    )
    waters = []
    ho = [a for a in hoh if a["name"] == "O"]
    ho.sort(key=lambda a: min_dist(a["xyz"], centers))
    for o in ho[:12]:
        mates = [x for x in hoh if x["resseq"] == o["resseq"]]
        waters.extend(mates)

    cluster = dna_near + bound_res + free_res + waters
    # unique by serial
    seen = set()
    uniq = []
    for a in cluster:
        if a["serial"] in seen:
            continue
        seen.add(a["serial"])
        uniq.append(a)

    xyz = [f"{len(uniq)}\n", "QM cluster: DNA phosphate + bound CaOx + incoming CaOx + waters\n"]
    for a in uniq:
        el = "Ca" if a["element"].upper() == "CA" else a["element"]
        pnt = a["xyz"]
        xyz.append(f"{el:<2s} {pnt[0]:12.6f} {pnt[1]:12.6f} {pnt[2]:12.6f}\n")
    OUT_QM_XYZ.write_text("".join(xyz))

    charge = estimate_cluster_charge(uniq)
    com_lines = [
        f"%chk=QM_association_cluster.chk",
        "%mem=16GB",
        "%nprocshared=16",
        f"#p B3LYP/6-31G(d) empiricaldispersion=gd3bj scrf=(smd,solvent=water)",
        "",
        "DNA phosphate + bound CaOx + incoming solution CaOx. Single-point / opt as needed.",
        "Check charge/multiplicity before production. This is a starting guess.",
        "",
        f"{charge} 1",
    ]
    for a in uniq:
        el = "Ca" if a["element"].upper() == "CA" else a["element"]
        pnt = a["xyz"]
        com_lines.append(f"{el:<2s} {pnt[0]:12.6f} {pnt[1]:12.6f} {pnt[2]:12.6f}")
    com_lines.append("")
    OUT_QM_COM.write_text("\n".join(com_lines) + "\n")
    return len(uniq)


def estimate_cluster_charge(atoms):
    # rough: phosphate ~ -1, each Ca +2, each oxalate -2, water 0
    n_ca = sum(1 for a in atoms if a["element"].upper() == "CA")
    n_ox = sum(1 for a in atoms if a["resname"] in ("COM", "COX") and a["element"].upper() == "C")
    n_ox_units = n_ox // 2
    n_p = sum(1 for a in atoms if a["element"].upper() == "P")
    return n_p * (-1) + n_ca * 2 + n_ox_units * (-2)


def main():
    dna, bound, keep_res, phosphates = load_duplex_and_com()
    ox_tmpl = oxalate_template(CAOX)

    # --- assembly: no bound COM ---
    na_a, ca_a, ox_a, wat_a, oxat_a = build_ion_sets(dna, phosphates, ox_tmpl, bound_seed=None)
    g_asm = dna_group(dna) + ion_groups(na_a, ca_a, ox_a, ox_tmpl, wat_a, ox_atoms=oxat_a)
    write_pdb(
        OUT_ASM,
        [
            "TITLE     ASSEMBLY START: BARE DNA + FREE CA/OXALATE (NUCLEATION)",
            "REMARK   1 No pre-bound COM. DNA phosphates neutralized with Na+.",
            "REMARK   1 Free Ca2+ and oxalate start in the water shell as ion pairs.",
            "REMARK   2 Use this to test whether DNA organizes Ca at 6.3 A and",
            "REMARK   2 whether oxalate bridges neighboring Ca (whewellite motif).",
            f"REMARK   3 {len(phosphates)} P, {len(na_a)} Na+, {len(ca_a)} Ca2+, {len(ox_a)} oxalate, {len(wat_a)} H2O.",
            f"REMARK   4 Total atoms: {count_atoms(g_asm)}. Geometric start, not equilibrated.",
        ],
        g_asm,
    )

    # --- growth: first N_SEED COM residues along one strand ---
    seed_res = keep_res[:N_SEED]
    seed = [a for a in bound if a["resseq"] in seed_res]
    na_g, ca_g, ox_g, wat_g, oxat_g = build_ion_sets(dna, phosphates, ox_tmpl, bound_seed=seed)
    g_gro = dna_group(dna) + ion_groups(na_g, ca_g, ox_g, ox_tmpl, wat_g, seed=seed, ox_atoms=oxat_g)
    write_pdb(
        OUT_GRO,
        [
            "TITLE     GROWTH START: DNA + SHORT CAOX SEED + FREE IONS",
            f"REMARK   1 Bound COM seed residues {seed_res} (one-strand phosphate ladder).",
            "REMARK   2 Free Ca2+/oxalate can add to the seed. Watch Ca-Ca ~6.3 A",
            "REMARK   2 and oxalate bridges that continue a whewellite-like chain.",
            f"REMARK   3 Seed {len(seed)} atoms; {len(ca_g)} free Ca2+, {len(ox_g)} oxalate, {len(wat_g)} H2O.",
            f"REMARK   4 Total atoms: {count_atoms(g_gro)}. Geometric start, not equilibrated.",
        ],
        g_gro,
    )

    nqm = build_qm_cluster()

    protocol = f"""DNA templating of whewellite — what to simulate
================================================

Question
--------
Can the B-DNA phosphate ladder (P-P ~6.3-7.0 A) pre-organize Ca2+ and
oxalate so that whewellite (COM, Ca-Ca ~6.29 A along c) nucleates
faster than it would in bulk water?

The decorated backbone PDB is the hypothesized PRODUCT. It does not
show how solution CaOx gets there. Use the three starts below.


Models (small droplet, this folder)
-----------------------------------
1. DNA_CaOx_assembly.pdb   {count_atoms(g_asm)} atoms
   Bare DNA + Na+ + {len(ca_a)} Ca2+ + {len(ox_a)} oxalate + {len(wat_a)} H2O
   Nucleation from solution. No bound COM.

2. DNA_CaOx_growth.pdb     {count_atoms(g_gro)} atoms
   DNA + {N_SEED}-site CaOx seed + free Ca2+/oxalate + water
   Growth / elongation along the backbone.

3. QM_association_cluster.xyz / .com   {nqm} atoms
   One phosphate + bound CaOx + incoming CaOx + nearby waters
   Gaussian binding / association energy (not whole-system MD).

4. DNA_CaOx_solvated.pdb
   Coexistence snapshot (fully decorated + free CaOx). Useful as
   an endpoint, not as a nucleation start.

5. DOCS/1BNA.pdb
   Official Drew-Dickerson dodecamer with real DC/DG/DA/DT names.
   Use this (not the CrystalMaker NUC export) for AMBER/GROMACS.


What "templating" looks like
----------------------------
Collect these from MD (and compare to a NO-DNA control with the same
Ca2+/oxalate/water):

  * Ca-OP RDF: inner-sphere peak ~2.3-2.5 A
  * Sequential Ca-Ca along one strand: peak near 6.3 A (COM c-axis)
  * Oxalate bridges between two DNA-bound Ca (COM motif)
  * Longer Ca-Ca ~10 A if a second COM row starts (a-axis)
  * Residence time of Ca on phosphate vs ion-pair lifetime in water
  * Free energy: DNA-bound CaOx vs CaOx(aq)
      umbrella / metadynamics on Ca-OP and Ca-Ca along the backbone

Catalysis (not just binding) means the DNA path has a lower barrier
or a higher flux of COM-like contacts than the no-DNA control.


Recommended compute split
-------------------------
Gaussian (this cluster file, 50-150 atoms)
  Binding energy, inner- vs outer-sphere, oxalate vs phosphate
  preference. B3LYP-D3BJ/6-31G(d) + SMD is a start; refine with
  a larger basis if the cluster group agrees.

Classical MD on a cluster (GROMACS / AMBER / OpenMM)
  Assembly and growth droplets, 50-200 ns, plus enhanced sampling.
  Force fields: DNA OL15 or BSC1; Ca2+ with 12-6-4 or NBFIX;
  oxalate GAFF2 + RESP charges from a Gaussian ESP; TIP3P/OPC water.
  Protonate DNA (reduce/tleap) from 1BNA, not from NUC atom names.

Do not run BOMD in Gaussian on the 1000-atom droplet.


Controls and caveats
--------------------
  * No-DNA box with the same Ca/oxalate concentration
  * Ca-phosphate can nucleate calcium phosphate, not COM — keep
    oxalate-rich and phosphate only on the DNA
  * Current waters are a first shell, not a periodic box; the MD
    group may resolvate in a 10 A box and add counterions properly
  * These coordinates are packed, not energy-minimized
"""
    PROTOCOL.write_text(protocol)
    print(protocol)
    print(f"Wrote {OUT_ASM}")
    print(f"Wrote {OUT_GRO}")
    print(f"Wrote {OUT_QM_XYZ}")
    print(f"Wrote {OUT_QM_COM}")
    print(f"Wrote {PROTOCOL}")


if __name__ == "__main__":
    main()
