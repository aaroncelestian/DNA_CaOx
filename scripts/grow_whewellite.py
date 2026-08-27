#!/usr/bin/env python3
"""
Grow rigid CaOx units from each DNA-bound seed along the COM a-chain.

Each CaC2O4·nH2O is one rigid molecule. New units are accepted only if
Ca–Ca ≥ 6.0 Å and inter-unit O–O ≥ 2.0 Å. That drops the crystallographic
3.84 Å Ca–Ca contact and the overlapping-patch clashes.
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
    MIN_CA_CA,
    MIN_O_O,
    ca_clear,
    is_ca,
    is_oxygen,
    min_pair,
    oo_clear,
    pair_distances,
    short_contact_summary,
    xyz_of,
)

ROOT = Path(__file__).resolve().parents[1]
BACKBONE = ROOT / "DNA_CaOx_backbone.pdb"
XTL = ROOT / "DOCS" / "Whewellite - xtl.pdb"
CAOX = ROOT / "DOCS" / "Whewellite ca_ox.pdb"
OUT_PDB = ROOT / "DNA_CaOx_whewellite_grown.pdb"
OUT_CA = ROOT / "DNA_CaOx_whewellite_grown_Ca_only.pdb"
OUT_NODNA = ROOT / "CaOx_whewellite_noDNA.pdb"
OUT_NODNA_CA = ROOT / "CaOx_whewellite_noDNA_Ca_only.pdb"
REPORT = ROOT / "DNA_CaOx_whewellite_grown_report.txt"

GROW_R = 10.0
MERGE_R = 0.85  # Å — two seeds claiming the same lattice site
SEED_KEEP = 0.70


def parse_atoms(path: Path, het_only=False):
    atoms = []
    cryst = None
    for line in path.open():
        if line.startswith("CRYST1"):
            cryst = dict(
                a=float(line[6:15]),
                b=float(line[15:24]),
                c=float(line[24:33]),
                alpha=float(line[33:40]),
                beta=float(line[40:47]),
                gamma=float(line[47:54]),
            )
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if het_only and not line.startswith("HETATM"):
            continue
        occ_s, bfac_s = line[54:60].strip(), line[60:66].strip()
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
                "occ": float(occ_s) if occ_s else 1.0,
                "bfac": float(bfac_s) if bfac_s else 0.0,
                "element": (line[76:78].strip() or line[12:16].strip()[0]),
            }
        )
    return atoms, cryst


def cell_vectors(cryst):
    a, b, c = cryst["a"], cryst["b"], cryst["c"]
    alpha = math.radians(cryst["alpha"])
    beta = math.radians(cryst["beta"])
    gamma = math.radians(cryst["gamma"])
    av = np.array([a, 0.0, 0.0])
    bv = np.array([b * math.cos(gamma), b * math.sin(gamma), 0.0])
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    cv = np.array([cx, cy, cz])
    return av, bv, cv


def norm(v):
    n = np.linalg.norm(v)
    return v * 0.0 if n < 1e-8 else v / n


def rotation_from_to(a, b):
    a, b = norm(a), norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-10:
        if c > 0:
            return np.eye(3)
        axis = norm(np.cross(a, np.array([1.0, 0.0, 0.0])))
        if np.linalg.norm(axis) < 0.1:
            axis = norm(np.cross(a, np.array([0.0, 1.0, 0.0])))
        return rotation_around(axis, math.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def rotation_around(axis, angle):
    axis = norm(axis)
    x, y, z = axis
    ca, sa = math.cos(angle), math.sin(angle)
    C = 1.0 - ca
    return np.array(
        [
            [ca + x * x * C, x * y * C - z * sa, x * z * C + y * sa],
            [y * x * C + z * sa, ca + y * y * C, y * z * C - x * sa],
            [z * x * C - y * sa, z * y * C + x * sa, ca + z * z * C],
        ]
    )


def format_atom(rec, serial, name, resname, chain, resseq, xyz, occ, bfac, element):
    if element.upper() == "CA":
        aname, element = "CA  ", "Ca"
    elif name.startswith("OW"):
        aname = f"{name:<4}"[:4]
    elif len(name) >= 4:
        aname = name[:4]
    else:
        aname = f" {name:<3}"[:4]
    return (
        f"{rec:<6}{serial:5d} {aname:4s} {resname:<3s} {chain:1s}{resseq:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{occ:6.2f}{bfac:6.2f}"
        f"          {element:>2s}\n"
    )


def expand_crystal(atoms, cryst, nmax=2):
    av, bv, cv = cell_vectors(cryst)
    out = []
    for ia in range(-nmax, nmax + 1):
        for ib in range(-nmax, nmax + 1):
            for ic in range(-nmax, nmax + 1):
                shift = ia * av + ib * bv + ic * cv
                for a in atoms:
                    b = dict(a)
                    b["xyz"] = a["xyz"] + shift
                    b["cell"] = (ia, ib, ic)
                    out.append(b)
    return out, (av, bv, cv)


def unique_ca_distances(ca_xyz, av, bv, cv, cutoff=11.0):
    """Unique Ca–Ca lengths in the COM lattice (with periodic images)."""
    pts = []
    for ia in range(-1, 2):
        for ib in range(-1, 2):
            for ic in range(-1, 2):
                for p in ca_xyz:
                    pts.append(p + ia * av + ib * bv + ic * cv)
    pts = np.array(pts)
    core = ca_xyz
    ds = []
    for p in core:
        d = np.linalg.norm(pts - p, axis=1)
        for x in d:
            if 0.5 < x <= cutoff:
                ds.append(x)
    ds = np.array(ds)
    # bin unique
    uniq = []
    for x in np.sort(ds):
        if not uniq or abs(x - uniq[-1][0]) > 0.15:
            uniq.append([x, 1])
        else:
            uniq[-1][0] = (uniq[-1][0] * uniq[-1][1] + x) / (uniq[-1][1] + 1)
            uniq[-1][1] += 1
    return [(float(a), int(b)) for a, b in uniq]


def fragment_around(ref_xyz, crystal_atoms, radius):
    frag = []
    for a in crystal_atoms:
        if a["element"].upper() == "H":
            continue
        if np.linalg.norm(a["xyz"] - ref_xyz) <= radius + 0.05:
            frag.append(a)
    return frag


def dna_ca_by_strand(atoms):
    """COM Ca residues in residue-number order, split into 11-site strands."""
    cas = [a for a in atoms if a["resname"] == "COM" and a["name"].strip().upper() == "CA"]
    cas.sort(key=lambda a: a["resseq"])
    strands = []
    cur = []
    for a in cas:
        if cur and a["resseq"] != cur[-1]["resseq"] + 1:
            strands.append(cur)
            cur = []
        cur.append(a)
    if cur:
        strands.append(cur)
    return strands


def local_frame(strand, i, helix_origin):
    n = len(strand)
    p = strand[i]["xyz"]
    if 0 < i < n - 1:
        t = strand[i + 1]["xyz"] - strand[i - 1]["xyz"]
    elif i < n - 1:
        t = strand[i + 1]["xyz"] - p
    else:
        t = p - strand[i - 1]["xyz"]
    t = norm(t)
    out = p - helix_origin
    out = out - t * np.dot(out, t)
    if np.linalg.norm(out) < 0.3:
        out = np.array([0.0, 0.0, 1.0])
        out = out - t * np.dot(out, t)
    out = norm(out)
    return t, out


def orient_fragment(frag, ref_xyz, a_crystal, t_dna, out_dna):
    """
    Map crystal so:
      - ref Ca -> DNA Ca
      - crystal a-axis -> DNA backbone tangent
      - remaining freedom: rotate about a so the fragment's
        mass-centroid lies on the solvent side of the DNA.
    """
    R1 = rotation_from_to(a_crystal, t_dna)
    coords = np.array([a["xyz"] for a in frag])
    rel = coords - ref_xyz
    rot1 = rel @ R1.T

    # rotate about t_dna; pick angle that puts centroid most outward
    best_ang, best_score = 0.0, -1e9
    for ang in np.linspace(0, 2 * math.pi, 72, endpoint=False):
        R2 = rotation_around(t_dna, ang)
        r = rot1 @ R2.T
        cent = r.mean(axis=0)
        # prefer centroid along +out_dna; penalize going backward along -out
        score = float(np.dot(cent, out_dna))
        if score > best_score:
            best_score, best_ang = score, ang
    R2 = rotation_around(t_dna, best_ang)
    R = R2 @ R1
    out = []
    for a in frag:
        b = dict(a)
        b["xyz"] = (a["xyz"] - ref_xyz) @ R.T
        out.append(b)
    return out, R


def merge_points(items, cutoff):
    """Greedy cluster with a 3-D hash; keep first, record multiplicity."""
    if not items:
        return []
    inv = 1.0 / cutoff
    buckets = defaultdict(list)
    kept = []
    for it in items:
        p = it["xyz"]
        ijk = (int(math.floor(p[0] * inv)), int(math.floor(p[1] * inv)), int(math.floor(p[2] * inv)))
        hit = None
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    for idx in buckets[(ijk[0] + di, ijk[1] + dj, ijk[2] + dk)]:
                        if np.linalg.norm(p - kept[idx]["xyz"]) <= cutoff:
                            hit = idx
                            break
                    if hit is not None:
                        break
                if hit is not None:
                    break
            if hit is not None:
                break
        if hit is not None:
            kept[hit]["sources"].append(it["source"])
            kept[hit]["n"] += 1
        else:
            rec = dict(it)
            rec["sources"] = [it["source"]]
            rec["n"] = 1
            buckets[ijk].append(len(kept))
            kept.append(rec)
    return kept


def load_rigid_caox(path: Path):
    atoms, _ = parse_atoms(path)
    ca = next(a for a in atoms if a["element"].upper() == "CA")
    carbons = [a for a in atoms if a["element"].upper() == "C"]
    ox_cent = np.mean([a["xyz"] for a in carbons], axis=0)
    rel = []
    for a in atoms:
        if a["element"].upper() == "H":
            continue
        b = dict(a)
        b["rel"] = a["xyz"] - ca["xyz"]
        rel.append(b)
    return {"atoms": rel, "ca_to_ox": ox_cent - ca["xyz"]}


def orient_unit(frag, ca_xyz, axis, twist, ox_dir):
    """Place rigid CaOx at ca_xyz with oxalate along ox_dir, then twist about axis."""
    R0 = rotation_from_to(frag["ca_to_ox"], ox_dir)
    R = rotation_around(axis, twist) @ R0
    out = []
    for a in frag["atoms"]:
        b = dict(a)
        b["xyz"] = a["rel"] @ R.T + ca_xyz
        out.append(b)
    return out


def try_place_unit(frag, ca_xyz, tangent, outward, dna_xyz, dna_o, kept_heavy, kept_o):
    """Rotate a rigid unit until Ca–Ca is already OK and O–O / DNA clear."""
    best, best_sc = None, 1e99
    dirs = [outward, -outward, tangent, -tangent]
    for ox_dir in dirs:
        if np.linalg.norm(ox_dir) < 0.2:
            continue
        for ang in np.linspace(0, 2 * math.pi, 24, endpoint=False):
            atoms = orient_unit(frag, ca_xyz, tangent, ang, ox_dir)
            heavy = xyz_of([a for a in atoms if a["element"].upper() != "CA"])
            oxyz = xyz_of([a for a in atoms if is_oxygen(a)])
            sc = 0.0
            if len(dna_xyz):
                d = min_pair(heavy, dna_xyz)
                if d < DNA_HEAVY:
                    sc += (DNA_HEAVY - d) ** 2 * 20.0
            if len(dna_o):
                d = min_pair(oxyz, dna_o)
                if d < MIN_O_O:
                    sc += (MIN_O_O - d) ** 2 * 40.0
            if len(kept_heavy):
                d = min_pair(heavy, kept_heavy)
                if d < DNA_HEAVY:
                    sc += (DNA_HEAVY - d) ** 2 * 15.0
            if len(kept_o):
                d = min_pair(oxyz, kept_o)
                if d < MIN_O_O:
                    sc += (MIN_O_O - d) ** 2 * 40.0
            if sc < best_sc:
                best_sc, best = sc, atoms
            if sc < 1e-9:
                return atoms, 0.0
    return best, best_sc


def write_ca_only(path, atoms, title):
    cas = [a for a in atoms if is_ca(a)]
    out = [
        "HEADER    CALCIUM SITES ONLY\n",
        f"TITLE     {title}\n",
    ]
    serial = 0
    for a in cas:
        serial += 1
        out.append(
            format_atom(
                "HETATM",
                serial,
                "CA",
                a["resname"],
                a["chain"],
                a["resseq"],
                a["xyz"],
                1.0,
                a.get("bfac", 20.0),
                "Ca",
            )
        )
    out.append("END\n")
    path.write_text("".join(out))


def main():
    dna_atoms, _ = parse_atoms(BACKBONE)
    xtl_atoms, cryst = parse_atoms(XTL)
    if cryst is None:
        raise SystemExit("No CRYST1 in whewellite xtl.pdb")

    expanded, (av, bv, cv) = expand_crystal(xtl_atoms, cryst, nmax=2)
    xtl_ca = np.array(
        [a["xyz"] for a in xtl_atoms if a["element"].upper() == "CA"]
    )
    uniq_d = unique_ca_distances(xtl_ca, av, bv, cv, cutoff=11.0)
    ref = next(a for a in xtl_atoms if a["element"].upper() == "CA")
    frag = load_rigid_caox(CAOX)

    com_atoms = [a for a in dna_atoms if a["resname"] == "COM"]
    nuc = [a for a in dna_atoms if a["resname"] == "NUC"]
    dna_xyz = xyz_of(nuc)
    dna_o = xyz_of([a for a in nuc if is_oxygen(a)])
    strands = dna_ca_by_strand(dna_atoms)

    seed_xyz = xyz_of(
        [a for a in com_atoms if a["name"].strip().upper() == "CA"]
    )
    seed_frames = {}  # resseq -> (R, tangent, outward)

    candidates = []
    for si, strand in enumerate(strands):
        helix_o = np.mean([a["xyz"] for a in strand], axis=0)
        for i, ca in enumerate(strand):
            t, out_vec = local_frame(strand, i, helix_o)
            R1 = rotation_from_to(av, t)
            best_ang, best_sc = 0.0, -1e9
            for ang in np.linspace(0, 2 * math.pi, 72, endpoint=False):
                R = rotation_around(t, ang) @ R1
                sc = float(np.dot(cv @ R.T, out_vec))
                if sc > best_sc:
                    best_sc, best_ang = sc, ang
            R = rotation_around(t, best_ang) @ R1
            seed_frames[ca["resseq"]] = (R, t, out_vec)
            for a in expanded:
                if a["element"].upper() != "CA":
                    continue
                xyz = (a["xyz"] - ref["xyz"]) @ R.T + ca["xyz"]
                dseed = float(np.linalg.norm(xyz - ca["xyz"]))
                if dseed < SEED_KEEP or dseed > GROW_R:
                    continue
                candidates.append({"xyz": xyz, "source": ca["resseq"], "strand": si})

    merged = merge_points(candidates, MERGE_R)
    # Prefer sites already near a seed (inner growth first)
    merged.sort(key=lambda c: float(np.linalg.norm(seed_xyz - c["xyz"], axis=1).min()))

    kept_ca = [p.copy() for p in seed_xyz]
    kept_heavy = [a["xyz"].copy() for a in com_atoms if not is_ca(a)]
    kept_o = [a["xyz"].copy() for a in list(nuc) + list(com_atoms) if is_oxygen(a)]
    grown_units = []
    rejected = {"ca": 0, "dna": 0, "oo": 0}

    for cand in merged:
        if not ca_clear(cand["xyz"], np.array(kept_ca), MIN_CA_CA):
            rejected["ca"] += 1
            continue
        if min_pair(cand["xyz"].reshape(1, 3), dna_xyz) < 2.50:
            rejected["dna"] += 1
            continue
        src = cand["sources"][0]
        _R, tangent, outward = seed_frames.get(src, (np.eye(3), np.array([1.0, 0, 0]), np.array([0, 0, 1.0])))
        unit, sc = try_place_unit(
            frag,
            cand["xyz"],
            tangent,
            outward,
            dna_xyz,
            dna_o,
            np.array(kept_heavy) if kept_heavy else np.zeros((0, 3)),
            np.array(kept_o) if kept_o else np.zeros((0, 3)),
        )
        if unit is None or sc > 1e-6:
            rejected["oo"] += 1
            continue
        grown_units.append({"atoms": unit, "n": cand["n"], "sources": cand["sources"]})
        kept_ca.append(cand["xyz"].copy())
        for a in unit:
            if not is_ca(a):
                kept_heavy.append(a["xyz"].copy())
            if is_oxygen(a):
                kept_o.append(a["xyz"].copy())

    shared = [u for u in grown_units if u["n"] >= 2]
    new_ca = np.array(kept_ca[len(seed_xyz) :]) if len(kept_ca) > len(seed_xyz) else np.zeros((0, 3))
    all_ca = np.array(kept_ca)
    pair_d = pair_distances(all_ca, 11.0)

    def count_near(arr, target, tol=0.25):
        if arr.size == 0:
            return 0
        return int(np.sum(np.abs(arr - target) <= tol))

    com_targets = [u[0] for u in uniq_d if u[0] <= 10.5]

    # Flatten for steric report
    check = []
    for a in nuc:
        check.append({**a, "element": a["element"]})
    for a in com_atoms:
        check.append({**a})
    for i, u in enumerate(grown_units, start=1):
        for a in u["atoms"]:
            check.append(
                {
                    "element": a["element"],
                    "name": a["name"],
                    "resname": "WHW",
                    "chain": "Z",
                    "resseq": i,
                    "xyz": a["xyz"],
                }
            )
    ster = short_contact_summary(check)

    lines = []
    lines.append("Rigid CaOx units grown 10 Å from DNA-bound seeds")
    lines.append("=" * 60)
    lines.append(
        f"COM cell: a={cryst['a']:.3f}  b={cryst['b']:.3f}  "
        f"c={cryst['c']:.3f}  beta={cryst['beta']:.2f}"
    )
    lines.append("COM Ca–Ca lengths ≤ 11 Å (periodic reference):")
    for d, n in uniq_d:
        note = "  [excluded by min Ca-Ca 6.0]" if d < MIN_CA_CA else ""
        lines.append(f"   {d:6.3f} Å   (x{n}){note}")
    lines.append("")
    lines.append("Each CaOx is a rigid molecule (DOCS/Whewellite ca_ox.pdb).")
    lines.append(f"Accepted only if Ca–Ca ≥ {MIN_CA_CA:.1f} Å and inter-unit O–O ≥ {MIN_O_O:.1f} Å.")
    lines.append("")
    lines.append(f"DNA-bound Ca seeds     : {len(seed_xyz)}")
    lines.append(f"Grown rigid CaOx units : {len(grown_units)}")
    lines.append(f"Sites claimed by ≥2 seeds: {len(shared)}")
    lines.append(
        f"Rejected candidates    : Ca–Ca {rejected['ca']}, DNA {rejected['dna']}, "
        f"O–O/orient {rejected['oo']}"
    )
    if shared:
        lines.append("  Shared sites (resseq sources):")
        for s in shared[:16]:
            src = ",".join(str(x) for x in sorted(set(s["sources"])))
            ca0 = next(a["xyz"] for a in s["atoms"] if is_ca(a))
            lines.append(f"    n={s['n']}  sources {src}  xyz {np.asarray(ca0).round(2)}")
        if len(shared) > 16:
            lines.append(f"    ... {len(shared) - 16} more")
    lines.append("")
    lines.append("Ca–Ca pairs ≤ 11 Å in the grown model vs COM:")
    if pair_d.size:
        for tgt in com_targets[:8]:
            n = count_near(pair_d, tgt, 0.25)
            lines.append(f"   COM {tgt:5.2f} Å  ±0.25  →  {n:4d} pairs in model")
        lines.append(
            f"   Model pairs: n={len(pair_d)}  min={pair_d.min():.2f}  "
            f"median={np.median(pair_d):.2f}  max={pair_d.max():.2f}"
        )
    lines.append("")
    n629 = 0
    if len(new_ca):
        for s in seed_xyz:
            d = np.linalg.norm(new_ca - s, axis=1)
            n629 += int(np.any(np.abs(d - 6.29) < 0.30))
    lines.append(
        f"DNA-bound Ca with a grown neighbor at COM 6.29 Å: {n629}/{len(seed_xyz)}"
    )
    lines.append("")
    lines.append("Steric check (inter-residue):")
    lines.append(
        f"  Ca–Ca < 6.0 Å: {ster['n_ca_short']}"
        + (f"  min {ster['ca_min']:.3f}" if ster["ca_min"] is not None else "  none")
    )
    lines.append(
        f"  O–O < 2.0 Å  : {ster['n_oo_short']}"
        + (f"  min {ster['oo_min']:.3f}" if ster["oo_min"] is not None else "  none")
    )
    lines.append("")
    lines.append("Alignment: COM a-axis (6.29 Å) || local DNA backbone tangent.")
    lines.append("3.84 Å intra-cell Ca–Ca is not used (below the 6 Å floor).")
    report = "\n".join(lines) + "\n"
    REPORT.write_text(report)
    print(report)

    remarks = [
        "HEADER    DNA-TEMPLATED RIGID CAOX GROWTH\n",
        "TITLE     RIGID CAOX UNITS, CA-CA >= 6 A, O-O >= 2 A\n",
        "REMARK   1 Each CaOx is a rigid molecule. Grown only at COM lattice\n",
        "REMARK   1 sites that keep Ca-Ca >= 6.0 A and inter-unit O-O >= 2.0 A.\n",
        "REMARK   2 Chain X = DNA-bound COM. Chain Z = grown rigid WHW units.\n",
        f"REMARK   3 Grown units {len(grown_units)}; shared sites {len(shared)}.\n",
        f"REMARK   3 COM cell a={cryst['a']:.3f} b={cryst['b']:.3f} c={cryst['c']:.3f} "
        f"beta={cryst['beta']:.2f}\n",
    ]

    def emit(include_dna, path):
        out = list(remarks)
        serial = 0

        def add(rec, name, resname, chain, resseq, xyz, element, bfac, occ=1.0):
            nonlocal serial
            serial += 1
            out.append(
                format_atom(rec, serial, name, resname, chain, resseq, xyz, occ, bfac, element)
            )

        if include_dna:
            prev = None
            for a in sorted(nuc, key=lambda t: (t["chain"], t["resseq"], t["serial"])):
                if prev is not None and a["chain"] != prev:
                    out.append("TER\n")
                add("ATOM", a["name"], "NUC", a["chain"], a["resseq"], a["xyz"], a["element"], 0.0)
                prev = a["chain"]
            out.append("TER\n")

        for a in sorted(com_atoms, key=lambda t: (t["resseq"], t["serial"])):
            add("HETATM", a["name"], "COM", "X", a["resseq"], a["xyz"], a["element"], 20.0)
        out.append("TER\n")

        for i, u in enumerate(grown_units, start=1):
            bfac = 10.0 if u["n"] >= 2 else 30.0
            carbons = [a["xyz"] for a in u["atoms"] if a["element"].upper() == "C"]
            ccent = np.mean(carbons, axis=0) if carbons else u["atoms"][0]["xyz"]
            c_i = o_ox = o_w = 0
            for a in u["atoms"]:
                el = a["element"]
                if is_ca(a):
                    name, el = "CA", "Ca"
                elif el.upper() == "C":
                    c_i += 1
                    name = f"C{c_i}"
                elif el.upper().startswith("O"):
                    if float(np.linalg.norm(a["xyz"] - ccent)) < 2.4:
                        o_ox += 1
                        name = f"O{o_ox}"
                    else:
                        o_w += 1
                        name = f"OW{o_w}"
                else:
                    name = a["name"]
                add("HETATM", name, "WHW", "Z", i, a["xyz"], el, bfac)
        out.append("END\n")
        path.write_text("".join(out))

    emit(True, OUT_PDB)
    emit(False, OUT_NODNA)

    all_written = []
    for a in com_atoms:
        all_written.append({**a, "bfac": 20.0})
    for i, u in enumerate(grown_units, start=1):
        bfac = 10.0 if u["n"] >= 2 else 30.0
        for a in u["atoms"]:
            all_written.append(
                {
                    "name": a["name"],
                    "resname": "WHW",
                    "chain": "Z",
                    "resseq": i,
                    "xyz": a["xyz"],
                    "element": a["element"],
                    "bfac": bfac,
                }
            )
    write_ca_only(OUT_CA, [a for a in all_written if True], "GROWN MODEL CA SITES")
    write_ca_only(
        OUT_NODNA_CA,
        [a for a in all_written if a["resname"] in ("COM", "WHW")],
        "NO-DNA CA SITES",
    )
    print(f"Wrote {OUT_PDB}")
    print(f"Wrote {OUT_NODNA}")
    print(f"Wrote {OUT_CA}")
    print(f"Wrote {OUT_NODNA_CA}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
