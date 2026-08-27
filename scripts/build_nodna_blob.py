#!/usr/bin/env python3
"""
Density-matched no-DNA CaOx blob from a thin gel PDB.

Same number of intact COM/WHW units, randomly packed in a sphere whose
volume matches the DNA-length coat — COM 3.84 Å contacts are allowed
(MIN_CA_CA_COM). This is the control: does pair order appear without DNA?
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geom_constraints import MIN_CA_CA_COM, is_ca  # noqa: E402
from grow_whewellite import format_atom, parse_atoms  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(2026)


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


def units_from_pdb(atoms):
    by = defaultdict(list)
    for a in atoms:
        if a["resname"] not in ("COM", "WHW"):
            continue
        by[(a["chain"], a["resseq"], a["resname"])].append(a)
    out = []
    for key in sorted(by, key=lambda k: (k[0], k[1])):
        recs = by[key]
        ca = next((a for a in recs if is_ca(a)), None)
        if ca is None:
            continue
        rel = [dict(a, xyz=np.asarray(a["xyz"], float) - ca["xyz"]) for a in recs]
        out.append(rel)
    return out


def coat_sphere_radius(dna_atoms, n_units) -> float:
    """Sphere with volume ≈ DNA cylinder (r_dna+4) × length, at least packing r."""
    if dna_atoms:
        xyz = np.array([a["xyz"] for a in dna_atoms])
        origin = xyz.mean(axis=0)
        _, _, vh = np.linalg.svd(xyz - origin)
        axis = vh[0]
        rel = xyz - origin
        z = rel @ axis
        r = np.linalg.norm(rel - np.outer(z, axis), axis=1)
        r_cyl = float(np.percentile(r, 90)) + 8.0
        h = float(z.max() - z.min()) + 6.0
        vol = math.pi * r_cyl**2 * max(h, 20.0)
    else:
        vol = n_units * (4.0 / 3.0) * math.pi * (4.5**3)
    r_sph = (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0)
    r_pack = 3.2 * (n_units ** (1.0 / 3.0))
    return float(max(r_sph, r_pack, 12.0))


def waters_from_pdb(atoms):
    return [a for a in atoms if a["resname"] == "HOH"]


def place_waters(pts, radius, solute_ca, min_oo=2.15, tries=400):
    placed = []
    for src in pts:
        ok = None
        for _ in range(tries):
            u = RNG.normal(size=3)
            u /= max(float(np.linalg.norm(u)), 1e-8)
            r = radius * (RNG.random() ** (1.0 / 3.0))
            xyz = u * r
            if solute_ca and float(np.linalg.norm(np.array(solute_ca) - xyz, axis=1).min()) < 2.2:
                continue
            if placed and float(np.linalg.norm(np.array(placed) - xyz, axis=1).min()) < min_oo:
                continue
            ok = xyz
            break
        if ok is None:
            u = RNG.normal(size=3)
            u /= max(float(np.linalg.norm(u)), 1e-8)
            ok = u * (radius + 1.5)
        placed.append(ok)
    return placed


def place_units(units, radius, min_ca=MIN_CA_CA_COM, tries=800):
    placed_ca = []
    placed = []
    for unit in units:
        ok = None
        for _ in range(tries):
            # uniform in sphere
            u = RNG.normal(size=3)
            u /= max(float(np.linalg.norm(u)), 1e-8)
            r = radius * (RNG.random() ** (1.0 / 3.0))
            ca = u * r
            if placed_ca:
                d = np.linalg.norm(np.array(placed_ca) - ca, axis=1).min()
                if d < min_ca:
                    continue
            R = random_rotation()
            atoms = []
            for a in unit:
                b = dict(a)
                b["xyz"] = a["xyz"] @ R.T + ca
                atoms.append(b)
            ok = atoms
            break
        if ok is None:
            # last resort: push out along a random direction
            u = RNG.normal(size=3)
            u /= max(float(np.linalg.norm(u)), 1e-8)
            ca = u * (radius + 2.0 + 0.4 * len(placed_ca))
            R = random_rotation()
            ok = []
            for a in unit:
                b = dict(a)
                b["xyz"] = a["xyz"] @ R.T + ca
                ok.append(b)
        placed.append(ok)
        placed_ca.append(next(a["xyz"] for a in ok if is_ca(a)))
    return placed


def write_blob(path: Path, units, remarks, waters=None):
    serial = 1
    lines = list(remarks)
    for resseq, atoms in enumerate(units, start=1):
        for a in atoms:
            el = a.get("element") or a["name"][:1]
            lines.append(
                format_atom(
                    "HETATM",
                    serial,
                    a["name"],
                    "COM",
                    "X",
                    resseq,
                    a["xyz"],
                    1.0,
                    20.0,
                    el,
                )
            )
            serial += 1
    if waters:
        w0 = len(units) + 1
        for i, xyz in enumerate(waters):
            lines.append(
                format_atom(
                    "HETATM",
                    serial,
                    "OW",
                    "HOH",
                    "W",
                    w0 + i,
                    xyz,
                    1.0,
                    30.0,
                    "O",
                )
            )
            serial += 1
    lines.append("END\n")
    path.write_text("".join(lines))


def main():
    ap = argparse.ArgumentParser(description="No-DNA density-matched CaOx blob")
    ap.add_argument("--gel", type=Path, required=True)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()
    atoms, _ = parse_atoms(args.gel)
    dna = [a for a in atoms if a["resname"] == "NUC"]
    units = units_from_pdb(atoms)
    if not units:
        raise SystemExit(f"No COM/WHW units in {args.gel}")
    radius = coat_sphere_radius(dna, len(units))
    packed = place_units(units, radius)
    hoh = waters_from_pdb(atoms)
    ca_xyz = [next(a["xyz"] for a in u if is_ca(a)) for u in packed]
    water_pts = place_waters([a["xyz"] for a in hoh], radius, ca_xyz) if hoh else []
    out = args.output or ROOT / (args.gel.stem.replace("_omm", "") + "_nodna.pdb")
    remarks = [
        "HEADER    CAOX BLOB NO DNA (DENSITY-MATCHED CONTROL)\n",
        f"REMARK   1 Source gel: {args.gel.name}\n",
        f"REMARK   2 {len(units)} COM units in a sphere r={radius:.1f} A\n",
        f"REMARK   3 {len(water_pts)} extra water O (HOH)\n",
        f"REMARK   4 Ca-Ca floor {MIN_CA_CA_COM:.2f} A (COM 3.84 allowed)\n",
        "REMARK   5 Relax with fire_openmm --no-com-targets\n",
    ]
    write_blob(out, packed, remarks, waters=water_pts)
    print(
        f"Wrote {out}  ({len(units)} units, {len(water_pts)} HOH, "
        f"sphere r={radius:.1f} Å)"
    )


if __name__ == "__main__":
    main()
