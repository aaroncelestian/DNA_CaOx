#!/usr/bin/env python3
"""
Pack a disordered CaOx + water shell around the relaxed gel-first model.

Keeps gel WHW residues 1–N fixed (already relaxed). Places additional
rigid CaC2O4·nH2O units and water O sites in the annulus

    SHELL_INNER Å ≤ d(point, gel heavy) ≤ SHELL_OUTER Å

(default 2.25–15 Å from the gel envelope), then writes a PDB for
fire_openmm_caox.py with --freeze-resseq-le N --no-com-targets.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dna_caox import (  # noqa: E402
    CAOX_PDB,
    clash_score,
    load_caox_fragment,
    transform_fragment,
)
from geom_constraints import (  # noqa: E402
    DNA_HEAVY,
    MIN_CA_CA,
    MIN_O_O,
    is_ca,
    is_oxygen,
    short_contact_summary,
    xyz_of,
)
from grow_whewellite import (  # noqa: E402
    XTL,
    cell_vectors,
    load_rigid_caox,
    local_frame,
    orient_unit,
    parse_atoms,
)
from grow_crystal_from_growth import alignment_matrix  # noqa: E402
from build_whewellite_patch import cut_crystallite_patch  # noqa: E402
from relax_whewellite_units import write_pdb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEL = ROOT / "DNA_CaOx_gel_altP_geom_omm.pdb"
DEFAULT_OUT = ROOT / "DNA_CaOx_gel_altP_geom_shell_lattice.pdb"
DEFAULT_OUT_SEEDED = ROOT / "DNA_CaOx_gel_altP_geom_shell_lattice_seeded.pdb"
DEFAULT_REPORT = ROOT / "DNA_CaOx_gel_altP_geom_shell_lattice_report.txt"
DEFAULT_REPORT_SEEDED = ROOT / "DNA_CaOx_gel_altP_geom_shell_lattice_seeded_report.txt"

SHELL_INNER = 2.25
SHELL_OUTER = 30.0
SHELL_INNER_GEL = 3.5
SHELL_INNER_DENSE = 6.0
SHELL_OUTER_DENSE = 10.0
SHELL_MID_INNER = 10.0
SHELL_MID_OUTER = 18.0
SHELL_BULK_INNER = 18.0
SHELL_BULK_OUTER = 30.0
# Radial phase zones (bfac tags for analysis / viewer)
BFAC_GEL_COAT = 18.0
BFAC_INTERMEDIATE = 22.0
BFAC_PRECRYSTAL = 24.0
BFAC_BULK = 26.0
SEED_ZONE_INNER = 27.0
SEED_ZONE_OUTER = 30.0
SEED_RADIUS = 10.0
SEED_EXCLUDE_CA = 3.5
SEED_BFAC = 12.0
WATER_SPACING = 2.35
LATTICE_MERGE = 1.2
RNG = np.random.default_rng(2026)

SATURATION = {
    "normal": {"max_units": 280, "max_waters": 360, "lattice_nmax": 2},
    "high": {"max_units": 850, "max_waters": 600, "lattice_nmax": 5},
    "packed": {
        "max_units": 2800,
        "max_waters": 1000,
        "lattice_nmax": 6,
        "zone_budgets": (300, 900, 1200),
    },
    "saturated": {
        "max_units": 9000,
        "max_waters": 3500,
        "lattice_nmax": 9,
        "zone_budgets": (400, 2200, 3500, 3200),
        "min_ca_sep": 3.85,
        "disordered_spacing": 2.0,
    },
}

# DNA → gel coat → intermediate (templated) → outer random orientations
RADIAL_ZONES = (
    {"inner": SHELL_INNER_GEL, "outer": 8.0, "bfac": BFAC_GEL_COAT, "disordered": True},
    {
        "inner": 8.0,
        "outer": 15.0,
        "bfac": BFAC_INTERMEDIATE,
        "disordered": False,
        "random_orient": False,
    },
    {
        "inner": 15.0,
        "outer": 22.0,
        "bfac": BFAC_PRECRYSTAL,
        "disordered": False,
        "random_orient": True,
        "random_frame": True,
    },
    {
        "inner": 22.0,
        "outer": 30.0,
        "bfac": BFAC_BULK,
        "disordered": False,
        "random_orient": True,
        "random_frame": True,
    },
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


def unit_clash(atoms, dna_xyz, dna_o, other_heavy, other_o) -> float:
    lig = np.array([a["xyz"] for a in atoms if a["element"].upper() != "CA"])
    lig_o = np.array([a["xyz"] for a in atoms if is_oxygen(a)])
    return clash_score(lig, lig_o, dna_xyz, dna_o, other_heavy, other_o)


def clone_fragment(frag, ca_xyz, outward, twist):
    atoms = transform_fragment(frag, ca_xyz, outward, twist)
    out = []
    for a in atoms:
        b = dict(a)
        b["resname"] = "WHW"
        b["chain"] = "X"
        out.append(b)
    return out


def format_shell_atoms(frag, placements, start_resseq: int):
    carbon_ids = {a["serial"] for a in frag["carbons"]}
    oxo_ids = {a["serial"] for a in frag["ox_o"]}
    water_ids = {a["serial"] for a in frag["waters"]}
    ca_id = frag["ca"]["serial"]
    shell_atoms = []
    resseq = start_resseq
    for unit_atoms in placements:
        c_i = o_ox = o_w = 0
        for a in unit_atoms:
            if a["serial"] == ca_id:
                name, el = "CA", "Ca"
            elif a["serial"] in carbon_ids:
                c_i += 1
                name, el = f"C{c_i}", "C"
            elif a["serial"] in oxo_ids:
                o_ox += 1
                name, el = f"O{o_ox}", "O"
            elif a["serial"] in water_ids:
                o_w += 1
                name, el = f"OW{o_w}", "O"
            else:
                name, el = a["name"][:4], a["element"]
            shell_atoms.append(
                {
                    "name": name,
                    "resname": "WHW",
                    "chain": "X",
                    "resseq": resseq,
                    "xyz": a["xyz"].copy(),
                    "element": el,
                    "bfac": 25.0,
                }
            )
        resseq += 1
    return shell_atoms, resseq


def format_oriented_units(placements, start_resseq: int):
    """Format orient_unit() output (no fragment serials)."""
    shell_atoms = []
    resseq = start_resseq
    for unit_atoms in placements:
        c_i = o_ox = o_w = 0
        unit_bfac = float(unit_atoms[0].get("bfac", 25.0)) if unit_atoms else 25.0
        for a in unit_atoms:
            el = a["element"].upper()
            if el == "CA":
                name, elem = "CA", "Ca"
            elif el == "C":
                c_i += 1
                name, elem = f"C{c_i}", "C"
            elif el == "O":
                if o_ox < 4:
                    o_ox += 1
                    name, elem = f"O{o_ox}", "O"
                else:
                    o_w += 1
                    name, elem = f"OW{o_w}", "O"
            shell_atoms.append(
                {
                    "name": name,
                    "resname": "WHW",
                    "chain": "X",
                    "resseq": resseq,
                    "xyz": np.asarray(a["xyz"], float).copy(),
                    "element": elem,
                    "bfac": float(a.get("bfac", unit_bfac)),
                }
            )
        resseq += 1
    return shell_atoms, resseq


def gel_ca_seeds(gel_whw, dna, origin, axis):
    """Per-gel-unit Ca with helix tangent and outward for lattice alignment."""
    by_res = defaultdict(list)
    for a in gel_whw:
        by_res[a["resseq"]].append(a)
    p_atoms = sorted(
        [a for a in dna if a["name"].strip().upper() == "P"],
        key=lambda a: (a["chain"], a["resseq"]),
    )
    p_xyz = np.array([a["xyz"] for a in p_atoms], float)
    seeds = []
    for res in sorted(by_res):
        cas = [a for a in by_res[res] if is_ca(a)]
        if not cas:
            continue
        ca_xyz = cas[0]["xyz"]
        d = np.linalg.norm(p_xyz - ca_xyz, axis=1)
        pi = int(d.argmin())
        p = p_atoms[pi]
        rel = p["xyz"] - origin
        t_ax = float(rel @ axis)
        radial = rel - axis * t_ax
        outward = radial / max(float(np.linalg.norm(radial)), 1e-8)
        same = [a for a in p_atoms if a["chain"] == p["chain"]]
        same.sort(key=lambda a: a["resseq"])
        idx = same.index(p)
        fake = [{"xyz": a["xyz"]} for a in same]
        tangent, out2 = local_frame(fake, idx, origin)
        if float(np.dot(outward, out2)) < 0:
            outward = -outward
        seeds.append(
            {
                "resseq": res,
                "xyz": ca_xyz.copy(),
                "tangent": tangent,
                "outward": outward,
            }
        )
    return seeds


def disordered_shell_candidates(
    seeds,
    gel_tree: cKDTree,
    shell_inner: float,
    shell_outer: float,
    spacing: float = 2.2,
):
    """Dense amorphous coat: CaOx sites along gel outward rays (random twist)."""
    cands = []
    seen = set()
    for seed in seeds:
        out = np.asarray(seed["outward"], float)
        tan = np.asarray(seed["tangent"], float)
        perp = np.cross(out, tan)
        pn = float(np.linalg.norm(perp))
        if pn > 1e-8:
            perp /= pn
        binorm = np.cross(out, perp)
        bn = float(np.linalg.norm(binorm))
        if bn > 1e-8:
            binorm /= bn
        for step in np.arange(shell_inner, shell_outer + 0.05, spacing):
            for az in np.linspace(0, 2 * math.pi, 14, endpoint=False):
                jitter = perp * (math.cos(az) * 1.1) + binorm * (math.sin(az) * 1.1)
                pos = seed["xyz"] + out * step + jitter
                d_gel = float(gel_tree.query(pos)[0])
                if not (shell_inner <= d_gel <= shell_outer):
                    continue
                key = tuple(np.round(pos, 1))
                if key in seen:
                    continue
                seen.add(key)
                cands.append(
                    {
                        "xyz": pos,
                        "tangent": seed["tangent"],
                        "outward": seed["outward"],
                        "twist": float(RNG.uniform(0, 2 * math.pi)),
                        "d_gel": d_gel,
                    }
                )
    cands.sort(key=lambda c: c["d_gel"])
    return cands


def lattice_shell_candidates(
    seeds,
    gel_tree: cKDTree,
    shell_inner: float,
    shell_outer: float,
    nmax: int,
):
    """Exact COM lattice translations from each gel Ca seed (a,b,c steps)."""
    _xtl_atoms, cryst = parse_atoms(XTL)
    av, bv, cv = cell_vectors(cryst)
    steps = []
    for ia in range(-nmax, nmax + 1):
        for ib in range(-nmax, nmax + 1):
            for ic in range(-nmax, nmax + 1):
                if ia == ib == ic == 0:
                    continue
                off = ia * av + ib * bv + ic * cv
                r = float(np.linalg.norm(off))
                if 2.5 < r <= shell_outer + 2.0:
                    steps.append((r, off))
    steps.sort(key=lambda t: t[0])

    cands = []
    seen_ca = []
    for seed in seeds:
        R = alignment_matrix(av, cv, seed["tangent"], seed["outward"])
        for _r, off in steps:
            pos = seed["xyz"] + off @ R.T
            d_gel, _ = gel_tree.query(pos)
            if not (shell_inner <= float(d_gel) <= shell_outer):
                continue
            if seen_ca and min(float(np.linalg.norm(pos - s)) for s in seen_ca) < (
                MIN_CA_CA - 0.15
            ):
                continue
            seen_ca.append(pos.copy())
            cands.append(
                {
                    "xyz": pos,
                    "tangent": seed["tangent"],
                    "outward": seed["outward"],
                    "twist": 0.0,
                    "d_gel": float(d_gel),
                }
            )
    cands.sort(key=lambda c: c["d_gel"])
    return cands


def strip_waters(unit_atoms):
    return [a for a in unit_atoms if not a["name"].startswith("OW")]


def pick_crystal_seed_site(
    seeds,
    gel_tree: cKDTree,
    zone_inner: float,
    zone_outer: float,
    nmax: int,
    anchor_resseq: int | None = None,
):
    """Lattice COM offset placing a whewellite patch center in the bulk annulus."""
    _xtl_atoms, cryst = parse_atoms(XTL)
    av, bv, cv = cell_vectors(cryst)
    if anchor_resseq is not None:
        anchor = next((s for s in seeds if s["resseq"] == anchor_resseq), None)
        if anchor is None:
            raise ValueError(f"gel anchor resseq {anchor_resseq} not found")
        anchors = [anchor]
    else:
        anchors = [seeds[len(seeds) // 2]]

    best = None
    for anchor in anchors:
        R = alignment_matrix(av, cv, anchor["tangent"], anchor["outward"])
        for ia in range(-nmax, nmax + 1):
            for ib in range(-nmax, nmax + 1):
                for ic in range(-nmax, nmax + 1):
                    if ia == ib == ic == 0:
                        continue
                    off = ia * av + ib * bv + ic * cv
                    r = float(np.linalg.norm(off))
                    if r < 8.0:
                        continue
                    pos = anchor["xyz"] + off @ R.T
                    d_gel = float(gel_tree.query(pos)[0])
                    if zone_inner <= d_gel <= zone_outer:
                        out_sc = float(np.dot(pos - anchor["xyz"], anchor["outward"]))
                        score = (d_gel, out_sc)
                        if best is None or score > best[0]:
                            best = (score, pos, anchor["tangent"], anchor["outward"], (ia, ib, ic))
    if best is None:
        return None
    _score, pos, tangent, outward, abc = best
    return {
        "xyz": pos,
        "tangent": tangent,
        "outward": outward,
        "d_gel": float(gel_tree.query(pos)[0]),
        "abc": abc,
    }


def filter_units_near_patch(units, patch_atoms, min_ca: float = SEED_EXCLUDE_CA):
    patch_ca = xyz_of([a for a in patch_atoms if is_ca(a)])
    if len(patch_ca) == 0:
        return units
    kept = []
    for unit_atoms in units:
        ca = next(a["xyz"] for a in unit_atoms if is_ca(a))
        if float(np.linalg.norm(patch_ca - ca, axis=1).min()) >= min_ca:
            kept.append(unit_atoms)
    return kept


def format_crystal_seed(patch_atoms, start_resseq: int):
    """One WHW residue per Ca; bfac marks crystalline seed."""
    cas = sorted(
        [a for a in patch_atoms if is_ca(a)],
        key=lambda a: float(np.linalg.norm(a["xyz"])),
    )
    cxyz = xyz_of(cas)
    by_local: dict[int, list] = defaultdict(list)
    for a in patch_atoms:
        if is_ca(a):
            local = cas.index(a) + 1
        else:
            local = int(np.argmin(np.linalg.norm(cxyz - a["xyz"], axis=1))) + 1
        by_local[local].append(a)

    shell_atoms = []
    resseq = start_resseq
    for local in sorted(by_local):
        c_i = o_ox = o_w = 0
        for a in by_local[local]:
            el = a["element"].upper()
            if el == "CA":
                name, elem = "CA", "Ca"
            elif el == "C":
                c_i += 1
                name, elem = f"C{c_i}", "C"
            elif el == "O":
                if o_ox < 4:
                    o_ox += 1
                    name, elem = f"O{o_ox}", "O"
                else:
                    o_w += 1
                    name, elem = f"OW{o_w}", "O"
            else:
                name, elem = a.get("name", "X")[:4], a["element"]
            shell_atoms.append(
                {
                    "name": name,
                    "resname": "WHW",
                    "chain": "X",
                    "resseq": resseq,
                    "xyz": np.asarray(a["xyz"], float).copy(),
                    "element": elem,
                    "bfac": SEED_BFAC,
                }
            )
        resseq += 1
    return shell_atoms, resseq


def place_lattice_units(
    cands,
    frag,
    dna_xyz,
    dna_o,
    placed_heavy,
    placed_o,
    n_twist=12,
    clash_max=8.0,
    max_placements: int | None = None,
    min_ca_sep: float | None = None,
    whw_frag=None,
):
    units = []
    heavy = list(placed_heavy)
    oxy = list(placed_o)
    unit_bfac = float(cands[0].get("bfac", 25.0)) if cands else 25.0
    for cand in cands:
        if max_placements is not None and len(units) >= max_placements:
            break
        ca_xyz = cand["xyz"]
        tangent = cand["tangent"]
        outward = cand["outward"]
        bfac = float(cand.get("bfac", unit_bfac))
        best_atoms, best_sc = None, 1e99
        other = np.array(heavy) if heavy else np.zeros((0, 3))
        other_o = np.array(oxy) if oxy else np.zeros((0, 3))
        trials: list[tuple[float, float, np.ndarray | None]] = []
        if cand.get("random_orient"):
            n_rand = int(cand.get("random_twists", 10))
            for _ in range(n_rand):
                if cand.get("random_frame"):
                    outward = RNG.normal(size=3)
                    outward /= max(float(np.linalg.norm(outward)), 1e-8)
                    twist = float(RNG.uniform(0, 2 * math.pi))
                    trials.append((twist, 0.0, outward))
                else:
                    twist = float(RNG.uniform(0, 2 * math.pi))
                    trials.append((twist, 0.0, None))
        elif cand.get("disordered"):
            trials = [(float(cand.get("twist", 0.0)), 0.0, None)]
        else:
            base_twist = float(cand.get("twist", 0.0))
            trials = [
                (base_twist + k * (2 * math.pi / n_twist), 0.0, None)
                for k in range(n_twist)
            ]
        for twist, _junk, out_override in trials:
            outward = (
                np.asarray(out_override, float)
                if out_override is not None
                else np.asarray(cand["outward"], float)
            )
            if cand.get("random_frame"):
                trial = clone_fragment(frag, ca_xyz, outward, twist)
            elif whw_frag is not None:
                trial = orient_unit(whw_frag, ca_xyz, cand["tangent"], twist, outward)
            else:
                trial = clone_fragment(frag, ca_xyz, outward, twist)
            trial_atoms = []
            for a in trial:
                b = dict(a)
                b["resname"] = "WHW"
                b["chain"] = "X"
                b["element"] = a.get("element", a["name"][:1])
                b["bfac"] = bfac
                trial_atoms.append(b)
            sc = unit_clash(trial_atoms, dna_xyz, dna_o, other, other_o)
            if sc < best_sc:
                best_sc, best_atoms = sc, trial_atoms
        if best_atoms is None or best_sc > clash_max:
            continue
        if heavy:
            d_ca = float(np.linalg.norm(np.array(heavy) - ca_xyz, axis=1).min())
            if min_ca_sep is not None:
                min_sep = float(min_ca_sep)
            else:
                min_sep = 3.45 if cand.get("disordered") else 4.05
            if d_ca < min_sep:
                continue
        best_atoms = strip_waters(best_atoms)
        units.append(best_atoms)
        for a in best_atoms:
            if a["element"].upper() != "H":
                heavy.append(a["xyz"].copy())
            if is_oxygen(a):
                oxy.append(a["xyz"].copy())
    return units, heavy, oxy


def pack_shell_waters(
    gel_tree: cKDTree,
    solute_xyz: np.ndarray,
    n_target: int,
    inner: float,
    outer: float,
    start_resseq: int,
):
    lo = solute_xyz.min(0) - outer - 2.0
    hi = solute_xyz.max(0) + outer + 2.0
    kept = []
    resseq = start_resseq
    spacing = WATER_SPACING
    xs = np.arange(lo[0], hi[0], spacing)
    ys = np.arange(lo[1], hi[1], spacing * math.sqrt(3) / 2)
    cands = []
    for iy, y in enumerate(ys):
        xoff = 0.5 * spacing if iy % 2 else 0.0
        for x in xs:
            for iz, z in enumerate(np.arange(lo[2], hi[2], spacing * 0.82)):
                pt = np.array(
                    [x + xoff, y, z + (0.35 * spacing if iz % 2 else 0.0)]
                )
                d_gel, _ = gel_tree.query(pt)
                if inner <= float(d_gel) <= outer:
                    cands.append((float(d_gel), pt))
    cands.sort(key=lambda t: t[0])
    solute_tree = cKDTree(solute_xyz) if len(solute_xyz) else None
    for _, pt in cands:
        if len(kept) >= n_target:
            break
        if solute_tree is not None:
            d_sol = float(solute_tree.query(pt)[0])
            if d_sol < 2.15:
                continue
        if kept:
            d_w = float(np.linalg.norm(np.array(kept) - pt, axis=1).min())
            if d_w < spacing * 0.88:
                continue
        kept.append(pt)
        yield {
            "name": "OW1",
            "resname": "WHW",
            "chain": "X",
            "resseq": resseq,
            "xyz": pt.copy(),
            "element": "O",
            "bfac": 30.0,
        }
        resseq += 1


def build_shell(
    gel_path: Path,
    out_path: Path,
    report_path: Path,
    *,
    mode: str = "disordered",
    saturation: str = "normal",
    shell_outer: float = SHELL_OUTER,
    shell_inner: float = SHELL_INNER,
    max_units: int | None = None,
    max_waters: int | None = None,
    n_twist: int = 16,
    crystal_seed: bool = False,
    seed_radius: float = SEED_RADIUS,
    seed_zone_inner: float = SEED_ZONE_INNER,
    seed_zone_outer: float = SEED_ZONE_OUTER,
    seed_anchor_resseq: int | None = None,
):
    sat = SATURATION.get(saturation, SATURATION["normal"])
    if max_units is None:
        max_units = sat["max_units"]
    if max_waters is None:
        max_waters = sat["max_waters"]

    atoms, _ = parse_atoms(gel_path)
    dna = [a for a in atoms if a["resname"] == "NUC"]
    gel_whw = [a for a in atoms if a["resname"] == "WHW"]
    gel_resseqs = sorted({a["resseq"] for a in gel_whw})
    gel_max = max(gel_resseqs)
    n_gel_units = len(gel_resseqs)

    gel_heavy = xyz_of([a for a in gel_whw if a["element"].upper() != "H"])
    gel_tree = cKDTree(gel_heavy)
    dna_xyz = xyz_of(dna)
    dna_o = xyz_of([a for a in dna if is_oxygen(a)])
    origin = dna_xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(dna_xyz - origin)
    axis = vt[0] / max(float(np.linalg.norm(vt[0])), 1e-8)

    frag = load_caox_fragment(CAOX_PDB)
    placed_heavy = list(gel_heavy)
    placed_o = list(dna_o)
    for a in gel_whw:
        if is_oxygen(a):
            placed_o.append(a["xyz"])

    if mode == "lattice":
        whw_frag = load_rigid_caox(CAOX_PDB)
        seeds = gel_ca_seeds(gel_whw, dna, origin, axis)
        nmax = sat["lattice_nmax"]
        zone_budgets = sat.get("zone_budgets")
        min_ca_sep = float(sat.get("min_ca_sep", 3.85))
        dis_spacing = float(sat.get("disordered_spacing", 2.0))
        placed_units = []
        zone_counts = []
        for zi, zspec in enumerate(RADIAL_ZONES):
            budget = (
                zone_budgets[zi]
                if zone_budgets and zi < len(zone_budgets)
                else max_units // len(RADIAL_ZONES)
            )
            if zspec.get("disordered"):
                cands = disordered_shell_candidates(
                    seeds,
                    gel_tree,
                    zspec["inner"],
                    zspec["outer"],
                    spacing=dis_spacing,
                )
                for c in cands:
                    c["bfac"] = zspec["bfac"]
                    c["disordered"] = True
            else:
                cands = lattice_shell_candidates(
                    seeds, gel_tree, zspec["inner"], zspec["outer"], nmax
                )
                for c in cands:
                    c["bfac"] = zspec["bfac"]
                    c["disordered"] = False
                    if zspec.get("random_orient"):
                        c["random_orient"] = True
                        c["random_frame"] = zspec.get("random_frame", False)
            new_units, placed_heavy, placed_o = place_lattice_units(
                cands,
                frag,
                dna_xyz,
                dna_o,
                placed_heavy,
                placed_o,
                n_twist=16,
                clash_max=12.0 if zspec.get("disordered") else 9.0,
                max_placements=budget,
                min_ca_sep=3.35 if zspec.get("disordered") else min_ca_sep,
                whw_frag=whw_frag,
            )
            placed_units.extend(new_units)
            zone_counts.append((zspec["inner"], zspec["outer"], len(new_units)))
            if len(placed_units) >= max_units:
                break
        # Surplus fill: any remaining lattice sites out to 30 Å
        if len(placed_units) < max_units:
            surplus = lattice_shell_candidates(
                seeds, gel_tree, SHELL_INNER_GEL, SHELL_OUTER, nmax + 1
            )
            for c in surplus:
                c["bfac"] = BFAC_BULK
                c["disordered"] = False
                c["random_orient"] = True
                c["random_frame"] = True
            new_units, placed_heavy, placed_o = place_lattice_units(
                surplus,
                frag,
                dna_xyz,
                dna_o,
                placed_heavy,
                placed_o,
                n_twist=12,
                clash_max=9.0,
                max_placements=max_units - len(placed_units),
                min_ca_sep=min_ca_sep,
                whw_frag=whw_frag,
            )
            placed_units.extend(new_units)
            zone_counts.append((SHELL_INNER_GEL, SHELL_OUTER, len(new_units)))
    else:
        lo = gel_heavy.min(0) - shell_outer - 3.0
        hi = gel_heavy.max(0) + shell_outer + 3.0
        placed_units = []
        fails = 0
        max_fails = 8000

        while len(placed_units) < max_units and fails < max_fails:
            center = RNG.uniform(lo, hi)
            d_gel, _ = gel_tree.query(center)
            if not (shell_inner <= float(d_gel) <= shell_outer):
                fails += 1
                continue
            outward = RNG.normal(size=3)
            outward /= max(float(np.linalg.norm(outward)), 1e-8)
            twist = float(RNG.uniform(0, 2 * math.pi))
            best_atoms, best_sc = None, 1e99
            other = np.array(placed_heavy)
            other_o = np.array(placed_o)
            for k in range(n_twist):
                trial = clone_fragment(
                    frag, center, outward, twist + k * (2 * math.pi / n_twist)
                )
                ca = next(a["xyz"] for a in trial if is_ca(a))
                if float(gel_tree.query(ca)[0]) < shell_inner - 0.05:
                    continue
                sc = unit_clash(trial, dna_xyz, dna_o, other, other_o)
                if sc < best_sc:
                    best_sc, best_atoms = sc, trial
            if best_atoms is None or best_sc > 5.0:
                fails += 1
                continue
            ca_pts = [a["xyz"] for a in best_atoms if is_ca(a)]
            if ca_pts:
                d_ca = np.linalg.norm(np.array(placed_heavy)[None, :, :] - ca_pts[0], axis=1)
                if float(d_ca.min()) < MIN_CA_CA - 0.05:
                    fails += 1
                    continue
            placed_units.append(best_atoms)
            for a in best_atoms:
                if a["element"].upper() != "H":
                    placed_heavy.append(a["xyz"].copy())
                if is_oxygen(a):
                    placed_o.append(a["xyz"].copy())
            fails = 0

    patch_atoms: list[dict] = []
    seed_site = None
    if crystal_seed:
        if mode != "lattice":
            raise ValueError("--crystal-seed requires --mode lattice")
        seeds = gel_ca_seeds(gel_whw, dna, origin, axis)
        seed_site = pick_crystal_seed_site(
            seeds,
            gel_tree,
            seed_zone_inner,
            seed_zone_outer,
            sat["lattice_nmax"] + 1,
            anchor_resseq=seed_anchor_resseq,
        )
        if seed_site is None:
            raise SystemExit(
                f"No bulk seed site in {seed_zone_inner:.1f}–{seed_zone_outer:.1f} Å "
                "from gel heavy atoms"
            )
        patch_atoms = cut_crystallite_patch(
            seed_site["xyz"],
            seed_site["tangent"],
            seed_site["outward"],
            radius=seed_radius,
            nmax=sat["lattice_nmax"] + 1,
        )
        placed_units = filter_units_near_patch(placed_units, patch_atoms)
        for a in patch_atoms:
            if a["element"].upper() != "H":
                placed_heavy.append(np.asarray(a["xyz"], float).copy())
            if is_oxygen(a):
                placed_o.append(np.asarray(a["xyz"], float).copy())

    solute_xyz = np.array(placed_heavy)
    if mode == "lattice":
        shell_atoms, next_res = format_oriented_units(placed_units, gel_max + 1)
    else:
        shell_atoms, next_res = format_shell_atoms(frag, placed_units, gel_max + 1)
    if patch_atoms:
        seed_atoms, next_res = format_crystal_seed(patch_atoms, next_res)
        shell_atoms.extend(seed_atoms)
    shell_waters = list(
        pack_shell_waters(
            gel_tree,
            solute_xyz,
            max_waters,
            shell_inner,
            shell_outer,
            next_res,
        )
    )

    # Tag gel with bfac 0 (frozen); shell with higher bfac
    for a in gel_whw:
        a["bfac"] = 0.0
    all_atoms = dna + gel_whw + shell_atoms + shell_waters
    ster = short_contact_summary(
        [{"element": a["element"], "name": a["name"], "resname": a["resname"],
          "chain": a["chain"], "resseq": a["resseq"], "xyz": a["xyz"]}
         for a in all_atoms if a["resname"] == "WHW"]
        + [{"element": a["element"], "name": a["name"], "resname": "NUC",
            "chain": a["chain"], "resseq": a["resseq"], "xyz": a["xyz"]}
           for a in dna]
    )

    mode_label = "radial zoned lattice (gel coat → intermediate → pre-crystal)"
    if crystal_seed:
        mode_label += " + outward bulk whewellite seed"
    fire_hint = (
        f"  python scripts/fire_openmm_caox.py {out_path.name} "
        f"--freeze-resseq-le {gel_max} --com-min-resseq {gel_max} "
        f"--com-ramp-steps 60 --rotation-free-steps 40"
        if mode == "lattice"
        else f"  python scripts/fire_openmm_caox.py {out_path.name} "
        f"--no-com-targets --freeze-resseq-le {gel_max}"
    )
    lines = [
        f"Gel + {mode_label} CaOx/water shell ({saturation} saturation)",
        "=" * 60,
        f"Gel source : {gel_path.name}  ({n_gel_units} WHW units, resseq 1–{gel_max})",
        f"Shell      : {shell_inner:.2f}–{shell_outer:.2f} Å from gel heavy atoms",
        f"Shell mode : {mode_label}",
        f"Shell units: {len(placed_units)} rigid CaC2O4·nH2O",
        f"Shell water: {len(shell_waters)} WHW O sites",
    ]
    if mode == "lattice" and zone_counts:
        lines.append("Radial zones (d_gel from gel heavy, bfac tag):")
        for zspec, (z0, z1, n) in zip(RADIAL_ZONES, zone_counts[: len(RADIAL_ZONES)]):
            lines.append(f"  {z0:.1f}–{z1:.1f} Å  bfac={zspec['bfac']:.0f}  units={n}")
        if len(zone_counts) > len(RADIAL_ZONES):
            z0, z1, n = zone_counts[-1]
            lines.append(
                f"  {z0:.1f}–{z1:.1f} Å  bfac={BFAC_BULK:.0f}  surplus fill units={n}"
            )
    n_seed_ca = sum(1 for a in patch_atoms if is_ca(a)) if patch_atoms else 0
    if crystal_seed and seed_site is not None:
        ia, ib, ic = seed_site["abc"]
        lines += [
            f"Crystal seed: {n_seed_ca} Ca whewellite patch at d_gel={seed_site['d_gel']:.1f} Å",
            f"  center COM offset ({ia:+d},{ib:+d},{ic:+d})  R={seed_radius:.1f} Å",
            f"  zone {seed_zone_inner:.1f}–{seed_zone_outer:.1f} Å; shell monomers within "
            f"{SEED_EXCLUDE_CA:.1f} Å of seed Ca removed",
        ]
    lines += [
        f"Total WHW  : {n_gel_units + len(placed_units) + n_seed_ca + len(shell_waters)} residues",
        "",
        "Sterics (inter-residue WHW + DNA)",
        f"  Ca–Ca < {MIN_CA_CA} Å: {ster['n_ca_short']}"
        + (f"  (min {ster['ca_min']:.3f})" if ster["ca_min"] is not None else ""),
        f"  O–O < {MIN_O_O} Å: {ster['n_oo_short']}"
        + (f"  (min {ster['oo_min']:.3f})" if ster["oo_min"] is not None else ""),
        "",
        "Relax with:",
        fire_hint,
    ]
    report = "\n".join(lines) + "\n"
    report_path.write_text(report)
    print(report)

    remarks = [
        f"HEADER    GEL + {mode.upper()} CAOX/WATER SHELL\n",
        f"TITLE     GEL RESSEQ 1-{gel_max} FROZEN; SHELL {shell_inner:.1f}-{shell_outer:.1f} A\n",
        f"REMARK   1 Gel units: {n_gel_units}; shell CaOx: {len(placed_units)}; "
        f"seed Ca: {n_seed_ca}; shell waters: {len(shell_waters)}\n",
        f"REMARK   2 Mode={mode} saturation={saturation}"
        + (" crystal_seed=1" if crystal_seed else "")
        + "\n",
    ]
    write_pdb(out_path, all_atoms, remarks)
    print(f"Wrote {out_path}")
    return gel_max, len(placed_units), len(shell_waters)


def main():
    ap = argparse.ArgumentParser(description="CaOx/water shell on gel-first model")
    ap.add_argument("--gel", type=Path, default=DEFAULT_GEL)
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--mode", choices=("disordered", "lattice"), default="lattice")
    ap.add_argument(
        "--saturation",
        choices=tuple(SATURATION),
        default="high",
        help="Shell packing density (high = more lattice images / units).",
    )
    ap.add_argument("--outer", type=float, default=SHELL_OUTER, help="Å from gel heavy atoms")
    ap.add_argument("--inner", type=float, default=SHELL_INNER)
    ap.add_argument("--max-units", type=int, default=None)
    ap.add_argument("--max-waters", type=int, default=None)
    ap.add_argument(
        "--crystal-seed",
        action="store_true",
        help="Embed authentic whewellite patch in bulk shell (22–28 Å).",
    )
    ap.add_argument("--seed-radius", type=float, default=SEED_RADIUS)
    ap.add_argument("--seed-zone-inner", type=float, default=SEED_ZONE_INNER)
    ap.add_argument("--seed-zone-outer", type=float, default=SEED_ZONE_OUTER)
    ap.add_argument(
        "--seed-anchor-resseq",
        type=int,
        default=None,
        help="Gel WHW resseq for lattice anchor (default: mid-gel).",
    )
    args = ap.parse_args()
    out = args.output
    report = args.report
    if args.crystal_seed and out == DEFAULT_OUT:
        out = DEFAULT_OUT_SEEDED
    if args.crystal_seed and report == DEFAULT_REPORT:
        report = DEFAULT_REPORT_SEEDED
    build_shell(
        args.gel,
        out,
        report,
        mode=args.mode,
        saturation=args.saturation,
        shell_outer=args.outer,
        shell_inner=args.inner,
        max_units=args.max_units,
        max_waters=args.max_waters,
        crystal_seed=args.crystal_seed,
        seed_radius=args.seed_radius,
        seed_zone_inner=args.seed_zone_inner,
        seed_zone_outer=args.seed_zone_outer,
        seed_anchor_resseq=args.seed_anchor_resseq,
    )


if __name__ == "__main__":
    main()
