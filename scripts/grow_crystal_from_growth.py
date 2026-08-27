#!/usr/bin/env python3
"""
Grow an authentic whewellite crystallite onto DNA_CaOx_growth.pdb.

Uses the 4-site COM seed on chain X, aligns Whewellite - xtl.pdb at each
seed Ca, merges overlapping lattice images, and keeps free ions + waters
outside the crystallite shell (clashing solvent is removed).

Default cut is a cylinder along the DNA: 30 Å perpendicular to the
helix axis (from the DNA envelope), length = DNA molecular length.
(Not a 30 Å sphere around each seed — that balloons past the termini.)
  0–6 Å   hydrated gel (backbone-attached)
  ~10 Å   local symmetry (~1 COM c-step, c ≈ 10.1 Å)
  ~20 Å   bulk-like (~2 c-steps)
  ~30 Å   outer radial shell
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_whewellite_patch import (  # noqa: E402
    COM_TARGETS,
)
from geom_constraints import DNA_HEAVY, is_ca, is_oxygen, pair_distances, xyz_of  # noqa: E402
from caox_units import (  # noqa: E402
    assign_unit_residues,
    drop_clashing_oxalate_units,
    drop_clashing_water_units,
    drop_close_ca_units,
    expand_units,
    merge_units,
    oxalate_quality,
    transform_unit,
    units_from_pbc,
)
from grow_whewellite import (  # noqa: E402
    XTL,
    expand_crystal,
    format_atom,
    local_frame,
    parse_atoms,
    rotation_around,
    rotation_from_to,
    unique_ca_distances,
)

ROOT = Path(__file__).resolve().parents[1]
GROWTH = ROOT / "DNA_CaOx_growth.pdb"
COM_C_STEP = 10.116

MERGE_R = 0.85
MIN_CA_CA = 3.50
MIN_O_O = 2.0
ION_CLASH = DNA_HEAVY

# Phosphate-distance shells (Å) for templating analysis
P_SHELLS = (
    (6.0, "0-6 A gel (backbone-attached)"),
    (10.0, "6-10 A inner shell"),
    (20.0, "10-20 A bulk-like (~2 c-steps)"),
    (30.0, "20-30 A outer slab"),
    (1e9, "30+ A cross-seed zone"),
)


def seed_strand(atoms):
    cas = [
        a
        for a in atoms
        if a["resname"] == "COM"
        and a["name"].strip().upper() == "CA"
        and a["chain"] == "X"
    ]
    cas.sort(key=lambda a: a["resseq"])
    if not cas:
        raise SystemExit("No COM seed Ca atoms on chain X in growth model")
    return cas


def phosphates(dna):
    return [a for a in dna if a["element"].upper() == "P"]


def seed_ca_from_phosphates(dna, origin, axis, stride: int = 1):
    """
    COM Ca at phosphates, grouped by chain in residue order.

    stride=1: every P. stride=2: every other P (same index on both strands).
    Ca sits ~2.8 Å outward from P along the helix radial.
    """
    stride = max(1, int(stride))
    by_chain: dict[str, list] = defaultdict(list)
    for p in phosphates(dna):
        by_chain[p["chain"]].append(p)
    strands = []
    for ch in sorted(by_chain):
        ps = sorted(by_chain[ch], key=lambda a: a["resseq"])[::stride]
        strand = []
        for p in ps:
            rel = p["xyz"] - origin
            t = float(rel @ axis)
            radial = rel - t * axis
            n = np.linalg.norm(radial)
            rhat = radial / n if n > 0.3 else np.array([1.0, 0.0, 0.0])
            xyz = p["xyz"] + 2.8 * rhat
            strand.append(
                {
                    "name": "CA",
                    "resname": "COM",
                    "chain": "X",
                    "resseq": p["resseq"],
                    "xyz": xyz,
                    "element": "CA",
                    "dna_chain": ch,
                }
            )
        strands.append(strand)
    return strands


def write_allp_seed_pdb(path: Path, dna, strands, remarks):
    """DNA + COM Ca at every phosphate (chain X, unique resseq) for analysis."""
    serial = 1
    lines = list(remarks)
    for a in dna:
        lines.append(
            format_atom(
                "ATOM",
                serial,
                a["name"],
                a["resname"],
                a["chain"],
                a["resseq"],
                a["xyz"],
                1.0,
                0.0,
                a.get("element", a["name"][:1]),
            )
        )
        serial += 1
    lines.append("TER\n")
    res = 1
    for strand in strands:
        for a in strand:
            lines.append(
                format_atom(
                    "HETATM",
                    serial,
                    "CA",
                    "COM",
                    "X",
                    res,
                    a["xyz"],
                    1.0,
                    20.0,
                    "CA",
                )
            )
            a["resseq"] = res
            serial += 1
            res += 1
    lines.append("END\n")
    path.write_text("".join(lines))


def alignment_matrix(av, cv, tangent, outward):
    R1 = rotation_from_to(av, tangent)
    best_ang, best_sc = 0.0, -1e9
    for ang in np.linspace(0, 2 * math.pi, 72, endpoint=False):
        R = rotation_around(tangent, ang) @ R1
        sc = float(np.dot(cv @ R.T, outward))
        if sc > best_sc:
            best_sc, best_ang = sc, ang
    return rotation_around(tangent, best_ang) @ R1


def crystal_nmax(extent: float, cryst: dict) -> int:
    """Enough periodic images to cover the DNA cylinder."""
    amin = min(cryst["a"], cryst["b"], cryst["c"])
    return max(2, int(math.ceil(extent / amin)) + 1)


def dna_slab_frame(dna_atoms: list[dict], pad: float = 0.5):
    """
    Helix axis from DNA heavy atoms (SVD). Axial span = DNA length.
    Radial envelope = max DNA distance from that axis.
    """
    heavy = [a for a in dna_atoms if a["element"].upper() != "H"]
    xyz = xyz_of(heavy)
    origin = xyz.mean(axis=0)
    _, _, vh = np.linalg.svd(xyz - origin, full_matrices=False)
    axis = vh[0]
    t = (xyz - origin) @ axis
    # Flip so t increases along the first-to-last phosphate if possible
    pxyz = xyz_of(phosphates(dna_atoms))
    if len(pxyz) >= 2:
        if float(np.dot(pxyz[-1] - pxyz[0], axis)) < 0:
            axis = -axis
            t = -t
    zmin, zmax = float(t.min()) - pad, float(t.max()) + pad
    radial = np.linalg.norm((xyz - origin) - t[:, None] * axis, axis=1)
    r_dna = float(radial.max())
    return origin, axis, zmin, zmax, r_dna


def in_dna_slab(xyz, origin, axis, zmin, zmax, r_max) -> bool:
    rel = xyz - origin
    z = float(np.dot(rel, axis))
    if z < zmin or z > zmax:
        return False
    radial = float(np.linalg.norm(rel - z * axis))
    return radial <= r_max + 0.05


def merge_oriented_atoms(items, cutoff):
    """Merge lattice copies within cutoff; count how many seeds claimed each site."""
    if not items:
        return []
    from scipy.spatial import cKDTree

    order = sorted(range(len(items)), key=lambda i: items[i].get("source", 0))
    xyz = np.array([items[i]["xyz"] for i in order], float)
    n = len(order)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = cKDTree(xyz).query_pairs(cutoff)
    for a, b in pairs:
        ia, ib = find(int(a)), find(int(b))
        if ia != ib:
            parent[ib] = ia  # keep lower index (earlier source)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    kept = []
    for members in groups.values():
        members.sort()
        rec = dict(items[order[members[0]]])
        rec["n_sources"] = len(members)
        kept.append(rec)
    return kept


def collect_oriented_units(
    strands, lattice_units, ref_xyz, helix_o, av, cv, keep_fn
):
    """Orient whole C2O4 / Ca / water units at each seed; merge by centroid."""
    if isinstance(strands[0], dict):
        strands = [strands]
    coms0 = np.array([u["com"] for u in lattice_units], float)
    tagged = []
    n_seed = 0
    n_all = sum(len(s) for s in strands)
    for strand in strands:
        for i, seed in enumerate(strand):
            tangent, outward = local_frame(strand, i, helix_o)
            R = alignment_matrix(av, cv, tangent, outward)
            com_t = (coms0 - ref_xyz) @ R.T + seed["xyz"]
            mask = keep_fn(com_t, seed["xyz"])
            src = n_seed
            n_seed += 1
            nkeep = 0
            for u, ok in zip(lattice_units, mask):
                if not ok:
                    continue
                tu = transform_unit(u, R, ref_xyz, seed["xyz"])
                tu["source"] = src
                tagged.append(tu)
                nkeep += 1
            print(
                f"  seed {n_seed}/{n_all}: {nkeep} units kept",
                flush=True,
            )
    return merge_units(tagged, MERGE_R)


def collect_oriented_fragments(
    strands, expanded, ref, helix_o, av, cv, keep_fn
):
    """Orient COM at each seed; keep atoms that pass keep_fn(xyz, seed_xyz)."""
    if isinstance(strands[0], dict):
        strands = [strands]
    meta = [a for a in expanded if a["element"].upper() != "H"]
    if not meta:
        return []
    xyz0 = np.array([a["xyz"] for a in meta], float)
    ref_xyz = np.asarray(ref["xyz"], float)
    tagged = []
    n_seed = 0
    for strand in strands:
        for i, seed in enumerate(strand):
            tangent, outward = local_frame(strand, i, helix_o)
            R = alignment_matrix(av, cv, tangent, outward)
            xyz = (xyz0 - ref_xyz) @ R.T + seed["xyz"]
            mask = keep_fn(xyz, seed["xyz"])
            src = n_seed
            n_seed += 1
            for a, p in zip((meta[j] for j in np.flatnonzero(mask)), xyz[mask]):
                b = dict(a)
                b["xyz"] = p
                b["source"] = src
                tagged.append(b)
            print(
                f"  seed {n_seed}/{sum(len(s) for s in strands)}: "
                f"{int(mask.sum())} atoms kept",
                flush=True,
            )
    return merge_oriented_atoms(tagged, MERGE_R)


def merge_atoms(items, cutoff):
    if not items:
        return []
    inv = 1.0 / cutoff
    buckets = defaultdict(list)
    kept = []
    for it in items:
        p = it["xyz"]
        ijk = (
            int(math.floor(p[0] * inv)),
            int(math.floor(p[1] * inv)),
            int(math.floor(p[2] * inv)),
        )
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
        if hit is None:
            rec = dict(it)
            rec["n_sources"] = 1
            buckets[ijk].append(len(kept))
            kept.append(rec)
        else:
            kept[hit]["n_sources"] = kept[hit].get("n_sources", 1) + 1
    return kept


def complete_hydrogens(kept, crystal):
    have = {id(a) for a in kept}
    oxyz = xyz_of([a for a in kept if is_oxygen(a)])
    if len(oxyz) == 0:
        return kept
    extra = []
    for a in crystal:
        if a["element"].upper() != "H":
            continue
        if id(a) in have:
            continue
        if float(np.linalg.norm(oxyz - a["xyz"], axis=1).min()) < 1.25:
            extra.append(a)
            have.add(id(a))
    return kept + extra


def drop_clash_ca(atoms, min_ca=MIN_CA_CA):
    from scipy.spatial import cKDTree

    cas = [a for a in atoms if is_ca(a)]
    others = [a for a in atoms if not is_ca(a)]
    keep_ca, keep_xyz = [], []
    for a in cas:
        if keep_xyz:
            if float(np.linalg.norm(np.asarray(keep_xyz) - a["xyz"], axis=1).min()) < min_ca:
                continue
        keep_ca.append(a)
        keep_xyz.append(a["xyz"])
    if not keep_ca:
        return []
    tree = cKDTree(np.asarray(keep_xyz, float))
    if not others:
        return keep_ca
    dmin, _ = tree.query(xyz_of(others), k=1)
    return keep_ca + [a for a, d in zip(others, dmin) if d < 5.5]


def drop_oo_clashes(atoms, min_oo=MIN_O_O):
    from scipy.spatial import cKDTree

    cas = {a["resseq"]: a for a in atoms if is_ca(a)}
    oxs = [a for a in atoms if is_oxygen(a)]
    if len(oxs) < 2:
        return atoms
    oxyz = xyz_of(oxs)
    pairs = cKDTree(oxyz).query_pairs(min_oo)
    drop = set()
    for i, j in pairs:
        a, b = oxs[i], oxs[j]
        if a.get("resseq") == b.get("resseq"):
            continue
        ca_a = cas.get(a.get("resseq"))
        ca_b = cas.get(b.get("resseq"))
        if ca_a is None or ca_b is None:
            continue
        da = float(np.linalg.norm(a["xyz"] - ca_a["xyz"]))
        db = float(np.linalg.norm(b["xyz"] - ca_b["xyz"]))
        drop.add(id(a if da >= db else b))
    return [a for a in atoms if id(a) not in drop]


def assign_whw_residues(atoms, shell_bfacs: dict[int, float] | None = None):
    from scipy.spatial import cKDTree

    cas = [a for a in atoms if is_ca(a)]
    cas.sort(key=lambda a: float(np.linalg.norm(a["xyz"])))
    cxyz = xyz_of(cas)
    tree = cKDTree(cxyz)
    xyz = xyz_of(atoms)
    _, nn = tree.query(xyz, k=1)
    ca_res = {id(c): i for i, c in enumerate(cas, start=1)}
    out = []
    for a, j in zip(atoms, nn):
        b = dict(a)
        b["resseq"] = ca_res[id(a)] if is_ca(a) else int(j) + 1
        b["resname"] = "WHW"
        b["chain"] = "Z"
        if shell_bfacs:
            b["bfac"] = shell_bfacs.get(b["resseq"], 20.0)
        out.append(b)
    return out


def seed_frames(strands, helix_o):
    if strands and isinstance(strands[0], dict):
        strands = [strands]
    frames = []
    idx = 0
    for strand in strands:
        for i, seed in enumerate(strand):
            tangent, outward = local_frame(strand, i, helix_o)
            frames.append(
                {"seed": seed, "tangent": tangent, "outward": outward, "index": idx}
            )
            idx += 1
    return frames


def ca_metrics(ca_xyz, pxyz, frames):
    """Distance to backbone (P) and outward c-step from nearest seed."""
    d_p = float(np.linalg.norm(pxyz - ca_xyz, axis=1).min()) if len(pxyz) else 0.0
    best = None
    for fr in frames:
        seed = fr["seed"]["xyz"]
        outward = fr["outward"]
        d_c = float(np.dot(ca_xyz - seed, outward))
        d_seed = float(np.linalg.norm(ca_xyz - seed))
        sc = d_seed
        if best is None or sc < best[0]:
            best = (sc, d_c, fr["index"])
    _d_seed, d_c, src = best or (0.0, 0.0, 0)
    c_step = max(0, int(round(d_c / COM_C_STEP))) if d_c > 0 else 0
    return d_p, d_c, c_step, src


def shell_label(d_p: float) -> str:
    prev = 0.0
    for edge, label in P_SHELLS:
        if d_p <= edge:
            return label
        prev = edge
    return P_SHELLS[-1][1]


def shell_bfac(d_p: float) -> float:
    if d_p <= 6.0:
        return 6.0
    if d_p <= 10.0:
        return 10.0
    if d_p <= 20.0:
        return 20.0
    if d_p <= 30.0:
        return 30.0
    return 40.0


def zone_report(cas_with_meta, dna, strand, helix_o):
    pxyz = xyz_of(phosphates(dna))
    frames = seed_frames(strand, helix_o)
    shell_counts = defaultdict(int)
    cstep_counts = defaultdict(int)
    shared_by_shell = defaultdict(int)
    lines = [
        "",
        "Distance from phosphate backbone (Ca → P min distance):",
    ]
    records = []

    for ca, ns in cas_with_meta:
        d_p, d_c, c_step, src = ca_metrics(ca["xyz"], pxyz, frames)
        label = shell_label(d_p)
        shell_counts[label] += 1
        cstep_counts[c_step] += 1
        if ns >= 2:
            shared_by_shell[label] += 1
        records.append((d_p, d_c, c_step, src, label, ns))

    for _edge, label in P_SHELLS:
        n = shell_counts[label]
        if n == 0 and label.startswith("30+"):
            continue
        shared = shared_by_shell[label]
        lines.append(f"  {label:28s}  Ca {n:4d}  (multi-seed sites {shared})")

    lines.append("")
    lines.append("Outward c-steps from nearest seed (c ≈ 10.1 Å):")
    for step in sorted(cstep_counts):
        lines.append(f"  c-step {step:2d}  : {cstep_counts[step]:4d} Ca")

    lines.append("")
    lines.append("Interpretation (templating zones):")
    lines.append("  0-6 Å    disordered hydrated gel — backbone-attached")
    lines.append("  ~10 Å    local whewellite symmetry (~1 c-step)")
    lines.append("  ~20 Å    bulk-like slab (~2 c-steps); outer edge may peel")
    lines.append("  ~30 Å    cross-seed lattice continuity may appear")
    return lines, records


def group_by_residue(atoms, resname):
    groups = defaultdict(list)
    for a in atoms:
        if a["resname"] == resname:
            groups[a["resseq"]].append(a)
    return groups


def residue_clashes(res_atoms, heavy_xyz):
    if len(heavy_xyz) == 0:
        return False
    rxyz = xyz_of(res_atoms)
    return float(np.linalg.norm(heavy_xyz - rxyz[:, None, :], axis=2).min()) < ION_CLASH


def filter_free_ions(atoms, heavy_xyz):
    keep = []
    for resname in ("NA", "CA", "OXL"):
        for _resseq, group in sorted(group_by_residue(atoms, resname).items()):
            if not residue_clashes(group, heavy_xyz):
                keep.extend(group)
    return keep


def filter_waters(atoms, heavy_xyz):
    kept = []
    removed = 0
    for _resseq, group in sorted(group_by_residue(atoms, "HOH").items()):
        if residue_clashes(group, heavy_xyz):
            removed += 1
            continue
        kept.extend(group)
    return kept, removed


def write_growth_pdb(path, dna, crystallite, ions, waters, remarks):
    out = list(remarks)
    serial = 0

    def add(rec, name, resname, chain, resseq, xyz, element, bfac, occ=1.0):
        nonlocal serial
        serial += 1
        out.append(
            format_atom(rec, serial, name, resname, chain, resseq, xyz, occ, bfac, element)
        )

    prev = None
    for a in sorted(dna, key=lambda t: (t["chain"], t["resseq"], t["serial"])):
        if prev is not None and a["chain"] != prev:
            out.append("TER\n")
        add("ATOM", a["name"], "NUC", a["chain"], a["resseq"], a["xyz"], a["element"], 0.0)
        prev = a["chain"]
    if dna:
        out.append("TER\n")

    for a in sorted(crystallite, key=lambda t: (t["chain"], t["resseq"], t["name"])):
        el = "Ca" if is_ca(a) else a["element"]
        name = "CA" if is_ca(a) else a["name"]
        bfac = a.get("bfac", 20.0)
        add("HETATM", name, "WHW", a.get("chain", "Z"), a["resseq"], a["xyz"], el, bfac)

    if crystallite:
        out.append("TER\n")

    for a in ions:
        add(
            "HETATM",
            a["name"],
            a["resname"],
            a["chain"],
            a["resseq"],
            a["xyz"],
            a["element"],
            15.0 if a["resname"] in ("CA", "OXL") else 10.0,
        )
    if ions:
        out.append("TER\n")

    for a in waters:
        add("HETATM", a["name"], "HOH", "W", a["resseq"], a["xyz"], a["element"], 0.0)

    out.append("END\n")
    path.write_text("".join(out))


def parse_args():
    p = argparse.ArgumentParser(description="Grow whewellite crystallite on DNA_CaOx_growth.pdb")
    p.add_argument(
        "--radius",
        type=float,
        default=30.0,
        help="Cut radius (Å). Cylinder: coating from DNA envelope. Sphere: from each seed.",
    )
    p.add_argument(
        "--seeds",
        choices=("chain-x", "all-p", "every-other"),
        default="chain-x",
        help="chain-x = existing 4 COM on one strand; all-p = every phosphate; "
        "every-other = every other P on both strands",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Keep every Nth phosphate (overrides --seeds stride). 2 = every other.",
    )
    p.add_argument(
        "--cut",
        choices=("cylinder", "sphere"),
        default=None,
        help="cylinder = DNA-length wrap; sphere = union of R-balls around seeds. "
        "Default: cylinder for chain-x, sphere for all-p.",
    )
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Output filename tag (default from radius and seed mode)",
    )
    p.add_argument(
        "--relax",
        action="store_true",
        help="Run rigid-body O-O relaxation after build (slow for large cuts)",
    )
    p.add_argument(
        "--no-relax",
        action="store_true",
        help="Skip relaxation even for small cuts",
    )
    return p.parse_args()


def main():
    args = parse_args()
    radius = args.radius
    r_tag = int(round(radius))
    p_mode = args.seeds in ("all-p", "every-other")
    stride = args.stride if args.stride is not None else (2 if args.seeds == "every-other" else 1)
    cut = args.cut or ("sphere" if p_mode else "cylinder")
    if args.tag:
        tag = args.tag
    elif args.seeds == "every-other" or stride > 1:
        tag = f"{r_tag}A_altP"
    elif args.seeds == "all-p":
        tag = f"{r_tag}A_allP"
    else:
        tag = f"{r_tag}A"
    out_pdb = ROOT / f"DNA_CaOx_growth_whewellite{tag}.pdb"
    report = ROOT / f"DNA_CaOx_growth_whewellite{tag}_report.txt"
    zone_file = ROOT / f"DNA_CaOx_growth_whewellite{tag}_zones.txt"

    growth_atoms, _ = parse_atoms(GROWTH)
    xtl_atoms, cryst = parse_atoms(XTL)
    if cryst is None:
        raise SystemExit("No CRYST1 in whewellite xtl.pdb")

    dna = [a for a in growth_atoms if a["resname"] == "NUC"]
    origin, axis, zmin, zmax, r_dna = dna_slab_frame(dna)
    helix_o = origin
    r_max = r_dna + radius
    dna_len = zmax - zmin

    if p_mode:
        strands = seed_ca_from_phosphates(dna, origin, axis, stride=stride)
        seed_pdb = ROOT / (
            "DNA_CaOx_growth_altP_seeds.pdb" if stride > 1 else "DNA_CaOx_growth_allP_seeds.pdb"
        )
        title = (
            f"EVERY OTHER P (STRIDE {stride}) ON BOTH STRANDS"
            if stride > 1
            else "ONE COM CA PER P ON BOTH STRANDS"
        )
        write_allp_seed_pdb(
            seed_pdb,
            dna,
            strands,
            [
                "HEADER    PHOSPHATE COM SEEDS\n",
                f"TITLE     {title}\n",
                "REMARK   1 Ca is 2.8 A outward from P along the helix radial.\n",
                f"REMARK   1 Phosphate stride={stride}.\n",
            ],
        )
        print(f"Wrote {seed_pdb} ({sum(len(s) for s in strands)} seeds)", flush=True)
    else:
        strands = [seed_strand(growth_atoms)]

    n_seeds = sum(len(s) for s in strands)
    seed_xyz = np.array([a["xyz"] for s in strands for a in s])

    if cut == "sphere":
        gen_r = radius + 2.0

        def keep_fn(xyz, seed):
            return np.linalg.norm(xyz - seed, axis=1) <= radius + 0.05

        cut_desc = (
            f"union of {radius:.1f} Å spheres around {n_seeds} seed Ca"
        )
    else:
        t_seed = (seed_xyz - origin) @ axis
        gen_r = 0.0
        for ts, s in zip(t_seed, seed_xyz):
            axial = max(abs(ts - zmin), abs(ts - zmax))
            seed_rad = float(np.linalg.norm((s - origin) - ts * axis))
            gen_r = max(gen_r, math.hypot(axial, r_max + seed_rad))
        gen_r += 2.0

        def keep_fn(xyz, seed):
            rel = xyz - origin
            z = rel @ axis
            radial = np.linalg.norm(rel - z[:, None] * axis, axis=1)
            dseed = np.linalg.norm(xyz - seed, axis=1)
            return (
                (z >= zmin)
                & (z <= zmax)
                & (radial <= r_max + 0.05)
                & (dseed <= gen_r + 0.05)
            )

        cut_desc = (
            f"cylinder along DNA, {radius:.1f} Å perpendicular "
            f"from DNA envelope (axis radius {r_max:.1f} Å)"
        )

    nmax = crystal_nmax(gen_r, cryst)
    print(
        f"Seeds {n_seeds} ({args.seeds}); cut {cut}; "
        f"R={radius:.0f} Å; gen_r={gen_r:.1f} nmax={nmax}",
        flush=True,
    )
    expanded1, (av, bv, cv) = expand_crystal(xtl_atoms, cryst, nmax=1)
    base_units = units_from_pbc(xtl_atoms, expanded1, av, bv, cv)
    n_ox = sum(u["kind"] == "oxalate" for u in base_units)
    n_ca_u = sum(u["kind"] == "ca" for u in base_units)
    n_w_u = sum(u["kind"] == "water" for u in base_units)
    print(
        f"Whewellite cell units: {n_ox} C2O4, {n_ca_u} Ca, {n_w_u} water/O",
        flush=True,
    )
    lattice_units = expand_units(base_units, av, bv, cv, nmax=nmax)
    ref = next(a for a in xtl_atoms if a["element"].upper() == "CA")
    xtl_ca = xyz_of([a for a in xtl_atoms if a["element"].upper() == "CA"])
    uniq_d = unique_ca_distances(xtl_ca, av, bv, cv, cutoff=min(radius + 2, 35.0))

    units = collect_oriented_units(
        strands, lattice_units, ref["xyz"], helix_o, av, cv, keep_fn
    )
    print(f"Unique lattice units after seed merge: {len(units)}", flush=True)
    if cut == "cylinder":
        units = [
            u
            for u in units
            if in_dna_slab(u["com"], origin, axis, zmin, zmax, r_max)
        ]
    units = drop_close_ca_units(units, MIN_CA_CA)
    print(
        f"After Ca–Ca ≥ {MIN_CA_CA:.2f} Å: "
        f"{sum(u['kind']=='ca' for u in units)} Ca, "
        f"{sum(u['kind']=='oxalate' for u in units)} C2O4",
        flush=True,
    )
    units = drop_clashing_oxalate_units(units, MIN_O_O)
    units = drop_clashing_water_units(units, MIN_O_O)
    print(
        f"After whole-unit O···O ≥ {MIN_O_O:.2f} Å: "
        f"{sum(u['kind']=='oxalate' for u in units)} C2O4, "
        f"{sum(u['kind']=='water' for u in units)} water",
        flush=True,
    )

    cas_pre = [
        (u["atoms"][0], u.get("n_sources", 1))
        for u in units
        if u["kind"] == "ca"
    ]
    zone_lines, zone_records = zone_report(cas_pre, dna, strands, helix_o)
    shell_bfacs = {}
    pxyz = xyz_of(phosphates(dna))
    frames = seed_frames(strands, helix_o)
    cas_sorted = sorted(
        [c for c, _ in cas_pre], key=lambda a: float(np.linalg.norm(a["xyz"]))
    )
    for i, ca in enumerate(cas_sorted, start=1):
        d_p, _, _, _ = ca_metrics(ca["xyz"], pxyz, frames)
        shell_bfacs[i] = shell_bfac(d_p)

    crystallite = assign_unit_residues(units, shell_bfacs)
    q = oxalate_quality(crystallite)

    n_com = sum(1 for a in growth_atoms if a["resname"] == "COM")
    n_na_before = sum(1 for a in growth_atoms if a["resname"] == "NA")
    n_ca_before = sum(1 for a in growth_atoms if a["resname"] == "CA")
    n_oxl_before = sum(1 for a in growth_atoms if a["resname"] == "OXL")
    n_hoh_before = sum(1 for a in growth_atoms if a["resname"] == "HOH" and a["name"] == "O")

    cryst_heavy = xyz_of([a for a in crystallite if a["element"].upper() not in ("H",)])
    ions = filter_free_ions(growth_atoms, cryst_heavy)
    waters, n_hoh_removed = filter_waters(growth_atoms, cryst_heavy)

    cas = [a for a in crystallite if is_ca(a)]
    cxyz = xyz_of(cas)
    pair = pair_distances(cxyz, min(radius + 2, 35.0))

    def nnear(target, tol=0.30):
        return int(np.sum(np.abs(pair - target) <= tol)) if pair.size else 0

    n_h = sum(1 for a in crystallite if a["element"].upper() == "H")
    n_shared = sum(1 for u in units if u.get("n_sources", 1) >= 2)
    n_multi_seed_ca = sum(1 for _d, _c, _s, _src, _lab, ns in zone_records if ns >= 2)
    n_ox_u = sum(u["kind"] == "oxalate" for u in units)
    n_w_out = sum(u["kind"] == "water" for u in units)

    if p_mode:
        seed_label = (
            f"{n_seeds} COM Ca, every other phosphate on both strands (stride={stride})"
            if stride > 1
            else f"{n_seeds} COM Ca, one per phosphate on both strands"
        )
    else:
        seed_label = (
            f"{n_seeds} COM Ca on chain X (residues "
            + ", ".join(str(a["resseq"]) for s in strands for a in s)
            + ")"
        )
    lines = [
        f"Whewellite crystallite on DNA_CaOx_growth ({tag} cut)",
        "=" * 60,
        f"Input          : {GROWTH.name}",
        f"Crystal source : {XTL.name}",
        f"Cell images    : nmax={nmax} ({len(lattice_units)} unit copies)",
        f"Seeds          : {seed_label}",
        f"Cut            : {cut_desc}",
        f"DNA length     : {dna_len:.1f} Å",
        f"DNA radial env.: {r_dna:.1f} Å from helix axis",
        "",
        "Units are whole C2O4, Ca, and water. Overlapping seeds keep the "
        "earliest complete molecule (no atom-wise averaging).",
        f"C2O4 intact     : {q['n_intact']}/{q['n_oxalate']}  "
        f"orphan C={q['orphan_c']}  "
        f"C–C {q['cc_median']:.3f} Å  C–O {q['co_median']:.3f} Å"
        if q["cc_median"]
        else f"C2O4 intact     : {q['n_intact']}/{q['n_oxalate']}",
        "B-factor on WHW encodes phosphate-distance shell: 6, 10, 20, 30, 40 Å.",
        "",
        f"Removed COM seed atoms     : {n_com}",
        f"Removed clashing waters    : {n_hoh_removed} / {n_hoh_before} molecules",
        f"Free ions kept             : Na {sum(1 for a in ions if a['resname']=='NA')}/{n_na_before}, "
        f"Ca {sum(1 for a in ions if a['resname']=='CA')}/{n_ca_before}, "
        f"oxalate {sum(1 for a in ions if a['resname']=='OXL')//6}/{n_oxl_before//6}",
        f"Waters kept                : {len(waters)//3} molecules",
        "",
        f"Crystallite atoms          : {len(crystallite)}  "
        f"(Ca {len(cas)}, C2O4 {n_ox_u}, water {n_w_out}, H {n_h})",
        f"Multi-seed lattice units   : {n_shared} merged units, {n_multi_seed_ca} Ca",
        f"Total atoms in output PDB  : {len(dna) + len(crystallite) + len(ions) + len(waters)}",
    ]
    lines.extend(zone_lines)
    lines.append("")
    lines.append("COM reference Ca–Ca (periodic):")
    for d, n in uniq_d:
        if d <= radius + 2:
            lines.append(f"   {d:6.3f} Å   (x{n})")
    lines.append("")
    lines.append(f"Ca–Ca in model (≤ {min(radius + 2, 35):.0f} Å):")
    for tgt in COM_TARGETS:
        if tgt <= radius + 2:
            lines.append(f"  {tgt:5.2f} Å ±0.30 : {nnear(tgt)} pairs")
    if pair.size:
        lines.append(
            f"  n pairs={len(pair)}  min={pair.min():.2f}  max={pair.max():.2f}"
        )
    lines.append("")
    lines.append(f"Output: {out_pdb.name}")
    lines.append(f"Zones : {zone_file.name}")

    report_text = "\n".join(lines) + "\n"
    report.write_text(report_text)
    zone_file.write_text(report_text)
    print(report_text)

    remarks = [
        "HEADER    GROWTH MODEL + WHEWELLITE CRYSTALLITE\n",
        f"TITLE     {cut.upper()} CUT {r_tag} A; SEEDS={args.seeds}\n",
        f"REMARK   1 Cut: {cut_desc}.\n",
        "REMARK   1 Authentic whewellite; whole C2O4/Ca/water units, no atom merge.\n",
        "REMARK   1 WHW B-factors tag P-distance shells: 6/10/20/30/40 A.\n",
        f"REMARK   2 Crystallite {len(crystallite)} atoms ({len(cas)} Ca); "
        f"{len(waters)//3} waters.\n",
        f"REMARK   3 COM cell a={cryst['a']:.3f} b={cryst['b']:.3f} c={cryst['c']:.3f} "
        f"beta={cryst['beta']:.2f}\n",
    ]
    write_growth_pdb(out_pdb, dna, crystallite, ions, waters, remarks)
    print(f"Wrote {out_pdb}")
    print(f"Wrote {report}")

    do_relax = args.relax or (radius <= 12.0 and not args.no_relax)
    relax_script = Path(__file__).resolve().parent / "relax_whewellite_units.py"
    if do_relax and relax_script.exists():
        import subprocess

        out_relaxed = ROOT / f"DNA_CaOx_growth_whewellite{tag}_relaxed.pdb"
        print(f"Running rigid-body O-O relaxation → {out_relaxed.name} ...")
        cmd = [sys.executable, str(relax_script), str(out_pdb), str(out_relaxed)]
        # All-P wraps have hundreds of WHW units; FIRE+OpenMM does the oxalate/Ca fit.
        if p_mode:
            cmd.append("--fast")
        subprocess.run(cmd, check=False)
    elif radius > 12.0 and not args.relax:
        print(
            f"Skipping auto-relax at {radius:.0f} Å (large model). "
            f"Run: python3 scripts/relax_whewellite_units.py {out_pdb.name}\n"
            f"Then: .venv/bin/python scripts/fire_openmm_caox.py {out_pdb.name}"
        )


if __name__ == "__main__":
    main()
