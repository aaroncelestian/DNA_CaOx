#!/usr/bin/env python3
"""
Build a DNA + calcium oxalate hydrate model.

Places one CaC2O4·nH2O unit at each phosphate so sequential Ca–Ca
distances track the B-DNA backbone P–P spacing (~6.3–7.0 Å), matching
the whewellite (COM) chain repeat of ~6.29 Å.

Ca2+ is kept near the phosphate helical radius. Putting Ca further
out (classic OP1/OP2 bisector) inflates Ca–Ca to ~8 Å on the helix.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geom_constraints import (  # noqa: E402
    DNA_HEAVY,
    MIN_CA_CA,
    MIN_O_O,
    is_ca,
    is_oxygen,
    separate_ca,
    short_contact_summary,
    xyz_of,
)

ROOT = Path(__file__).resolve().parents[1]
DNA_PDB = ROOT / "DOCS" / "DNA.pdb"
CAOX_PDB = ROOT / "DOCS" / "Whewellite ca_ox.pdb"
OUT_PDB = ROOT / "DNA_CaOx_backbone.pdb"
REPORT = ROOT / "DNA_CaOx_backbone_report.txt"

CA_O_PHOS = 2.40
R_SCALE = 0.97  # Ca cylinder vs phosphate cylinder; keeps Ca–Ca in 6–7 Å


def parse_pdb_atoms(path: Path):
    atoms = []
    conect = defaultdict(set)
    with path.open() as fh:
        for line in fh:
            if line.startswith(("ATOM  ", "HETATM")):
                serial = int(line[6:11])
                name = line[12:16].strip()
                resname = line[17:20].strip()
                chain = line[21].strip() or "A"
                try:
                    resseq = int(line[22:26])
                except ValueError:
                    resseq = 0
                xyz = np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                )
                element = line[76:78].strip() if len(line) >= 78 else ""
                if not element:
                    element = "".join(c for c in name if c.isalpha())[:2].title()
                    if name.upper().startswith("CA"):
                        element = "Ca"
                atoms.append(
                    {
                        "serial": serial,
                        "name": name,
                        "resname": resname,
                        "chain": chain,
                        "resseq": resseq,
                        "xyz": xyz,
                        "element": element,
                    }
                )
            elif line.startswith("CONECT"):
                cols = [line[i : i + 5] for i in range(6, len(line.rstrip()), 5)]
                nums = []
                for col in cols:
                    col = col.strip()
                    if not col:
                        continue
                    try:
                        nums.append(int(col))
                    except ValueError:
                        pass
                if nums:
                    a0 = nums[0]
                    for b in nums[1:]:
                        conect[a0].add(b)
                        conect[b].add(a0)
    return atoms, conect


def format_atom(rec, serial, name, resname, chain, resseq, xyz, occ, bfac, element):
    if element.upper() == "CA":
        aname = "CA  "
        element = "Ca"
    elif name.startswith("OW"):
        aname = f"{name:<4}"[:4]
    elif len(name) >= 4:
        aname = name[:4]
    elif len(element) == 2:
        aname = f"{element:>2}{name[len(element):]:<2}"[:4]
    else:
        aname = f" {name:<3}"[:4]
    return (
        f"{rec:<6}{serial:5d} {aname:4s} {resname:>3s} {chain:1s}{resseq:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{occ:6.2f}{bfac:6.2f}"
        f"          {element:>2s}\n"
    )


def norm(v):
    n = np.linalg.norm(v)
    return v * 0.0 if n < 1e-8 else v / n


def rotation_matrix_from_to(a, b):
    a = norm(a)
    b = norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-10:
        if c > 0:
            return np.eye(3)
        axis = norm(np.cross(a, np.array([1.0, 0.0, 0.0])))
        if np.linalg.norm(axis) < 0.1:
            axis = norm(np.cross(a, np.array([0.0, 1.0, 0.0])))
        return rotation_around_axis(axis, math.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def rotation_around_axis(axis, angle):
    axis = norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def apply_rt(xyz, R, origin_from, origin_to):
    return (xyz - origin_from) @ R.T + origin_to


def by_serial(atoms):
    return {a["serial"]: a for a in atoms}


def phosphate_groups(atoms, conect):
    lookup = by_serial(atoms)
    groups = []
    for a in atoms:
        if a["element"].upper() != "P":
            continue
        neigh = [lookup[i] for i in conect.get(a["serial"], ()) if i in lookup]
        oxygens = [n for n in neigh if n["element"].upper().startswith("O")]
        nonbridge, bridge = [], []
        for o in oxygens:
            o_others = [
                lookup[i]
                for i in conect.get(o["serial"], ())
                if i in lookup and i != a["serial"]
            ]
            heavy = [x for x in o_others if x["element"].upper() != "H"]
            (nonbridge if not heavy else bridge).append(o)
        groups.append({"p": a, "op": nonbridge, "o_bridge": bridge, "all_o": oxygens})
    return groups


def sequential_p_pairs(atoms, conect, p_serials):
    lookup = by_serial(atoms)
    pset = set(p_serials)
    pairs = []
    for ps in p_serials:
        seen = {ps}
        frontier = [(ps, 0)]
        found = []
        while frontier:
            node, dist = frontier.pop(0)
            if dist >= 8:
                continue
            for nxt in conect.get(node, ()):
                if nxt in seen:
                    continue
                seen.add(nxt)
                if nxt in pset and nxt != ps and 4 <= dist + 1 <= 8:
                    found.append(nxt)
                elif nxt not in pset:
                    frontier.append((nxt, dist + 1))
        for q in found:
            if ps < q:
                d = np.linalg.norm(lookup[ps]["xyz"] - lookup[q]["xyz"])
                pairs.append((ps, q, d))
    return pairs


def order_strands(p_serials, pairs):
    adj = defaultdict(list)
    for a, b, d in pairs:
        if 5.0 <= d <= 8.5:
            adj[a].append((b, d))
            adj[b].append((a, d))
    unused = set(p_serials)
    strands = []
    while unused:
        ends = [p for p in unused if sum(1 for n, _ in adj[p] if n in unused) <= 1]
        start = ends[0] if ends else next(iter(unused))
        strand = [start]
        unused.remove(start)
        while True:
            nxts = [(n, d) for n, d in adj[strand[-1]] if n in unused]
            if not nxts:
                break
            nxts.sort(key=lambda t: t[1])
            strand.append(nxts[0][0])
            unused.remove(nxts[0][0])
        strands.append(strand)
    strands.sort(key=lambda s: (-len(s), s[0]))
    return strands


def pair_duplexes(strands, lookup):
    """Pair 11-P strands whose phosphate centroids are nearest."""
    long_s = [s for s in strands if len(s) >= 6]
    cents = {
        id(s): np.mean([lookup[i]["xyz"] for i in s], axis=0) for s in long_s
    }
    unused = long_s[:]
    duplexes = []
    while unused:
        s0 = unused.pop(0)
        if not unused:
            duplexes.append([s0])
            break
        unused.sort(key=lambda s: np.linalg.norm(cents[id(s0)] - cents[id(s)]))
        s1 = unused.pop(0)
        duplexes.append([s0, s1])
    leftovers = [s for s in strands if s not in long_s]
    if leftovers:
        duplexes.append(leftovers)
    return duplexes


def fit_helix_axis(points):
    pts = np.asarray(points, float)
    origin = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - origin)
    return origin, norm(vt[0])


def radial_frame(xyz, origin, axis):
    w = xyz - origin
    z = float(np.dot(w, axis))
    rho = w - axis * z
    r = np.linalg.norm(rho)
    rhat = norm(rho)
    that = axis
    bhat = np.cross(that, rhat)
    return r, rhat, that, bhat, z


def local_offset_from_ca(ca_xyz, p_xyz, origin, axis):
    """Express Ca−P in the local helical frame (radial, axial, azimuthal)."""
    r_p, rhat, that, bhat, _ = radial_frame(p_xyz, origin, axis)
    d = ca_xyz - p_xyz
    return (float(np.dot(d, rhat)), float(np.dot(d, that)), float(np.dot(d, bhat)))


def ca_from_local_offset(p_xyz, origin, axis, offset):
    _, rhat, that, bhat, _ = radial_frame(p_xyz, origin, axis)
    o_r, o_t, o_b = offset
    return p_xyz + o_r * rhat + o_t * that + o_b * bhat


def seed_ca_at_phosphate(p_xyz, ops, origin, axis, r_scale=R_SCALE):
    """
    Seed Ca in the local helical frame of this phosphate.

    Stay near the phosphate cylinder (small radial offset) so sequential
    Ca–Ca will track P–P after the offset is copied along the strand.
    Reach a chemically usable Ca–O contact (~2.4 Å) using azimuthal and
    axial displacement — not by sitting on the phosphorus.
    """
    _, rhat, that, bhat, _ = radial_frame(p_xyz, origin, axis)
    if not ops:
        ca = p_xyz + 2.6 * bhat
        return ca, rhat

    op_xyz = np.asarray(ops, float)
    best = None
    best_sc = 1e99
    # o_r small: keep helical radius ≈ phosphate radius
    for o_r in np.linspace(-0.35, 0.35, 7):
        for o_t in np.linspace(-3.2, 3.2, 33):
            for o_b in np.linspace(-3.2, 3.2, 33):
                ca = p_xyz + o_r * rhat + o_t * that + o_b * bhat
                d_p = float(np.linalg.norm(ca - p_xyz))
                if d_p < 2.55 or d_p > 3.60:
                    continue
                d_op = np.linalg.norm(op_xyz - ca, axis=1)
                dmin = float(d_op.min())
                sc = (dmin - CA_O_PHOS) ** 2
                if len(d_op) > 1:
                    sc += 0.05 * (float(np.sort(d_op)[1]) - 3.4) ** 2
                # Prefer a slightly inward, mostly axial offset so Ca–Ca
                # tracks P–P after helical parallel transport.
                sc += 0.12 * (o_r + 0.20) ** 2
                sc += 0.04 * o_b**2
                if sc < best_sc:
                    best_sc = sc
                    best = ca
    if best is None:
        best = p_xyz + 2.6 * bhat
    return best, rhat


def load_caox_fragment(path: Path):
    atoms, conect = parse_pdb_atoms(path)
    ca = next(a for a in atoms if a["element"].upper() == "CA")
    carbons = [a for a in atoms if a["element"].upper() == "C"]
    ox_o, waters = [], []
    carbon_ids = {c["serial"] for c in carbons}
    for a in atoms:
        if a["serial"] == ca["serial"] or a["element"].upper() == "C":
            continue
        bonded = conect.get(a["serial"], ())
        if any(s in carbon_ids for s in bonded):
            ox_o.append(a)
        else:
            waters.append(a)
    ox_atoms = carbons + ox_o
    ox_cent = np.mean([a["xyz"] for a in ox_atoms], axis=0)
    return {
        "atoms": atoms,
        "conect": conect,
        "ca": ca,
        "ox_cent": ox_cent,
        "ca_to_ox": ox_cent - ca["xyz"],
        "carbons": carbons,
        "ox_o": ox_o,
        "waters": waters,
    }


def transform_fragment(frag, ca_target, outward, twist):
    R = rotation_around_axis(outward, twist) @ rotation_matrix_from_to(
        frag["ca_to_ox"], outward
    )
    out = []
    for a in frag["atoms"]:
        b = dict(a)
        b["xyz"] = apply_rt(a["xyz"], R, frag["ca"]["xyz"], ca_target)
        out.append(b)
    return out


def clash_score(placed_xyz, placed_o, dna_xyz, dna_o, other_xyz, other_o):
    """Rigid-unit clash: DNA heavy + inter-unit O–O (min 2 Å)."""
    score = 0.0
    d = np.linalg.norm(placed_xyz[:, None, :] - dna_xyz[None, :, :], axis=2)
    close = d < DNA_HEAVY
    if np.any(close):
        score += float(np.sum((DNA_HEAVY - d[close]) ** 2)) * 8.0
    if len(placed_o) and len(dna_o):
        do = np.linalg.norm(placed_o[:, None, :] - dna_o[None, :, :], axis=2)
        hit = do < MIN_O_O
        if np.any(hit):
            score += float(np.sum((MIN_O_O - do[hit]) ** 2)) * 25.0
    if len(other_xyz):
        d2 = np.linalg.norm(placed_xyz[:, None, :] - other_xyz[None, :, :], axis=2)
        close2 = d2 < DNA_HEAVY
        if np.any(close2):
            score += float(np.sum((DNA_HEAVY - d2[close2]) ** 2)) * 4.0
    if len(placed_o) and len(other_o):
        d3 = np.linalg.norm(placed_o[:, None, :] - other_o[None, :, :], axis=2)
        hit3 = d3 < MIN_O_O
        if np.any(hit3):
            score += float(np.sum((MIN_O_O - d3[hit3]) ** 2)) * 25.0
    return score


def assign_nucleotide_residues(dna_atoms, conect, strands, lookup, duplexes):
    """Walk CONECT from each P to tag DNA atoms with chain/resseq."""
    pset = {p for s in strands for p in s}
    tag = {}  # serial -> (chain, resseq)
    chain_of_strand = {}
    letters = "ABCD"
    n = 0
    for duplex in duplexes:
        for strand in duplex:
            if n < len(letters):
                chain_of_strand[id(strand)] = letters[n]
                n += 1
    for strand in strands:
        chain = chain_of_strand.get(id(strand), "A")
        for resseq, ps in enumerate(strand, start=2):  # 5' terminal is residue 1 (no P)
            tag[ps] = (chain, resseq)
            seen = {ps}
            q = [ps]
            while q:
                node = q.pop()
                for nxt in conect.get(node, ()):
                    if nxt in seen:
                        continue
                    if nxt in pset and nxt != ps:
                        continue
                    seen.add(nxt)
                    # do not flood through neighboring nucleotide P
                    tag[nxt] = (chain, resseq)
                    q.append(nxt)
    # leftover atoms: nearest tagged atom
    tagged_xyz = np.array([lookup[s]["xyz"] for s in tag])
    tagged_ids = list(tag)
    for a in dna_atoms:
        if a["serial"] in tag:
            continue
        d = np.linalg.norm(tagged_xyz - a["xyz"], axis=1)
        tag[a["serial"]] = tag[tagged_ids[int(np.argmin(d))]]
    return tag


def main():
    dna_atoms, dna_conect = parse_pdb_atoms(DNA_PDB)
    lookup = by_serial(dna_atoms)
    groups = phosphate_groups(dna_atoms, dna_conect)
    p_serials = [g["p"]["serial"] for g in groups]
    group_by_p = {g["p"]["serial"]: g for g in groups}

    pairs = sequential_p_pairs(dna_atoms, dna_conect, p_serials)
    strands = [s for s in order_strands(p_serials, pairs) if len(s) >= 3]
    duplexes = pair_duplexes(strands, lookup)

    dna_xyz = np.array([a["xyz"] for a in dna_atoms])
    frag = load_caox_fragment(CAOX_PDB)

    placements = []
    ca_records = []  # si, j, p_serial, ca_xyz
    placed_heavy = []
    placed_o = []
    dna_o = xyz_of([a for a in dna_atoms if is_oxygen(a)])

    # map strand -> duplex axis
    strand_axis = {}
    for duplex in duplexes:
        pts = [lookup[p]["xyz"] for s in duplex for p in s]
        origin, axis = fit_helix_axis(pts)
        for s in duplex:
            strand_axis[id(s)] = (origin, axis)

    # Pass 1: Ca sites (rigid-unit centers), then enforce Ca–Ca ≥ 6 Å.
    sites = []  # si, j, ps, ca_xyz, outward
    for si, strand in enumerate(strands):
        origin, axis = strand_axis[id(strand)]
        g0 = group_by_p[strand[0]]
        seed, _ = seed_ca_at_phosphate(
            g0["p"]["xyz"], [o["xyz"] for o in g0["op"]], origin, axis
        )
        offset = local_offset_from_ca(seed, g0["p"]["xyz"], origin, axis)
        for j, ps in enumerate(strand):
            g = group_by_p[ps]
            ca_xyz = ca_from_local_offset(g["p"]["xyz"], origin, axis, offset)
            ops = [o["xyz"] for o in g["op"]]
            if ops:
                op = min(ops, key=lambda x: np.linalg.norm(np.asarray(x) - ca_xyz))
                vec = ca_xyz - op
                d_op = float(np.linalg.norm(vec))
                if 1e-6 < d_op < 2.15:
                    ca_xyz = op + vec * (2.20 / d_op)
                elif d_op > 2.85:
                    pull = min(0.22, d_op - 2.50)
                    ca_xyz = ca_xyz - pull * vec / d_op
            _, rhat, _, _, _ = radial_frame(g["p"]["xyz"], origin, axis)
            outward = rhat if np.linalg.norm(rhat) > 0.2 else norm(ca_xyz - origin)
            sites.append([si, j, ps, ca_xyz, outward])

    sep = separate_ca([s[3] for s in sites], MIN_CA_CA)
    for k, site in enumerate(sites):
        site[3] = sep[k]

    # Pass 2: attach a rigid CaOx at each Ca; rotate to keep O–O ≥ 2 Å.
    for si, j, ps, ca_xyz, outward in sites:
        ligand_best = None
        best_sc = 1e99
        other = np.array(placed_heavy) if placed_heavy else np.zeros((0, 3))
        other_o = np.array(placed_o) if placed_o else np.zeros((0, 3))
        g = group_by_p[ps]
        origin, axis = strand_axis[id(next(s for s in strands if ps in s))]
        _, _, that, bhat, _ = radial_frame(g["p"]["xyz"], origin, axis)
        directions = [outward, -outward, that, -that, bhat, -bhat]
        for ox_dir in directions:
            if np.linalg.norm(ox_dir) < 0.2:
                continue
            for deg in range(0, 360, 8):
                atoms = transform_fragment(frag, ca_xyz, ox_dir, math.radians(deg))
                ligands = np.array(
                    [a["xyz"] for a in atoms if a["element"].upper() != "CA"]
                )
                lig_o = np.array([a["xyz"] for a in atoms if is_oxygen(a)])
                sc = clash_score(ligands, lig_o, dna_xyz, dna_o, other, other_o)
                if sc < best_sc:
                    best_sc = sc
                    ligand_best = atoms
                if sc < 1e-9:
                    break
            if best_sc < 1e-9:
                break
        placements.append(ligand_best)
        ca_xyz = next(a["xyz"] for a in ligand_best if is_ca(a))
        ca_records.append((si, j, ps, ca_xyz.copy(), best_sc))
        for a in ligand_best:
            if a["element"].upper() != "H":
                placed_heavy.append(a["xyz"].copy())
            if is_oxygen(a):
                placed_o.append(a["xyz"].copy())

    # Second pass: re-orient each rigid unit against all other units + DNA.
    water_ids = {a["serial"] for a in frag["waters"]}
    refined = []
    for idx, atoms in enumerate(placements):
        si, j, ps, ca_xyz, outward = sites[idx]
        others = [p for k, p in enumerate(placements) if k != idx]
        other_xyz, other_o = [], []
        for unit in others:
            for a in unit:
                if a["element"].upper() != "H":
                    other_xyz.append(a["xyz"])
                if is_oxygen(a):
                    other_o.append(a["xyz"])
        other = np.array(other_xyz) if other_xyz else np.zeros((0, 3))
        other_o = np.array(other_o) if other_o else np.zeros((0, 3))
        g = group_by_p[ps]
        origin, axis = strand_axis[id(next(s for s in strands if ps in s))]
        _, _, that, bhat, _ = radial_frame(g["p"]["xyz"], origin, axis)
        best, best_sc = atoms, 1e99
        for ox_dir in (outward, -outward, that, -that, bhat, -bhat):
            if np.linalg.norm(ox_dir) < 0.2:
                continue
            for deg in range(0, 360, 8):
                trial = transform_fragment(frag, ca_xyz, ox_dir, math.radians(deg))
                ligands = np.array([a["xyz"] for a in trial if a["element"].upper() != "CA"])
                lig_o = np.array([a["xyz"] for a in trial if is_oxygen(a)])
                sc = clash_score(ligands, lig_o, dna_xyz, dna_o, other, other_o)
                if sc < best_sc:
                    best_sc, best = sc, trial
                if sc < 1e-9:
                    break
            if best_sc < 1e-9:
                break
        refined.append(best)
    placements = refined

    # Drop hydrate waters that still violate inter-unit O–O; keep oxalate intact.
    for idx, atoms in enumerate(placements):
        others_o = []
        for k, unit in enumerate(placements):
            if k == idx:
                continue
            others_o.extend(a["xyz"] for a in unit if is_oxygen(a))
        others_o = np.array(others_o + list(dna_o)) if len(dna_o) else np.array(others_o)
        kept = []
        for a in atoms:
            if a["serial"] in water_ids and len(others_o):
                if float(np.linalg.norm(others_o - a["xyz"], axis=1).min()) < MIN_O_O:
                    continue
            kept.append(a)
        placements[idx] = kept

    # Tiny rigid translations so leftover oxalate–DNA O–O reach 2.0 Å.
    for _ in range(10):
        moved = False
        for idx, atoms in enumerate(placements):
            foreign = [p.copy() for p in dna_o]
            for k, unit in enumerate(placements):
                if k == idx:
                    continue
                foreign.extend(a["xyz"].copy() for a in unit if is_oxygen(a))
            if not foreign:
                continue
            foreign = np.array(foreign)
            push = np.zeros(3)
            for a in atoms:
                if not is_oxygen(a):
                    continue
                delta = a["xyz"] - foreign
                d = np.linalg.norm(delta, axis=1)
                j = int(np.argmin(d))
                if d[j] < MIN_O_O:
                    nrm = d[j] if d[j] > 1e-8 else 1.0
                    push += (MIN_O_O - d[j] + 0.02) * delta[j] / nrm
                    moved = True
            if np.linalg.norm(push) > 1e-8:
                for a in atoms:
                    a["xyz"] = a["xyz"] + push
        if not moved:
            break
    for rec_i, rec in enumerate(ca_records):
        ca = next(a["xyz"] for a in placements[rec_i] if is_ca(a))
        ca_records[rec_i] = (rec[0], rec[1], rec[2], ca.copy(), rec[4])

    # ---- report ----
    seq_d = []
    lines = []
    lines.append("DNA + calcium oxalate hydrate backbone model")
    lines.append("=" * 60)
    lines.append(f"DNA source : {DNA_PDB.name}")
    lines.append(f"CaOx source: {CAOX_PDB.name}")
    lines.append(f"Phosphates : {len(groups)}")
    lines.append(
        f"Strands    : {len(strands)} decorated "
        f"({', '.join(str(len(s)) for s in strands)} P each)"
    )
    lines.append(f"Duplexes   : {len(duplexes)}")
    lines.append(f"CaOx units : {len(placements)}")
    lines.append("")
    lines.append("Sequential P–P distances on decorated strands (Å)")
    for si, strand in enumerate(strands):
        lines.append(f"  Strand {si + 1}:")
        for a, b in zip(strand, strand[1:]):
            d = float(np.linalg.norm(lookup[a]["xyz"] - lookup[b]["xyz"]))
            lines.append(f"    P {a:4d} – P {b:4d}   {d:6.3f}")
    lines.append("")
    lines.append("Sequential Ca–Ca distances (Å)  [target 6.0–7.0]")
    for si, strand in enumerate(strands):
        cas = [rec for rec in ca_records if rec[0] == si]
        cas.sort(key=lambda r: r[1])
        lines.append(f"  Strand {si + 1}:")
        for a, b in zip(cas, cas[1:]):
            d = float(np.linalg.norm(a[3] - b[3]))
            seq_d.append(d)
            flag = "OK" if MIN_CA_CA <= d <= 7.0 else ("LOW" if d < MIN_CA_CA else "HIGH")
            lines.append(f"    Ca@{a[2]:4d} – Ca@{b[2]:4d}   {d:6.3f}  {flag}")
    in_range = 0
    if seq_d:
        arr = np.array(seq_d)
        in_range = int(((arr >= MIN_CA_CA) & (arr <= 7.0)).sum())
        lines.append("")
        lines.append(
            f"Ca–Ca sequential: n={len(arr)}  min={arr.min():.3f}  "
            f"median={np.median(arr):.3f}  max={arr.max():.3f}  "
            f"mean={arr.mean():.3f}"
        )
        lines.append(f"In 6.0–7.0 Å window: {in_range}/{len(arr)}")

    lines.append("")
    # All-pair steric check on the finished rigid units
    check_atoms = []
    for site, atoms in enumerate(placements, start=1):
        for a in atoms:
            check_atoms.append(
                {"element": a["element"], "name": a["name"], "resname": "COM",
                 "chain": "X", "resseq": site, "xyz": a["xyz"]}
            )
    tags_tmp = assign_nucleotide_residues(
        dna_atoms, dna_conect, strands, lookup, duplexes
    )
    for a in dna_atoms:
        ch, rs = tags_tmp[a["serial"]]
        check_atoms.append(
            {"element": a["element"], "name": a["name"], "resname": "NUC",
             "chain": ch, "resseq": rs, "xyz": a["xyz"]}
        )
    ster = short_contact_summary(check_atoms)
    lines.append("")
    lines.append("Rigid-unit steric rules: Ca–Ca ≥ 6.0 Å, inter-unit O–O ≥ 2.0 Å")
    lines.append(
        f"  All Ca–Ca < 6.0 Å: {ster['n_ca_short']}"
        + (f"  (min {ster['ca_min']:.3f})" if ster["ca_min"] is not None else "")
    )
    lines.append(
        f"  Inter-residue O–O < 2.0 Å: {ster['n_oo_short']}"
        + (f"  (min {ster['oo_min']:.3f})" if ster["oo_min"] is not None else "")
    )
    lines.append("")
    lines.append("Ca–O(nonbridging phosphate) and Ca–P distances (Å)")
    for rec in ca_records:
        g = group_by_p[rec[2]]
        ca = rec[3]
        d_op = sorted(float(np.linalg.norm(ca - o["xyz"])) for o in g["op"])
        d_p = float(np.linalg.norm(ca - g["p"]["xyz"]))
        dstr = "  ".join(f"{d:.3f}" for d in d_op[:3])
        lines.append(f"  Ca @ P {rec[2]:4d}:  OP {dstr}   P {d_p:.3f}")

    report = "\n".join(lines) + "\n"
    REPORT.write_text(report)
    print(report)

    # ---- PDB ----
    tags = assign_nucleotide_residues(
        dna_atoms, dna_conect, strands, lookup, duplexes
    )
    remarks = [
        "REMARK   1 DNA (1BNA CrystalMaker export) with CaC2O4.nH2O",
        "REMARK   1 at every phosphate along the sugar-phosphate backbone.",
        "REMARK   2 B-DNA sequential P-P is 6.2-7.1 A; whewellite Ca-Ca along",
        "REMARK   2 the 6.29 A axis matches this spacing. Ca2+ is placed on the",
        "REMARK   2 phosphate helical cylinder so sequential Ca-Ca stays 6-7 A",
        "REMARK   2 and each Ca remains near a non-bridging phosphate oxygen.",
        "REMARK   3 Each CaOx hydrate is a rigid unit. Inter-unit Ca-Ca >= 6.0 A",
        "REMARK   3 and O-O >= 2.0 A (intramolecular O-O is unchanged).",
        "REMARK   4 Chains A+B and C+D are the two 1BNA-packing duplexes.",
        "REMARK   4 Chain X = COM residues (one CaOx hydrate unit per phosphate).",
    ]
    if seq_d:
        arr = np.array(seq_d)
        remarks.append(
            f"REMARK   5 Sequential Ca-Ca: mean {arr.mean():.2f} A "
            f"(min {arr.min():.2f}, max {arr.max():.2f}); "
            f"{in_range}/{len(arr)} in 6.0-7.0 A."
        )

    out = [
        "HEADER    DNA-TEMPLATED CALCIUM OXALATE HYDRATE\n",
        "TITLE     1BNA PHOSPHATE BACKBONE WITH COM UNITS AT 6-7 A CA-CA\n",
    ]
    for r in remarks:
        out.append(r + "\n")

    prev_chain = None
    for a in sorted(
        dna_atoms, key=lambda t: (tags[t["serial"]][0], tags[t["serial"]][1], t["serial"])
    ):
        chain, resseq = tags[a["serial"]]
        if prev_chain is not None and chain != prev_chain:
            out.append("TER\n")
        el = "Ca" if a["element"].upper() == "CA" else a["element"]
        out.append(
            format_atom(
                "ATOM",
                a["serial"],
                a["name"],
                "NUC",
                chain,
                resseq,
                a["xyz"],
                1.00,
                0.00,
                el,
            )
        )
        prev_chain = chain
    out.append("TER\n")

    max_serial = max(a["serial"] for a in dna_atoms)
    serial = max_serial
    bonds = defaultdict(list)
    for a0, nbrs in dna_conect.items():
        for b in nbrs:
            bonds[a0].append(b)

    carbon_ids = {a["serial"] for a in frag["carbons"]}
    oxo_ids = {a["serial"] for a in frag["ox_o"]}
    water_ids = {a["serial"] for a in frag["waters"]}
    ca_id = frag["ca"]["serial"]

    for site, atoms in enumerate(placements, start=1):
        local_map = {}
        c_i = o_ox = o_w = 0
        ca_serial = None
        for a in atoms:
            if a["serial"] in water_ids:
                dmin = float(np.linalg.norm(dna_xyz - a["xyz"], axis=1).min())
                if dmin < 2.10:
                    continue  # water displaced by DNA
            serial += 1
            local_map[a["serial"]] = serial
            if a["serial"] == ca_id:
                name, el = "CA", "Ca"
                ca_serial = serial
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
            out.append(
                format_atom(
                    "HETATM",
                    serial,
                    name,
                    "COM",
                    "X",
                    site,
                    a["xyz"],
                    1.00,
                    20.00,
                    el,
                )
            )
        for old_a, nbrs in frag["conect"].items():
            if old_a not in local_map:
                continue
            for old_b in nbrs:
                if old_b in local_map:
                    bonds[local_map[old_a]].append(local_map[old_b])
        g = group_by_p[ca_records[site - 1][2]]
        ca_xyz = next(a["xyz"] for a in atoms if a["element"].upper() == "CA")
        op_sorted = sorted(g["op"], key=lambda o: np.linalg.norm(ca_xyz - o["xyz"]))
        if ca_serial and op_sorted:
            bonds[ca_serial].append(op_sorted[0]["serial"])
            bonds[op_sorted[0]["serial"]].append(ca_serial)
            if len(op_sorted) > 1 and np.linalg.norm(ca_xyz - op_sorted[1]["xyz"]) < 3.2:
                bonds[ca_serial].append(op_sorted[1]["serial"])
                bonds[op_sorted[1]["serial"]].append(ca_serial)

    for s in sorted(bonds):
        partners = sorted(set(bonds[s]))
        for i in range(0, len(partners), 4):
            chunk = partners[i : i + 4]
            out.append(f"CONECT{s:5d}" + "".join(f"{p:5d}" for p in chunk) + "\n")
    out.append("END\n")
    OUT_PDB.write_text("".join(out))
    print(f"Wrote {OUT_PDB}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
