#!/usr/bin/env python3
"""
Gel-first DNA–CaOx model: every phosphate gets a CaOx hydrate unit.

Unlike grow_crystal_from_growth.py (whewellite unit-cell cut), this path:
  * places Ca2+ at each phosphate using bond-valence OP chelation
  * attaches a rigid CaC2O4·nH2O fragment with *independent* orientation
    per site (no lattice alignment, no COM Ca–Ca targets)
  * enforces only steric rules (Ca–Ca ≥ 6 Å, inter-unit O–O ≥ 2 Å)

Use this as the honest nucleation/gel starting point before optional
clash relief (DLS or FIRE *without* whewellite distance restraints).
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dna_caox import (  # noqa: E402
    CAOX_PDB,
    DNA_PDB,
    assign_nucleotide_residues,
    by_serial,
    clash_score,
    format_atom,
    load_caox_fragment,
    order_strands,
    pair_duplexes,
    parse_pdb_atoms,
    phosphate_groups,
    sequential_p_pairs,
    transform_fragment,
)
from geom_constraints import (  # noqa: E402
    MIN_CA_CA,
    MIN_O_O,
    is_ca,
    is_oxygen,
    separate_ca,
    short_contact_summary,
    xyz_of,
)
from phosphate_ca_binding import (  # noqa: E402
    D_TARGET,
    binding_site_summary,
    optimal_ca_at_phosphate,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_PDB = ROOT / "DNA_CaOx_gel_first.pdb"
OUT_SEEDS = ROOT / "DNA_CaOx_gel_first_seeds.pdb"
REPORT = ROOT / "DNA_CaOx_gel_first_report.txt"
RNG = np.random.default_rng(2026)


def strand_tangent(strand_sites, idx: int):
    """Helix tangent at site index within one strand's site list."""
    n = len(strand_sites)
    p = strand_sites[idx]["ca"]
    if 0 < idx < n - 1:
        t = strand_sites[idx + 1]["ca"] - strand_sites[idx - 1]["ca"]
    elif idx < n - 1:
        t = strand_sites[idx + 1]["ca"] - p
    else:
        t = p - strand_sites[idx - 1]["ca"]
    nrm = float(np.linalg.norm(t))
    return t / nrm if nrm > 1e-8 else np.array([0.0, 1.0, 0.0])


def place_gel_units_geometry(sites, frag, dna_xyz, dna_o, *, n_twist: int = 24):
    """Oxalate plane biased outward; twist scanned for sterics (not random)."""
    by_strand = defaultdict(list)
    for s in sites:
        by_strand[s["si"]].append(s)
    for si in by_strand:
        by_strand[si].sort(key=lambda r: r["j"])

    placements = []
    placed_heavy, placed_o = [], []

    for site in sites:
        ca_xyz = site["ca"]
        outward = site["outward"]
        strand_sites = by_strand[site["si"]]
        local_i = next(i for i, s in enumerate(strand_sites) if s is site)
        tangent = strand_tangent(strand_sites, local_i)
        best_atoms, best_sc = None, 1e99
        other = np.array(placed_heavy) if placed_heavy else np.zeros((0, 3))
        other_o = np.array(placed_o) if placed_o else np.zeros((0, 3))

        for k in range(n_twist):
            ang = k * (2 * math.pi / n_twist)
            atoms = transform_fragment(frag, ca_xyz, outward, ang)
            lig = np.array([a["xyz"] for a in atoms if a["element"].upper() != "CA"])
            lig_o = np.array([a["xyz"] for a in atoms if is_oxygen(a)])
            sc = clash_score(lig, lig_o, dna_xyz, dna_o, other, other_o)
            if sc < best_sc:
                best_sc, best_atoms = sc, atoms
            if sc < 1e-9:
                break

        placements.append(best_atoms)
        site["clash"] = best_sc
        site["tangent"] = tangent
        for a in best_atoms:
            if a["element"].upper() != "H":
                placed_heavy.append(a["xyz"].copy())
            if is_oxygen(a):
                placed_o.append(a["xyz"].copy())

    return placements


def write_seed_pdb(path: Path, dna_atoms, seed_strands, remarks: list[str]):
    """DNA + COM Ca at every phosphate (chain X)."""
    serial = 1
    lines = list(remarks)
    for a in dna_atoms:
        lines.append(
            format_atom(
                "ATOM",
                serial,
                a["name"],
                a.get("resname", "NUC"),
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
    for strand in seed_strands:
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
            serial += 1
            res += 1
    lines.append("END\n")
    path.write_text("".join(lines))


def fit_helix_axis(points):
    pts = np.asarray(points, float)
    origin = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - origin)
    axis = vt[0]
    n = float(np.linalg.norm(axis))
    return origin, axis / n if n > 1e-8 else axis


def place_gel_units(
    sites,
    frag,
    dna_xyz,
    dna_o,
    *,
    n_twist: int = 24,
):
    """Attach one rigid CaOx per site with independent random azimuth."""
    placements = []
    placed_heavy, placed_o = [], []

    for idx, site in enumerate(sites):
        ca_xyz = site["ca"]
        p_xyz = site["p_xyz"]
        outward = site["outward"]
        twist0 = float(RNG.uniform(0, 2 * math.pi))
        best_atoms, best_sc = None, 1e99
        other = np.array(placed_heavy) if placed_heavy else np.zeros((0, 3))
        other_o = np.array(placed_o) if placed_o else np.zeros((0, 3))

        # Oxalate points roughly away from DNA; spin is uncorrelated between sites.
        for k in range(n_twist):
            ang = twist0 + k * (2 * math.pi / n_twist)
            atoms = transform_fragment(frag, ca_xyz, outward, ang)
            lig = np.array([a["xyz"] for a in atoms if a["element"].upper() != "CA"])
            lig_o = np.array([a["xyz"] for a in atoms if is_oxygen(a)])
            sc = clash_score(lig, lig_o, dna_xyz, dna_o, other, other_o)
            if sc < best_sc:
                best_sc, best_atoms = sc, atoms
            if sc < 1e-9:
                break

        placements.append(best_atoms)
        ca_xyz = next(a["xyz"] for a in best_atoms if is_ca(a))
        site["ca"] = ca_xyz
        site["clash"] = best_sc
        for a in best_atoms:
            if a["element"].upper() != "H":
                placed_heavy.append(a["xyz"].copy())
            if is_oxygen(a):
                placed_o.append(a["xyz"].copy())

    return placements


def refine_orientations(placements, sites, frag, dna_xyz, dna_o, n_twist: int = 24):
    refined = []
    for idx, atoms in enumerate(placements):
        ca_xyz = sites[idx]["ca"]
        outward = sites[idx]["outward"]
        twist0 = float(RNG.uniform(0, 2 * math.pi))
        other_xyz, other_o = [], []
        for k, unit in enumerate(placements):
            if k == idx:
                continue
            for a in unit:
                if a["element"].upper() != "H":
                    other_xyz.append(a["xyz"])
                if is_oxygen(a):
                    other_o.append(a["xyz"])
        other = np.array(other_xyz) if other_xyz else np.zeros((0, 3))
        other_o = np.array(other_o) if other_o else np.zeros((0, 3))
        best, best_sc = atoms, 1e99
        for k in range(n_twist):
            ang = twist0 + k * (2 * math.pi / n_twist)
            trial = transform_fragment(frag, ca_xyz, outward, ang)
            lig = np.array([a["xyz"] for a in trial if a["element"].upper() != "CA"])
            lig_o = np.array([a["xyz"] for a in trial if is_oxygen(a)])
            sc = clash_score(lig, lig_o, dna_xyz, dna_o, other, other_o)
            if sc < best_sc:
                best_sc, best = sc, trial
        refined.append(best)
        sites[idx]["clash"] = best_sc
    return refined


def write_gel_pdb(
    path: Path,
    dna_atoms,
    dna_conect,
    strands,
    duplexes,
    lookup,
    placements,
    sites,
    frag,
    remarks: list[str],
):
    tags = assign_nucleotide_residues(
        dna_atoms, dna_conect, strands, lookup, duplexes
    )
    out = [
        "HEADER    DNA-CAOX GEL-FIRST (NO WHEWELLITE LATTICE)\n",
        "TITLE     CAOX HYDRATE AT EVERY PHOSPHATE; INDEPENDENT ORIENTATIONS\n",
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
                "ATOM", a["serial"], a["name"], "NUC", chain, resseq,
                a["xyz"], 1.0, 0.0, el,
            )
        )
        prev_chain = chain
    out.append("TER\n")

    dna_xyz = np.array([a["xyz"] for a in dna_atoms])
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

    for site_i, atoms in enumerate(placements, start=1):
        local_map = {}
        c_i = o_ox = o_w = 0
        ca_serial = None
        for a in atoms:
            if a["serial"] in water_ids:
                if float(np.linalg.norm(dna_xyz - a["xyz"], axis=1).min()) < 2.10:
                    continue
            serial += 1
            local_map[a["serial"]] = serial
            if a["serial"] == ca_id:
                name, el, ca_serial = "CA", "Ca", serial
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
                    "HETATM", serial, name, "COM", "X", site_i,
                    a["xyz"], 1.0, 20.0, el,
                )
            )
        for old_a, nbrs in frag["conect"].items():
            if old_a not in local_map:
                continue
            for old_b in nbrs:
                if old_b in local_map:
                    bonds[local_map[old_a]].append(local_map[old_b])
        g = sites[site_i - 1]
        ca_xyz = g["ca"]
        op_sorted = sorted(g["ops"], key=lambda o: np.linalg.norm(ca_xyz - o["xyz"]))
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
    path.write_text("".join(out))


def main():
    ap = argparse.ArgumentParser(description="Gel-first DNA–CaOx at phosphates (no xtl lattice)")
    ap.add_argument(
        "--alt-p",
        action="store_true",
        help="Place CaOx at every other phosphate per strand (~6.8 Å Ca spacing).",
    )
    ap.add_argument(
        "--orient",
        choices=("random", "geometry"),
        default="random",
        help="Oxalate placement: random twist per site or geometry-biased outward.",
    )
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--seeds", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    tag = "gel_altP_geom" if args.alt_p and args.orient == "geometry" else (
        "gel_altP" if args.alt_p else (
            "gel_geom" if args.orient == "geometry" else "gel_first"
        )
    )
    out_pdb = args.output or ROOT / f"DNA_CaOx_{tag}.pdb"
    out_seeds = args.seeds or ROOT / f"DNA_CaOx_{tag}_seeds.pdb"
    report_path = args.report or ROOT / f"DNA_CaOx_{tag}_report.txt"

    dna_atoms, dna_conect = parse_pdb_atoms(DNA_PDB)
    lookup = by_serial(dna_atoms)
    groups = phosphate_groups(dna_atoms, dna_conect)
    group_by_p = {g["p"]["serial"]: g for g in groups}
    p_serials = [g["p"]["serial"] for g in groups]

    pairs = sequential_p_pairs(dna_atoms, dna_conect, p_serials)
    strands = [s for s in order_strands(p_serials, pairs) if len(s) >= 3]
    duplexes = pair_duplexes(strands, lookup)

    strand_axis = {}
    for duplex in duplexes:
        pts = [lookup[p]["xyz"] for s in duplex for p in s]
        origin, axis = fit_helix_axis(pts)
        for s in duplex:
            strand_axis[id(s)] = (origin, axis)

    frag = load_caox_fragment(CAOX_PDB)
    dna_xyz = np.array([a["xyz"] for a in dna_atoms])
    dna_o = xyz_of([a for a in dna_atoms if is_oxygen(a)])

    sites = []
    for si, strand in enumerate(strands):
        origin, axis = strand_axis[id(strand)]
        for j, ps in enumerate(strand):
            if args.alt_p and (j % 2):
                continue
            g = group_by_p[ps]
            ops = [o["xyz"] for o in g["op"]]
            ca, bv = optimal_ca_at_phosphate(
                g["p"]["xyz"], ops, origin, axis,
            )
            radial = g["p"]["xyz"] - origin
            radial = radial - axis * float(np.dot(radial, axis))
            outward = radial / max(float(np.linalg.norm(radial)), 1e-8)
            sites.append(
                {
                    "si": si,
                    "j": j,
                    "ps": ps,
                    "p_xyz": g["p"]["xyz"],
                    "ops": g["op"],
                    "ca": ca,
                    "bv": bv,
                    "outward": outward,
                }
            )

    ca_pts = separate_ca([s["ca"] for s in sites], MIN_CA_CA)
    for s, ca in zip(sites, ca_pts):
        s["ca"] = ca

    placements = (
        place_gel_units_geometry(sites, frag, dna_xyz, dna_o)
        if args.orient == "geometry"
        else place_gel_units(sites, frag, dna_xyz, dna_o)
    )
    if args.orient == "random":
        placements = refine_orientations(placements, sites, frag, dna_xyz, dna_o)

    # Drop clashing waters; nudge oxalate O off DNA if needed.
    water_ids = {a["serial"] for a in frag["waters"]}
    for idx, atoms in enumerate(placements):
        others_o = list(dna_o)
        for k, unit in enumerate(placements):
            if k == idx:
                continue
            others_o.extend(a["xyz"] for a in unit if is_oxygen(a))
        others_o = np.array(others_o)
        kept = []
        for a in atoms:
            if a["serial"] in water_ids and len(others_o):
                if float(np.linalg.norm(others_o - a["xyz"], axis=1).min()) < MIN_O_O:
                    continue
            kept.append(a)
        placements[idx] = kept

    # Report
    orient_desc = (
        "geometry-biased outward (O–P–O plane, twist scanned)"
        if args.orient == "geometry"
        else "random orientation per site"
    )
    p_desc = "every other P per strand" if args.alt_p else "every phosphate"
    lines = [
        f"DNA–CaOx gel-first model ({tag}; no whewellite lattice)",
        "=" * 60,
        binding_site_summary(),
        "",
        f"DNA source : {DNA_PDB.name}",
        f"CaOx source: {CAOX_PDB.name} (rigid fragment, {orient_desc})",
        f"Phosphates : {len(groups)} total  →  {len(sites)} CaOx units ({p_desc})",
        f"Strands    : {len(strands)} ({', '.join(str(len(s)) for s in strands)} P each)",
        "",
        "This is NOT cut from Whewellite - xtl.pdb. Order emerges only after",
        "optional relaxation without COM distance restraints.",
        "",
        f"Ca–OP target (BV, 6-fold): {D_TARGET:.3f} Å",
        "",
        "Per-phosphate Ca placement (Å / v.u.)",
    ]
    bv_sums = []
    for s in sites:
        g = group_by_p[s["ps"]]
        ca = s["ca"]
        d_ops = sorted(float(np.linalg.norm(ca - o["xyz"])) for o in g["op"])
        dstr = "  ".join(f"{d:.3f}" for d in d_ops[:2])
        bv = s["bv"]
        bv_sums.append(bv.get("bv_sum", 0.0))
        lines.append(
            f"  P {s['ps']:4d}: OP {dstr}   P {float(np.linalg.norm(ca - g['p']['xyz'])):.3f}  "
            f"BV(P)={bv.get('bv_sum', 0):.2f}  mode={bv.get('mode', '?')}"
        )

    seq_d = []
    lines.append("")
    lines.append("Sequential Ca–Ca along strands (Å)  [DNA P–P is ~6.2–7.0; not forced]")
    for si, strand in enumerate(strands):
        recs = [s for s in sites if s["si"] == si]
        recs.sort(key=lambda r: r["j"])
        lines.append(f"  Strand {si + 1}:")
        for a, b in zip(recs, recs[1:]):
            d = float(np.linalg.norm(a["ca"] - b["ca"]))
            seq_d.append(d)
            flag = "OK" if MIN_CA_CA <= d <= 7.5 else ("LOW" if d < MIN_CA_CA else "HIGH")
            lines.append(f"    Ca@{a['ps']:4d} – Ca@{b['ps']:4d}   {d:6.3f}  {flag}")

    if seq_d:
        arr = np.array(seq_d)
        lines.append("")
        lines.append(
            f"Ca–Ca sequential: n={len(arr)}  min={arr.min():.3f}  "
            f"median={np.median(arr):.3f}  max={arr.max():.3f}"
        )

    check_atoms = []
    for i, atoms in enumerate(placements, start=1):
        for a in atoms:
            check_atoms.append(
                {
                    "element": a["element"], "name": a["name"], "resname": "COM",
                    "chain": "X", "resseq": i, "xyz": a["xyz"],
                }
            )
    tags_tmp = assign_nucleotide_residues(
        dna_atoms, dna_conect, strands, lookup, duplexes
    )
    for a in dna_atoms:
        ch, rs = tags_tmp[a["serial"]]
        check_atoms.append(
            {
                "element": a["element"], "name": a["name"], "resname": "NUC",
                "chain": ch, "resseq": rs, "xyz": a["xyz"],
            }
        )
    ster = short_contact_summary(check_atoms)
    lines.append("")
    lines.append(f"Median BV from phosphate O only: {np.median(bv_sums):.2f} v.u.")
    lines.append(f"(Full Ca2+ balance needs oxalate + water; target total = 2.0 v.u.)")
    lines.append("")
    lines.append("Sterics (rigid units)")
    lines.append(
        f"  Ca–Ca < {MIN_CA_CA} Å: {ster['n_ca_short']}"
        + (f"  (min {ster['ca_min']:.3f})" if ster["ca_min"] is not None else "")
    )
    lines.append(
        f"  Inter-residue O–O < {MIN_O_O} Å: {ster['n_oo_short']}"
        + (f"  (min {ster['oo_min']:.3f})" if ster["oo_min"] is not None else "")
    )

    report = "\n".join(lines) + "\n"
    report_path.write_text(report)
    print(report)

    remarks = [
        f"REMARK   1 Gel-first ({tag}): CaOx hydrate at {p_desc}; no xtl lattice.",
        "REMARK   2 Ca2+ placed by bond-valence OP chelation (~2.37 A target).",
        f"REMARK   3 Oxalate orientation: {orient_desc}.",
        "REMARK   4 Relax with fire_openmm; use --no-com-targets for honest gel.",
        f"REMARK   5 {len(sites)} COM units on chain X.",
    ]
    write_gel_pdb(
        out_pdb, dna_atoms, dna_conect, strands, duplexes, lookup,
        placements, sites, frag, remarks,
    )

    # Seed PDB for symmetry / viewer (flat COM Ca list, one per phosphate).
    seed_strands = []
    for strand in strands:
        row = []
        for j, ps in enumerate(strand):
            if args.alt_p and (j % 2):
                continue
            ca = next(s["ca"] for s in sites if s["ps"] == ps)
            row.append({"xyz": ca.copy()})
        if row:
            seed_strands.append(row)
    dna_for_seeds = [
        {
            "name": a["name"],
            "resname": "NUC",
            "chain": tags_tmp[a["serial"]][0],
            "resseq": tags_tmp[a["serial"]][1],
            "xyz": a["xyz"],
            "element": a["element"],
        }
        for a in dna_atoms
    ]
    write_seed_pdb(
        out_seeds,
        dna_for_seeds,
        seed_strands,
        [
            f"REMARK   1 COM Ca seeds for {tag}\n",
            f"REMARK   2 {len(sites)} seeds on chain X ({p_desc})\n",
        ],
    )

    print(f"Wrote {out_pdb}")
    print(f"Wrote {out_seeds}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
