#!/usr/bin/env python3
"""
OpenMM nucleation MD: DNA (1BNA) vs no-DNA, same Ca2+ / oxalate / water.

This is the kinetics test the packed+FIRE models cannot do. Default is
minimization + a short NVT smoke run. For production:

  .venv/bin/python scripts/md_nucleation.py --ns 10

Force fields: amber14 DNA (OL15) + TIP3P-FB water + Joung/Cheatham Ca2+
+ oxalate XML (scripts/ff/oxalate.xml). DNA from DOCS/1BNA.pdb (real
DC/DG/DA/DT names), not CrystalMaker NUC.

Ca and oxalate start in solution (not inner-sphere on P) so the run can
ask whether DNA organizes them. The no-DNA arm uses the same box size
and the same ion counts.

Observables: figures/crystallinity/DNA_CaOx_md_observables.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

DNA_PDB = ROOT / "DOCS" / "1BNA.pdb"
CAOX_FRAG = ROOT / "DOCS" / "Whewellite ca_ox.pdb"
OXL_FF = ROOT / "scripts" / "ff" / "oxalate.xml"
OUT_DIR = ROOT / "md"
FIG_DIR = ROOT / "figures" / "crystallinity"

ATOM_RE = re.compile(
    r"^(ATOM|HETATM)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S)\s+(-?\d+)\s+"
    r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+(\S+)\s*$"
)


def oxalate_rel():
    from grow_whewellite import parse_atoms

    atoms, _ = parse_atoms(CAOX_FRAG)
    carbons = [a for a in atoms if a["element"].upper() == "C"]
    oxygens = [a for a in atoms if a["element"].upper() == "O"]
    ccent = np.mean([a["xyz"] for a in carbons], axis=0)
    carbons.sort(key=lambda a: a["name"])
    oxygens.sort(key=lambda a: np.linalg.norm(a["xyz"] - ccent))
    names = ["C1", "C2", "O1", "O2", "O3", "O4"]
    pts = [a["xyz"] - ccent for a in carbons[:2]] + [a["xyz"] - ccent for a in oxygens[:4]]
    return list(zip(names, pts))


def pdb_atom_name(name: str) -> str:
    name = name.strip()
    if len(name) >= 4:
        return name[:4]
    return f" {name:<3}"[:4]


def write_hetatm(serial: int, name: str, resn: str, chain: str, resid: int, xyz, element: str) -> str:
    an = pdb_atom_name(name)
    return (
        f"HETATM{serial:5d} {an} {resn:>3s} {chain}{resid:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 20.00          {element:>2s}\n"
    )


def rewrite_1bna(src: Path, dest: Path) -> None:
    """Column-pad 1BNA ATOM records; drop crystal waters (we add solvent)."""
    lines = ["HEADER    1BNA FOR OPENMM\n"]
    prev_chain = None
    serial = 1
    for raw in src.read_text().splitlines():
        m = ATOM_RE.match(raw.rstrip())
        if not m:
            continue
        rec, _ser, name, resn, chain, resid, x, y, z, occ, bfac, el = m.groups()
        if resn == "HOH":
            continue
        if prev_chain is not None and chain != prev_chain:
            lines.append("TER\n")
        prev_chain = chain
        an = pdb_atom_name(name)
        lines.append(
            f"ATOM  {serial:5d} {an} {resn:>3s} {chain}{int(resid):4d}    "
            f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}"
            f"{float(occ):6.2f}{float(bfac):6.2f}          {el:>2s}\n"
        )
        serial += 1
    lines.append("TER\nEND\n")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(lines))


def write_oxl_pdb(path: Path, centers: list[np.ndarray]):
    rel = oxalate_rel()
    serial = 1
    lines = ["HEADER    OXALATE DROPLET\n"]
    conect = []
    for res, c in enumerate(centers, start=1):
        base = serial
        for name, xyz in rel:
            p = xyz + c
            el = name[0]
            lines.append(write_hetatm(serial, name, "OXL", "X", res, p, el))
            serial += 1
        c1, c2, o1, o2, o3, o4 = range(base, base + 6)
        conect.append((c1, c2, o1, o2))
        conect.append((c2, o3, o4))
    for row in conect:
        lines.append("CONECT" + "".join(f"{i:5d}" for i in row) + "\n")
    lines.append("END\n")
    path.write_text("".join(lines))


def add_oxl_bonds(topology):
    existing = set()
    for a, b in topology.bonds():
        existing.add((min(a.index, b.index), max(a.index, b.index)))
    pairs = [("C1", "C2"), ("C1", "O1"), ("C1", "O2"), ("C2", "O3"), ("C2", "O4")]
    for res in topology.residues():
        if res.name != "OXL":
            continue
        atoms = {a.name.strip(): a for a in res.atoms()}
        if "C1" not in atoms:
            continue
        for n1, n2 in pairs:
            i, j = atoms[n1].index, atoms[n2].index
            key = (min(i, j), max(i, j))
            if key not in existing:
                topology.addBond(atoms[n1], atoms[n2])
                existing.add(key)


def write_ca_pdb(path: Path, centers: list[np.ndarray]):
    lines = ["HEADER    CALCIUM IONS\n"]
    for i, xyz in enumerate(centers, start=1):
        lines.append(write_hetatm(i, "CA", "CA", "Y", i, xyz, "CA"))
    lines.append("END\n")
    path.write_text("".join(lines))


def xyz_nm(positions, unit):
    return np.array([np.array(p.value_in_unit(unit.nanometer)) for p in positions])


def xyz_ang(positions, unit):
    return np.array([np.array(p.value_in_unit(unit.angstrom)) for p in positions])


def collect_p_op(topology, positions, unit):
    p_xyz, op_xyz, p_meta = [], [], []
    by_res = {}
    pos = xyz_ang(positions, unit)
    for atom in topology.atoms():
        res = atom.residue
        key = (res.chain.id if res.chain.id else "?", res.index, res.id)
        by_res.setdefault(key, {"P": None, "OP": []})
        name = (atom.name or "").strip()
        el = atom.element.symbol if atom.element else ""
        if el == "P" or name == "P":
            by_res[key]["P"] = pos[atom.index]
            p_meta.append((key[0], int(res.id) if str(res.id).isdigit() else res.index))
            p_xyz.append(pos[atom.index])
        if name in ("OP1", "OP2", "O1P", "O2P"):
            by_res[key]["OP"].append(pos[atom.index])
            op_xyz.append(pos[atom.index])
    p_arr = np.array(p_xyz) if p_xyz else np.zeros((0, 3))
    op_arr = np.array(op_xyz) if op_xyz else np.zeros((0, 3))
    return p_arr, op_arr, p_meta


def place_solution_ions(n_ca, origin, pxyz, rng, *, min_p=8.0, max_p=14.0, min_ca=4.0):
    """Ca in a 8–14 Å shell from phosphates (assembly start, not inner-sphere)."""
    ca_pos = []
    tries = 0
    while len(ca_pos) < n_ca and tries < 8000:
        tries += 1
        if len(pxyz):
            p = pxyz[int(rng.integers(0, len(pxyz)))]
            vec = rng.normal(size=3)
            vec /= max(float(np.linalg.norm(vec)), 1e-8)
            # Bias slightly outward from DNA centroid.
            out = p - origin
            on = float(np.linalg.norm(out)) or 1.0
            vec = 0.7 * vec + 0.3 * (out / on)
            vec /= max(float(np.linalg.norm(vec)), 1e-8)
            r = float(rng.uniform(min_p, max_p))
            cand = p + r * vec
            if np.linalg.norm(cand - pxyz, axis=1).min() < min_p:
                continue
        else:
            cand = origin + rng.normal(scale=6.0, size=3)
        if ca_pos:
            if min(np.linalg.norm(cand - c) for c in ca_pos) < min_ca:
                continue
        ca_pos.append(cand)
    if len(ca_pos) < n_ca:
        raise SystemExit(f"Could only place {len(ca_pos)}/{n_ca} Ca ions")
    return ca_pos


def place_oxalate(ca_pos, rng, offset=4.2):
    ox = []
    for ca in ca_pos:
        v = rng.normal(size=3)
        v /= max(float(np.linalg.norm(v)), 1e-8)
        ox.append(np.asarray(ca, float) + offset * v)
    return ox


def observables_from_positions(tag, topology, pos_ang, n_atoms):
    names = [
        (
            a.residue.name,
            (a.name or "").strip(),
            a.element.symbol if a.element else "?",
            a.residue.chain.id if a.residue.chain.id else "?",
            a.residue.id,
        )
        for a in topology.atoms()
    ]
    ca_ix = [i for i, (r, n, e, *_rest) in enumerate(names) if e == "Ca" or (r == "CA" and n == "CA")]
    p_ix = [i for i, (_r, n, e, *_rest) in enumerate(names) if e == "P" or n == "P"]
    op_ix = [i for i, (_r, n, e, *_rest) in enumerate(names) if n in ("OP1", "OP2", "O1P", "O2P")]
    ca = pos_ang[ca_ix] if ca_ix else np.zeros((0, 3))
    p = pos_ang[p_ix] if p_ix else np.zeros((0, 3))
    op = pos_ang[op_ix] if op_ix else np.zeros((0, 3))

    out = {
        "tag": tag,
        "ca": ca,
        "p": p,
        "op": op,
        "n_atoms": n_atoms,
        "n_bound": 0,
        "seq_d": np.array([]),
        "n_seq_629": 0,
        "n_seq_384": 0,
        "n_ca_op_inner": 0,
        "ca_op_min": np.array([]),
    }
    if len(ca) and len(op):
        d_op = np.linalg.norm(ca[:, None, :] - op[None, :, :], axis=2).min(axis=1)
        out["ca_op_min"] = d_op
        out["n_ca_op_inner"] = int((d_op < 2.7).sum())
        bound = np.where(d_op < 2.7)[0]
        out["n_bound"] = int(len(bound))
        if len(p_ix) and len(bound):
            p_meta = []
            for i in p_ix:
                chain, resid = names[i][3], names[i][4]
                rid = int(resid) if str(resid).isdigit() else i
                p_meta.append((chain, rid, pos_ang[i]))
            seq = []
            by_chain = {}
            for bi in bound:
                d = [np.linalg.norm(ca[bi] - pm[2]) for pm in p_meta]
                j = int(np.argmin(d))
                chain, rid, _ = p_meta[j]
                by_chain.setdefault(chain, []).append((rid, ca[bi]))
            for chain, items in by_chain.items():
                items.sort(key=lambda t: t[0])
                for a, b in zip(items, items[1:]):
                    if b[0] - a[0] <= 2:
                        seq.append(float(np.linalg.norm(a[1] - b[1])))
            out["seq_d"] = np.array(seq, float)
            if len(seq):
                out["n_seq_629"] = int(np.sum(np.abs(out["seq_d"] - 6.29) <= 0.40))
                out["n_seq_384"] = int(np.sum(np.abs(out["seq_d"] - 3.843) <= 0.25))
    return out


def build_and_run(
    *,
    with_dna: bool,
    n_ca: int,
    n_ox: int,
    pad_nm: float,
    temperature: float,
    ns: float,
    dt_fs: float,
    platform_name: str | None,
    tag: str,
    box_nm=None,
    min_iters: int = 400,
):
    import openmm
    from openmm import app, unit

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7 if with_dna else 11)

    ff_files = ["amber14-all.xml", "amber14/tip3pfb.xml"]
    if OXL_FF.exists():
        ff_files.append(str(OXL_FF))
    ff = app.ForceField(*ff_files)

    origin = np.zeros(3)
    pxyz = np.zeros((0, 3))
    modeller = None
    if with_dna:
        if not DNA_PDB.exists():
            raise SystemExit(f"Missing {DNA_PDB}")
        cleaned = OUT_DIR / "1bna_openmm.pdb"
        rewrite_1bna(DNA_PDB, cleaned)
        pdb = app.PDBFile(str(cleaned))
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(ff)
        pxyz, _op, _meta = collect_p_op(modeller.topology, modeller.positions, unit)
        dna_xyz = xyz_ang(modeller.positions, unit)
        origin = dna_xyz.mean(axis=0)
        ca_pos = place_solution_ions(n_ca, origin, pxyz, rng)
    else:
        origin = (10.0 * np.array(box_nm) / 2.0) if box_nm is not None else np.zeros(3)
        ca_pos = []
        tries = 0
        while len(ca_pos) < n_ca and tries < 4000:
            tries += 1
            if box_nm is not None:
                cand = rng.uniform(0.15, 0.85, size=3) * (10.0 * np.array(box_nm))
            else:
                cand = origin + rng.normal(scale=8.0, size=3)
            if ca_pos and min(np.linalg.norm(cand - c) for c in ca_pos) < 4.0:
                continue
            ca_pos.append(cand)
        if len(ca_pos) < n_ca:
            raise SystemExit(f"no-DNA: placed {len(ca_pos)}/{n_ca} Ca")
        from openmm.app import Topology

        modeller = app.Modeller(Topology(), [])

    ox_cent = place_oxalate(ca_pos[:n_ox] if n_ox <= len(ca_pos) else ca_pos, rng)
    while len(ox_cent) < n_ox:
        ox_cent.append(np.asarray(ca_pos[-1], float) + rng.normal(scale=5.0, size=3))

    ca_pdb = OUT_DIR / f"{tag}_ca.pdb"
    oxl_pdb = OUT_DIR / f"{tag}_oxalate.pdb"
    write_ca_pdb(ca_pdb, ca_pos)
    write_oxl_pdb(oxl_pdb, ox_cent)
    ca_file = app.PDBFile(str(ca_pdb))
    oxl = app.PDBFile(str(oxl_pdb))
    add_oxl_bonds(oxl.topology)
    modeller.add(ca_file.topology, ca_file.positions)
    modeller.add(oxl.topology, oxl.positions)

    if box_nm is not None:
        box = openmm.Vec3(*box_nm) * unit.nanometer
        modeller.addSolvent(ff, model="tip3p", boxSize=box, neutralize=True)
    else:
        modeller.addSolvent(ff, model="tip3p", padding=pad_nm * unit.nanometer, neutralize=True)

    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
    )
    integrator = openmm.LangevinMiddleIntegrator(
        temperature * unit.kelvin, 1.0 / unit.picosecond, dt_fs * unit.femtosecond
    )
    if platform_name:
        platform = openmm.Platform.getPlatformByName(platform_name)
        sim = app.Simulation(modeller.topology, system, integrator, platform)
    else:
        sim = app.Simulation(modeller.topology, system, integrator)
    sim.context.setPositions(modeller.positions)
    print(f"{tag}: minimizing ({min_iters} iters)...", flush=True)
    sim.minimizeEnergy(maxIterations=min_iters)
    n_steps = max(0, int(round(ns * 1e6 / dt_fs))) if ns > 0 else 0
    out_pdb = OUT_DIR / f"{tag}_min.pdb"
    with out_pdb.open("w") as f:
        app.PDBFile.writeFile(
            sim.topology, sim.context.getState(getPositions=True).getPositions(), f
        )
    print(f"  wrote {out_pdb}  n_atoms={sim.topology.getNumAtoms()}", flush=True)

    if n_steps:
        dcd = OUT_DIR / f"{tag}.dcd"
        sim.reporters.append(app.DCDReporter(str(dcd), max(1, n_steps // 20)))
        print(f"{tag}: NVT {ns} ns ({n_steps} steps)...", flush=True)
        sim.step(n_steps)
        out_fin = OUT_DIR / f"{tag}_last.pdb"
        with out_fin.open("w") as f:
            app.PDBFile.writeFile(
                sim.topology,
                sim.context.getState(getPositions=True).getPositions(),
                f,
            )
        print(f"  wrote {out_fin}", flush=True)

    state = sim.context.getState(getPositions=True)
    pos = np.array(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    box_vec = sim.topology.getPeriodicBoxVectors()
    box_out = None
    if box_vec is not None:
        box_out = [float(box_vec[i][i].value_in_unit(unit.nanometer)) for i in range(3)]
    obs = observables_from_positions(tag, sim.topology, pos, len(pos))
    obs["box_nm"] = box_out
    return obs


def observables_from_pdb(path: Path, tag: str):
    from openmm import unit
    from openmm.app import PDBFile

    pdb = PDBFile(str(path))
    pos = xyz_ang(pdb.positions, unit)
    obs = observables_from_positions(tag, pdb.topology, pos, len(pos))
    box = pdb.topology.getPeriodicBoxVectors()
    if box is not None:
        obs["box_nm"] = [float(box[i][i].value_in_unit(unit.nanometer)) for i in range(3)]
    else:
        obs["box_nm"] = None
    return obs


def write_observables(results: list[dict], path: Path):
    lines = [
        "MD nucleation observables (OpenMM; DNA vs no-DNA)",
        "=" * 52,
        "",
        "Smoke / short NVT, not 50-200 ns. Ca and oxalate start in solution.",
        "Inner-sphere Ca-OP ~2.3-2.5 A; sequential strand Ca-Ca ~6.3 A if P templates a.",
        "",
    ]
    for r in results:
        lines.append(
            f"{r['tag']}: n_atoms={r['n_atoms']}  n_Ca={len(r['ca'])}  "
            f"n_P={len(r['p'])}  n_OP={len(r.get('op', []))}"
        )
        if r.get("box_nm"):
            lines.append(f"  box (nm): {r['box_nm'][0]:.2f} x {r['box_nm'][1]:.2f} x {r['box_nm'][2]:.2f}")
        if len(r["ca"]) >= 2:
            d = np.linalg.norm(r["ca"][:, None, :] - r["ca"][None, :, :], axis=2)
            iu = np.triu_indices(len(r["ca"]), 1)
            dd = d[iu]
            lines.append(
                f"  Ca-Ca: min={dd.min():.2f}  median={np.median(dd):.2f}  "
                f"n(3.84+/-0.25)={int(np.sum(np.abs(dd - 3.843) <= 0.25))}  "
                f"n(6.29+/-0.40)={int(np.sum(np.abs(dd - 6.29) <= 0.40))}"
            )
        if len(r.get("ca_op_min", [])):
            dp = r["ca_op_min"]
            lines.append(
                f"  Ca-OP min: median={np.median(dp):.2f}  "
                f"n(<2.7 A inner-sphere)={int((dp < 2.7).sum())}"
            )
        lines.append(
            f"  DNA-bound Ca: {r.get('n_bound', 0)}  "
            f"seq Ca-Ca n={len(r.get('seq_d', []))}  "
            f"seq 6.29={r.get('n_seq_629', 0)}  seq 3.84={r.get('n_seq_384', 0)}"
        )
        if len(r.get("seq_d", [])):
            lines.append(f"  seq Ca-Ca median={float(np.median(r['seq_d'])):.2f} A")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def plot_md(results: list[dict], path: Path):
    import matplotlib.pyplot as plt
    import matplotlib_config

    matplotlib_config.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax0, ax1 = axes
    for r in results:
        label = "DNA" if "dna" in r["tag"] and "nodna" not in r["tag"] else "no-DNA"
        if len(r.get("ca_op_min", [])):
            ax0.hist(r["ca_op_min"], bins=20, range=(1.8, 12), alpha=0.55, label=label)
        if len(r["ca"]) >= 2:
            d = np.linalg.norm(r["ca"][:, None, :] - r["ca"][None, :, :], axis=2)
            dd = d[np.triu_indices(len(r["ca"]), 1)]
            ax1.hist(dd, bins=24, range=(2.5, 20), alpha=0.55, label=label)
    ax0.axvline(2.45, color="0.3", ls="--", lw=1, label="inner-sphere")
    ax0.set_xlabel("min Ca–OP (Å)")
    ax0.set_ylabel("count")
    ax0.set_title("Ca–phosphate contact")
    ax0.legend()
    ax1.axvline(3.84, color="C3", ls="--", lw=1, label="3.84 Å")
    ax1.axvline(6.29, color="C2", ls="--", lw=1, label="6.29 Å")
    ax1.set_xlabel("Ca–Ca (Å)")
    ax1.set_ylabel("pairs")
    ax1.set_title("Ca–Ca distances")
    ax1.legend()
    fig.suptitle("MD smoke: DNA vs no-DNA (last frame)", fontsize=12)
    matplotlib_config.savefig(path)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description="DNA vs no-DNA CaOx nucleation MD (OpenMM)")
    ap.add_argument("--n-ca", type=int, default=8)
    ap.add_argument("--n-ox", type=int, default=8)
    ap.add_argument("--pad", type=float, default=0.8, help="solvent padding (nm) for the DNA arm")
    ap.add_argument("--temperature", type=float, default=310.0)
    ap.add_argument("--ns", type=float, default=0.005, help="NVT length (ns); 0 = minimize only")
    ap.add_argument("--dt", type=float, default=2.0, help="timestep fs")
    ap.add_argument("--platform", type=str, default=None)
    ap.add_argument("--min-iters", type=int, default=250)
    ap.add_argument("--skip-dna", action="store_true")
    ap.add_argument("--skip-nodna", action="store_true")
    ap.add_argument(
        "--from-last",
        action="store_true",
        help="Score existing md/md_*_last.pdb instead of running MD",
    )
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    if args.from_last:
        for tag in ("md_dna", "md_nodna"):
            last = OUT_DIR / f"{tag}_last.pdb"
            if not last.exists():
                last = OUT_DIR / f"{tag}_min.pdb"
            if not last.exists():
                print(f"skip {tag}: no PDB in {OUT_DIR}")
                continue
            if tag == "md_dna" and args.skip_dna:
                continue
            if tag == "md_nodna" and args.skip_nodna:
                continue
            results.append(observables_from_pdb(last, tag))
        if not results:
            raise SystemExit("No last-frame PDBs to score")
    else:
        box_nm = None
        if not args.skip_dna:
            dna = build_and_run(
                with_dna=True,
                n_ca=args.n_ca,
                n_ox=args.n_ox,
                pad_nm=args.pad,
                temperature=args.temperature,
                ns=args.ns,
                dt_fs=args.dt,
                platform_name=args.platform,
                tag="md_dna",
                min_iters=args.min_iters,
            )
            results.append(dna)
            box_nm = dna.get("box_nm")
        if not args.skip_nodna:
            results.append(
                build_and_run(
                    with_dna=False,
                    n_ca=args.n_ca,
                    n_ox=args.n_ox,
                    pad_nm=args.pad,
                    temperature=args.temperature,
                    ns=args.ns,
                    dt_fs=args.dt,
                    platform_name=args.platform,
                    tag="md_nodna",
                    box_nm=box_nm,
                    min_iters=args.min_iters,
                )
            )
    out_txt = FIG_DIR / "DNA_CaOx_md_observables.txt"
    write_observables(results, out_txt)
    plot_md(results, FIG_DIR / "DNA_CaOx_md_observables.png")


if __name__ == "__main__":
    main()
