#!/usr/bin/env python3
"""Intact CaOx chemical units: C2O4, Ca, and water.

Atom-wise lattice merges orphan carbons and fuse neighboring oxalates.
These helpers partition a crystal into whole molecules, merge by unit
centroid, and recover C2O4 for FIRE without union-finding all carbons.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

CC_MIN, CC_MAX = 1.40, 1.66
CO_MIN, CO_MAX = 1.18, 1.45
OH_MAX = 1.25


def atom_el(a) -> str:
    e = str(a.get("element", "")).strip().upper()
    n = str(a.get("name", "")).strip().upper()
    if e.startswith("CA") or n == "CA":
        return "CA"
    if e.startswith("C") or n.startswith("C"):
        return "C"
    if e.startswith("O") or n.startswith("O"):
        return "O"
    if e.startswith("H") or n.startswith("H"):
        return "H"
    return e or n[:1]


def _indices(els, kind):
    return [i for i, e in enumerate(els) if e == kind]


def pair_carbons(xyz: np.ndarray, c_idx: list[int]):
    """Mutual nearest C–C pairs inside the oxalate C–C window."""
    if len(c_idx) < 2:
        return []
    cxyz = xyz[c_idx]
    _d, j = cKDTree(cxyz).query(cxyz, k=min(2, len(c_idx)))
    if len(c_idx) == 1:
        return []
    nn = _d[:, 1]
    nj = j[:, 1]
    used = set()
    pairs = []
    for i in np.argsort(nn):
        i = int(i)
        if i in used:
            continue
        if not (CC_MIN <= float(nn[i]) <= CC_MAX):
            continue
        k = int(nj[i])
        if k in used:
            continue
        dk = float(np.linalg.norm(cxyz[i] - cxyz[k]))
        if not (CC_MIN <= dk <= CC_MAX):
            continue
        pairs.append((c_idx[i], c_idx[k]))
        used.add(i)
        used.add(k)
    return pairs


def partition_units(atoms: list[dict]):
    """
    Split atoms into complete oxalate (2C+4O), Ca, and water units.

    Incomplete C2 fragments are omitted (never emitted as partial sticks).
    Returns (oxalates, cas, waters) as lists of index lists into `atoms`.
    """
    if not atoms:
        return [], [], []
    els = [atom_el(a) for a in atoms]
    xyz = np.array([a["xyz"] for a in atoms], float)
    c_idx = _indices(els, "C")
    o_idx = _indices(els, "O")
    ca_idx = _indices(els, "CA")
    h_idx = _indices(els, "H")

    pairs = pair_carbons(xyz, c_idx)
    used_o: set[int] = set()
    oxalates: list[list[int]] = []
    if o_idx and pairs:
        oxyz = xyz[o_idx]
        otree = cKDTree(oxyz)
        for c1, c2 in pairs:
            oxy = []
            for ci in (c1, c2):
                cand = []
                for h in otree.query_ball_point(xyz[ci], CO_MAX):
                    oi = o_idx[int(h)]
                    if oi in used_o:
                        continue
                    d = float(np.linalg.norm(xyz[ci] - xyz[oi]))
                    if CO_MIN <= d <= CO_MAX:
                        cand.append((d, oi))
                cand.sort()
                for _, oi in cand[:2]:
                    oxy.append(oi)
                    used_o.add(oi)
            if len(oxy) == 4:
                oxalates.append([c1, c2, *oxy])
            else:
                for oi in oxy:
                    used_o.discard(oi)

    ox_atoms = {i for g in oxalates for i in g}
    waters: list[list[int]] = []
    used_h: set[int] = set()
    for oi in o_idx:
        if oi in ox_atoms:
            continue
        grp = [oi]
        for hi in h_idx:
            if hi in used_h:
                continue
            if float(np.linalg.norm(xyz[hi] - xyz[oi])) <= OH_MAX:
                grp.append(hi)
                used_h.add(hi)
        waters.append(grp)
    cas = [[i] for i in ca_idx]
    return oxalates, cas, waters


def units_from_pbc(primary: list[dict], expanded: list[dict], av, bv, cv) -> list[dict]:
    """Complete C2O4 that straddle cell faces, then unique-copy into one cell."""
    units = units_from_atoms(expanded)
    M = np.linalg.inv(np.column_stack((av, bv, cv)))
    uniq: dict[tuple, dict] = {}
    for u in units:
        if u["kind"] == "water":
            continue
        frac = np.mod(M @ u["com"], 1.0)
        key = (u["kind"],) + tuple(np.round(frac.astype(float), 3))
        uniq.setdefault(key, u)
    ox_ca = list(uniq.values())
    used_o = set()
    for u in ox_ca:
        for a in u["atoms"]:
            if atom_el(a) == "O":
                used_o.add(tuple(np.round(np.mod(M @ a["xyz"], 1.0), 3)))
    waters = []
    for a in primary:
        if atom_el(a) != "O":
            continue
        key = tuple(np.round(np.mod(M @ a["xyz"], 1.0), 3))
        if key in used_o:
            continue
        recs = [dict(a)]
        xyz = np.array([r["xyz"] for r in recs], float)
        waters.append(
            {"kind": "water", "atoms": recs, "xyz": xyz, "com": xyz.mean(axis=0)}
        )
    return ox_ca + waters


def units_from_atoms(atoms: list[dict]) -> list[dict]:
    ox, cas, wat = partition_units(atoms)
    out = []
    for kind, groups in (("oxalate", ox), ("ca", cas), ("water", wat)):
        for idx in groups:
            xyz = np.array([atoms[i]["xyz"] for i in idx], float)
            recs = [dict(atoms[i]) for i in idx]
            out.append(
                {
                    "kind": kind,
                    "atoms": recs,
                    "xyz": xyz,
                    "com": xyz.mean(axis=0),
                }
            )
    return out


def transform_unit(unit: dict, R: np.ndarray, ref: np.ndarray, origin: np.ndarray) -> dict:
    xyz = (unit["xyz"] - ref) @ R.T + origin
    atoms = []
    for a, p in zip(unit["atoms"], xyz):
        b = dict(a)
        b["xyz"] = p
        atoms.append(b)
    return {"kind": unit["kind"], "atoms": atoms, "xyz": xyz, "com": xyz.mean(axis=0)}


def expand_units(units: list[dict], av, bv, cv, nmax: int) -> list[dict]:
    out = []
    for ia in range(-nmax, nmax + 1):
        for ib in range(-nmax, nmax + 1):
            for ic in range(-nmax, nmax + 1):
                shift = ia * av + ib * bv + ic * cv
                for u in units:
                    xyz = u["xyz"] + shift
                    atoms = []
                    for a, p in zip(u["atoms"], xyz):
                        b = dict(a)
                        b["xyz"] = p
                        atoms.append(b)
                    out.append(
                        {
                            "kind": u["kind"],
                            "atoms": atoms,
                            "xyz": xyz,
                            "com": xyz.mean(axis=0),
                            "cell": (ia, ib, ic),
                        }
                    )
    return out


def merge_units(items: list[dict], cutoff: float) -> list[dict]:
    """Keep the earliest source when unit centroids fall within cutoff."""
    if not items:
        return []
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for u in items:
        by_kind[u["kind"]].append(u)
    kept: list[dict] = []
    for kind, group in by_kind.items():
        order = sorted(
            range(len(group)),
            key=lambda i: (group[i].get("source", 0), i),
        )
        xyz = np.array([group[i]["com"] for i in order], float)
        parent = list(range(len(order)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        if len(order) >= 2:
            for a, b in cKDTree(xyz).query_pairs(cutoff):
                ia, ib = find(int(a)), find(int(b))
                if ia != ib:
                    parent[ib] = ia
        buckets = defaultdict(list)
        for i in range(len(order)):
            buckets[find(i)].append(i)
        for members in buckets.values():
            members.sort()
            rec = dict(group[order[members[0]]])
            rec["n_sources"] = len(members)
            kept.append(rec)
    return kept


def drop_close_ca_units(units: list[dict], min_ca: float) -> list[dict]:
    cas = [u for u in units if u["kind"] == "ca"]
    other = [u for u in units if u["kind"] != "ca"]
    cas.sort(key=lambda u: u.get("source", 0))
    keep = []
    for u in cas:
        p = u["com"]
        if keep and float(np.linalg.norm(np.array([c["com"] for c in keep]) - p, axis=1).min()) < min_ca:
            continue
        keep.append(u)
    return keep + other


def drop_clashing_oxalate_units(units: list[dict], min_oo: float) -> list[dict]:
    """Drop the later-source oxalate when any inter-unit O···O is below min_oo."""
    ox = [u for u in units if u["kind"] == "oxalate"]
    rest = [u for u in units if u["kind"] != "oxalate"]
    ox.sort(key=lambda u: u.get("source", 0))
    if len(ox) < 2:
        return units
    oxyz = []
    owners = []
    for i, u in enumerate(ox):
        for a, p in zip(u["atoms"], u["xyz"]):
            if atom_el(a) != "O":
                continue
            oxyz.append(p)
            owners.append(i)
    oxyz = np.array(oxyz, float)
    drop = set()
    for a, b in cKDTree(oxyz).query_pairs(min_oo):
        ia, ib = owners[int(a)], owners[int(b)]
        if ia == ib:
            continue
        drop.add(max(ia, ib))
    keep_ox = [u for i, u in enumerate(ox) if i not in drop]
    return keep_ox + rest


def drop_clashing_water_units(units: list[dict], min_oo: float) -> list[dict]:
    waters = [u for u in units if u["kind"] == "water"]
    rest = [u for u in units if u["kind"] != "water"]
    if not waters:
        return units
    exist = []
    for u in rest:
        for a, p in zip(u["atoms"], u["xyz"]):
            if atom_el(a) == "O":
                exist.append(p)
    tree = cKDTree(np.array(exist, float)) if exist else None
    waters.sort(key=lambda u: u.get("source", 0))
    kept_w = []
    kept_o = []
    for u in waters:
        o = np.array(
            [p for a, p in zip(u["atoms"], u["xyz"]) if atom_el(a) == "O"],
            float,
        )
        if len(o) == 0:
            continue
        clash = False
        if tree is not None:
            d, _ = tree.query(o, k=1)
            if float(np.min(d)) < min_oo:
                clash = True
        if not clash and kept_o:
            d, _ = cKDTree(np.array(kept_o)).query(o, k=1)
            if float(np.min(d)) < min_oo:
                clash = True
        if clash:
            continue
        kept_w.append(u)
        kept_o.extend(o)
    return rest + kept_w


def flatten_units(units: list[dict]) -> list[dict]:
    atoms = []
    for u in units:
        for a in u["atoms"]:
            b = dict(a)
            b["unit_kind"] = u["kind"]
            b["n_sources"] = u.get("n_sources", 1)
            b["source"] = u.get("source", 0)
            atoms.append(b)
    return atoms


def assign_unit_residues(units: list[dict], shell_bfacs: dict[int, float] | None = None):
    """One WHW residue per chemical unit. Ca first so symmetry still finds WHW Ca."""
    cas = [u for u in units if u["kind"] == "ca"]
    oxs = [u for u in units if u["kind"] == "oxalate"]
    wats = [u for u in units if u["kind"] == "water"]
    cas.sort(key=lambda u: float(np.linalg.norm(u["com"])))
    out = []
    ca_xyz = np.array([u["com"] for u in cas], float) if cas else np.zeros((0, 3))
    tree = cKDTree(ca_xyz) if len(ca_xyz) else None

    def bfac_for(com):
        if not shell_bfacs or tree is None or len(ca_xyz) == 0:
            return 20.0
        _, j = tree.query(com, k=1)
        ca_res = int(j) + 1
        return float(shell_bfacs.get(ca_res, 20.0))

    def emit(u, chain, res, name_of):
        bf = bfac_for(u["com"])
        recs = []
        for k, a in enumerate(u["atoms"]):
            if atom_el(a) == "H":
                continue
            b = dict(a)
            b["resname"] = "WHW"
            b["chain"] = chain
            b["resseq"] = res
            b["name"], b["element"] = name_of(k, a)
            b["bfac"] = bf
            b["n_sources"] = u.get("n_sources", 1)
            recs.append(b)
        return recs

    for res, u in enumerate(cas, start=1):
        out.extend(
            emit(u, "Z", res, lambda k, a: ("CA", "CA"))
        )
    ox_names = ("C1", "C2", "O1", "O2", "O3", "O4")
    for res, u in enumerate(oxs, start=1):
        els = [atom_el(a) for a in u["atoms"]]
        ordered = [u["atoms"][i] for i in (
            [j for j, e in enumerate(els) if e == "C"]
            + [j for j, e in enumerate(els) if e == "O"]
        )]
        u_ord = dict(u)
        u_ord["atoms"] = ordered
        out.extend(
            emit(
                u_ord,
                "X",
                res,
                lambda k, a: (
                    ox_names[k] if k < len(ox_names) else atom_el(a),
                    atom_el(a),
                ),
            )
        )
    for res, u in enumerate(wats, start=1):
        out.extend(
            emit(u, "W", res, lambda k, a: ("OW", "O") if atom_el(a) == "O" else (a["name"], atom_el(a)))
        )
    return out


def find_oxalates_strict(atoms: list[dict]):
    """
    Recover intact C2O4 for FIRE.

    Prefer WHW residues that already are 2C+4O. Remaining atoms are
    partitioned with exclusive carbon pairing (no 1.72 Å C union-find).
    """
    ca_idx = [i for i, a in enumerate(atoms) if atom_el(a) == "CA"]
    by_res: dict[tuple, list[int]] = defaultdict(list)
    for i, a in enumerate(atoms):
        by_res[(a.get("chain", "Z"), int(a.get("resseq", 0)))].append(i)

    oxalates: list[list[int]] = []
    used = set()
    for idx in by_res.values():
        els = [atom_el(atoms[i]) for i in idx]
        n_c = sum(e == "C" for e in els)
        n_o = sum(e == "O" for e in els)
        if n_c == 2 and n_o == 4:
            oxalates.append(list(idx))
            used.update(idx)

    leftover = [a for i, a in enumerate(atoms) if i not in used]
    leftover_i = [i for i in range(len(atoms)) if i not in used]
    if leftover:
        ox2, _ca, _w = partition_units(leftover)
        for g in ox2:
            oxalates.append([leftover_i[j] for j in g])
            used.update(leftover_i[j] for j in g)

    water = [
        i
        for i, a in enumerate(atoms)
        if i not in used and atom_el(a) == "O"
    ]
    return oxalates, ca_idx, water


def oxalate_quality(atoms: list[dict]) -> dict:
    ox, ca_idx, water = find_oxalates_strict(atoms)
    xyz = np.array([a["xyz"] for a in atoms], float)
    n_ok = 0
    cc, co = [], []
    for g in ox:
        carbons = [i for i in g if atom_el(atoms[i]) == "C"]
        oxygens = [i for i in g if atom_el(atoms[i]) == "O"]
        if len(carbons) == 2 and len(oxygens) == 4:
            n_ok += 1
            cc.append(float(np.linalg.norm(xyz[carbons[0]] - xyz[carbons[1]])))
            for ci in carbons:
                ds = sorted(float(np.linalg.norm(xyz[ci] - xyz[oi])) for oi in oxygens)[:2]
                co.extend(ds)
    n_c = sum(atom_el(a) == "C" for a in atoms)
    return {
        "n_oxalate": len(ox),
        "n_intact": n_ok,
        "n_ca": len(ca_idx),
        "n_water": len(water),
        "n_carbon": n_c,
        "orphan_c": n_c - 2 * n_ok,
        "cc_median": float(np.median(cc)) if cc else None,
        "co_median": float(np.median(co)) if co else None,
        "cc_minmax": (min(cc), max(cc)) if cc else None,
        "co_minmax": (min(co), max(co)) if co else None,
    }
