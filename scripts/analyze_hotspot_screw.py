#!/usr/bin/env python3
"""
Hotspot cluster analysis: local screw / glide, not an extended COM lattice.

The templating gel does not have enough Ca for a 30 nm crystal. This script
asks whether hotspot Ca (and phosphate-bound seed Ca) already carry the DNA
screw, a 2_1 (COM P2_1/n), or a glide along the helix axis.

  .venv/bin/python scripts/analyze_hotspot_screw.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib_config  # noqa: E402
from find_symmetry import (  # noqa: E402
    HOTSPOT_CLUSTER_CUT,
    HOTSPOT_MIN_SIZE,
    cluster_local,
    load_dna_heavy,
    load_phosphate_xyz,
    load_whw_ca,
)
from helix_frame import basis_from_dna, to_helix  # noqa: E402
from strand_order import load_phosphates_labeled, strand_groups  # noqa: E402

PDB = ROOT / "DNA_CaOx_templating_gel_omm.pdb"
CSV = ROOT / "figures/crystallinity/DNA_CaOx_templating_gel_omm_ca_metrics.csv"
OUT_PNG = ROOT / "figures/crystallinity/DNA_CaOx_templating_hotspot_screw.png"
OUT_TXT = ROOT / "figures/crystallinity/DNA_CaOx_templating_hotspot_screw.txt"
OUT_JSON = ROOT / "figures/crystallinity/DNA_CaOx_templating_hotspot_screw.json"

COM_384 = 3.843
COM_A = 6.290
DNA_TWIST = 36.0  # deg / bp, textbook B-DNA
DNA_RISE = 3.38  # Å / bp
MAP_TOL = 1.20  # Å — local assembly, not a crystal map
N_SHUFFLE = 400
RNG = np.random.default_rng(2026)


def rotation_about_axis(axis: np.ndarray, ang: float) -> np.ndarray:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
    c, s = math.cos(ang), math.sin(ang)
    x, y, z = axis
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )


def apply_screw(pts, origin, axis, ang_rad, rise):
    R = rotation_about_axis(axis, ang_rad)
    return origin + (pts - origin) @ R.T + rise * axis


def apply_21(pts, origin, axis, period):
    return apply_screw(pts, origin, axis, math.pi, 0.5 * period)


def apply_glide(pts, origin, axis, n_plane, trans):
    """Mirror through a plane that contains `axis`, then translate along it."""
    n = n_plane - axis * float(n_plane @ axis)
    n = n / max(float(np.linalg.norm(n)), 1e-8)
    rel = pts - origin
    img = pts - 2.0 * np.outer(rel @ n, n)
    return img + trans * axis


def map_hits(src, dest, tol=MAP_TOL, exclude_self=True):
    """Fraction of src images that land on some dest point (not the same index)."""
    if len(src) == 0 or len(dest) == 0:
        return 0.0, 0
    d = np.linalg.norm(src[:, None, :] - dest[None, :, :], axis=2)
    if exclude_self and len(src) == len(dest):
        np.fill_diagonal(d, 1e9)
    mind = d.min(axis=1)
    n = int((mind <= tol).sum())
    return n / len(src), n


def helix_cyl(pts, origin, axis, e_x, e_y):
    hx = np.array([to_helix(p, origin, e_x, axis, e_y) for p in pts])
    r = np.hypot(hx[:, 0], hx[:, 2])
    phi = np.degrees(np.arctan2(hx[:, 2], hx[:, 0]))
    z = hx[:, 1]
    return r, phi, z, hx


def wrap_abs(dphi):
    d = np.abs(dphi) % 360.0
    return np.minimum(d, 360.0 - d)


def phosphate_screw(pdb: Path, origin, axis, e_x, e_y):
    phos = load_phosphates_labeled(pdb)
    if len(phos) < 4:
        return {"twist": DNA_TWIST, "rise": DNA_RISE, "pp": 6.7, "n_step": 0}
    pxyz = np.array([p["xyz"] for p in phos])
    r, phi, z, _ = helix_cyl(pxyz, origin, axis, e_x, e_y)
    twists, rises, pps = [], [], []
    for pix in strand_groups(phos):
        for a, b in zip(pix, pix[1:]):
            dphi = wrap_abs(phi[b] - phi[a])
            dz = abs(float(z[b] - z[a]))
            if 2.2 < dz < 5.2 and 20 < dphi < 50:
                twists.append(dphi)
                rises.append(dz)
                pps.append(float(np.linalg.norm(pxyz[a] - pxyz[b])))
    if not twists:
        return {"twist": DNA_TWIST, "rise": DNA_RISE, "pp": 6.7, "n_step": 0}
    return {
        "twist": float(np.median(twists)),
        "rise": float(np.median(rises)),
        "pp": float(np.median(pps)),
        "n_step": int(len(twists)),
        "twist_mean": float(np.mean(twists)),
        "rise_mean": float(np.mean(rises)),
    }


def pair_table(r, phi, z, pts):
    n = len(pts)
    rows = []
    if n < 2:
        return rows
    for i in range(n):
        for j in range(i + 1, n):
            dphi = float(wrap_abs(phi[j] - phi[i]))
            dz = float(abs(z[j] - z[i]))
            dr = float(abs(r[j] - r[i]))
            d = float(np.linalg.norm(pts[i] - pts[j]))
            rows.append({"i": i, "j": j, "dphi": dphi, "dz": dz, "dr": dr, "d": d})
    return rows


def classify_pairs(pairs, twist, rise):
    """Count pairs matching DNA screw steps, 2_1, or same-height dyad."""
    n = {"dna1": 0, "dna2": 0, "screw21": 0, "dyad": 0, "com384": 0, "com629": 0, "cyl": 0}
    for p in pairs:
        cyl = p["dr"] < 2.2
        if cyl:
            n["cyl"] += 1
        if cyl and abs(p["dphi"] - twist) < 12 and abs(p["dz"] - rise) < 1.15:
            n["dna1"] += 1
        if cyl and abs(p["dphi"] - 2 * twist) < 16 and abs(p["dz"] - 2 * rise) < 1.6:
            n["dna2"] += 1
        if cyl and abs(p["dphi"] - 180) < 22 and abs(p["dz"] - 0.5 * COM_A) < 1.15:
            n["screw21"] += 1
        if cyl and abs(p["dphi"] - 180) < 25 and p["dz"] < 1.2:
            n["dyad"] += 1
        if abs(p["d"] - COM_384) <= 0.25:
            n["com384"] += 1
        if abs(p["d"] - COM_A) <= 0.40:
            n["com629"] += 1
    n["n_pairs"] = len(pairs)
    return n


def shuffle_phi_counts(r, phi, z, pts, twist, rise, n_shuf=N_SHUFFLE):
    """Keep (r, z); randomize azimuth — destroys screw/glide, keeps radial density."""
    acc = {k: [] for k in ("dna1", "dna2", "screw21", "dyad")}
    for _ in range(n_shuf):
        phi_s = RNG.uniform(-180, 180, size=len(phi))
        # rebuild cartesian in helix frame for distances that use pts? classify uses dphi/dz/dr and d from pts.
        # com384/629 use Euclidean d — shuffling φ changes d. Rebuild pts in helix frame.
        ang = np.radians(phi_s)
        hx = np.stack([r * np.cos(ang), z, r * np.sin(ang)], axis=1)
        # pair_table only needs r, phi, z and pts for Euclidean d
        dummy = hx  # distances in helix cartesian ≈ lab if orthonormal
        pairs = pair_table(r, phi_s, z, dummy)
        c = classify_pairs(pairs, twist, rise)
        for k in acc:
            acc[k].append(c[k])
    return {k: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95))} for k, v in acc.items()}


def operator_scores(pts, origin, axis, e_x, e_y, twist, rise):
    out = []
    specs = [
        ("DNA screw 1-step", math.radians(twist), rise, "screw"),
        ("DNA screw 2-step", math.radians(2 * twist), 2 * rise, "screw"),
        ("textbook 36° / 3.38 Å", math.radians(DNA_TWIST), DNA_RISE, "screw"),
        ("2_1 (period 6.29 Å)", math.pi, 0.5 * COM_A, "21"),
        ("2_1 (period 2×P rise)", math.pi, rise, "21"),
    ]
    for name, ang, h, kind in specs:
        if kind == "21":
            img = apply_21(pts, origin, axis, 2 * h)
        else:
            img = apply_screw(pts, origin, axis, ang, h)
        frac, n = map_hits(img, pts)
        out.append({"name": name, "frac": frac, "n": n, "kind": kind})

    best_glide = {"name": "glide (best plane)", "frac": 0.0, "n": 0, "kind": "glide", "az": None, "t": None}
    for az in np.linspace(0, 180, 18, endpoint=False):
        n_plane = math.cos(math.radians(az)) * e_x + math.sin(math.radians(az)) * e_y
        for t in (rise, 2 * rise, 0.5 * COM_A, COM_A):
            img = apply_glide(pts, origin, axis, n_plane, t)
            frac, n = map_hits(img, pts)
            if frac > best_glide["frac"]:
                best_glide = {
                    "name": f"glide az={az:.0f}° t={t:.2f} Å",
                    "frac": frac,
                    "n": n,
                    "kind": "glide",
                    "az": float(az),
                    "t": float(t),
                }
    out.append(best_glide)

    best_dyad = {"name": "C2 perp. helix (best)", "frac": 0.0, "n": 0, "kind": "dyad", "az": None}
    for az in np.linspace(0, 180, 18, endpoint=False):
        dyad = math.cos(math.radians(az)) * e_x + math.sin(math.radians(az)) * e_y
        img = apply_screw(pts, origin, dyad, math.pi, 0.0)
        frac, n = map_hits(img, pts)
        if frac > best_dyad["frac"]:
            best_dyad = {
                "name": f"C2 perp az={az:.0f}°",
                "frac": frac,
                "n": n,
                "kind": "dyad",
                "az": float(az),
            }
    out.append(best_dyad)
    return out


def load_metrics(path: Path):
    rows = []
    with path.open() as f:
        for rec in csv.DictReader(f):
            rec["hotspot"] = int(float(rec.get("hotspot") or 0))
            rec["p_bound"] = int(float(rec.get("p_bound") or 0))
            rec["d_p_A"] = float(rec["d_p_A"])
            rec["pair_corr"] = float(rec.get("pair_corr") or 0)
            rec["axial_A"] = float(rec.get("axial_A") or 0)
            rows.append(rec)
    return rows


def cluster_nn(pts_c):
    if len(pts_c) < 2:
        return None, None
    d = np.linalg.norm(pts_c[:, None] - pts_c[None, :], axis=2)
    np.fill_diagonal(d, 1e9)
    nn = d.min(axis=1)
    return float(np.median(nn)), float(nn.min())


def main():
    if not PDB.exists() or not CSV.exists():
        raise SystemExit(f"Missing {PDB.name} or metrics CSV")
    ca, _bfac = load_whw_ca(PDB)
    rows = load_metrics(CSV)
    if len(rows) != len(ca):
        raise SystemExit(f"CSV/PDB Ca mismatch: {len(rows)} vs {len(ca)}")
    hot = np.array([r["hotspot"] for r in rows], bool)
    seed = np.array([r["p_bound"] for r in rows], bool)
    dna = load_dna_heavy(PDB)
    origin, axis, e_x, e_y = basis_from_dna(dna, ca[seed] if seed.any() else ca)
    p_screw = phosphate_screw(PDB, origin, axis, e_x, e_y)
    twist, rise = p_screw["twist"], p_screw["rise"]

    hot_ix = np.where(hot)[0]
    seed_ix = np.where(seed)[0]
    clusters = []
    if len(hot_ix) >= HOTSPOT_MIN_SIZE:
        for cl in cluster_local(ca[hot_ix], cutoff=HOTSPOT_CLUSTER_CUT, min_size=HOTSPOT_MIN_SIZE):
            glob = hot_ix[cl]
            clusters.append(glob)

    results = {
        "n_ca": int(len(ca)),
        "n_hot": int(hot.sum()),
        "n_seed": int(seed.sum()),
        "n_cluster": len(clusters),
        "phosphate": p_screw,
        "map_tol_A": MAP_TOL,
    }

    sets = {
        "hotspots": hot_ix,
        "P-bound seeds": seed_ix,
        "hot ∩ seed": np.where(hot & seed)[0],
    }
    set_scores = {}
    set_pairs = {}
    set_null = {}
    for name, ix in sets.items():
        pts = ca[ix]
        if len(pts) < 3:
            continue
        r, phi, z, _ = helix_cyl(pts, origin, axis, e_x, e_y)
        pairs = pair_table(r, phi, z, pts)
        counts = classify_pairs(pairs, twist, rise)
        null = shuffle_phi_counts(r, phi, z, pts, twist, rise)
        ops = operator_scores(pts, origin, axis, e_x, e_y, twist, rise)
        set_scores[name] = ops
        set_pairs[name] = counts
        set_null[name] = null

    cluster_rows = []
    for ci, glob in enumerate(clusters, start=1):
        pts = ca[glob]
        r, phi, z, hx = helix_cyl(pts, origin, axis, e_x, e_y)
        pairs = pair_table(r, phi, z, pts)
        counts = classify_pairs(pairs, twist, rise)
        ops = operator_scores(pts, origin, axis, e_x, e_y, twist, rise)
        nn_med, nn_min = cluster_nn(pts)
        best = max(ops, key=lambda o: o["frac"])
        cluster_rows.append(
            {
                "id": ci,
                "n": int(len(glob)),
                "mean_dp": float(np.mean([rows[i]["d_p_A"] for i in glob])),
                "mean_z": float(np.mean(z)),
                "mean_r": float(np.mean(r)),
                "phi_span": float(wrap_abs(phi.max() - phi.min()) if len(phi) else 0),
                "z_span": float(z.max() - z.min()) if len(z) else 0,
                "nn_med": nn_med,
                "nn_min": nn_min,
                "dna1": counts["dna1"],
                "dna2": counts["dna2"],
                "screw21": counts["screw21"],
                "dyad": counts["dyad"],
                "com384": counts["com384"],
                "com629": counts["com629"],
                "best_op": best["name"],
                "best_frac": best["frac"],
                "best_n": best["n"],
                "ops": ops,
            }
        )

    results["sets"] = {
        k: {"ops": set_scores.get(k, []), "pairs": set_pairs.get(k, {}), "null": set_null.get(k, {})}
        for k in sets
        if k in set_pairs
    }
    results["clusters"] = [{k: v for k, v in c.items() if k != "ops"} | {"best_op": c["best_op"]} for c in cluster_rows]
    # keep ops in json too
    results["clusters"] = cluster_rows

    lines = [
        "Hotspot screw / glide analysis — templating gel",
        "=" * 58,
        "Not a crystal search. Small clusters; DNA helix is itself a screw.",
        f"Model: {PDB.name}  n_Ca={len(ca)}  hotspots={int(hot.sum())}  P-bound seeds={int(seed.sum())}",
        f"Map tolerance {MAP_TOL:.2f} A (local, not 0.25 A lattice).",
        "",
        "DNA phosphate screw (measured sequential P along strands):",
        f"  n_steps={p_screw['n_step']}  twist={twist:.1f} deg  rise={rise:.2f} A  "
        f"P-P={p_screw['pp']:.2f} A",
        f"  textbook B-DNA: {DNA_TWIST:.0f} deg / {DNA_RISE:.2f} A",
        "",
    ]
    for name, payload in results["sets"].items():
        c = payload["pairs"]
        nll = payload["null"]
        lines.append(f"--- {name}  (n={len(sets[name])}, pairs={c.get('n_pairs', 0)}) ---")
        lines.append(
            f"  pair motifs: DNA 1-step {c.get('dna1', 0)} "
            f"(null {nll.get('dna1', {}).get('mean', 0):.1f}, p95 {nll.get('dna1', {}).get('p95', 0):.0f})"
        )
        lines.append(
            f"             DNA 2-step {c.get('dna2', 0)} "
            f"(null {nll.get('dna2', {}).get('mean', 0):.1f}, p95 {nll.get('dna2', {}).get('p95', 0):.0f})"
        )
        lines.append(
            f"             2_1 (180 deg, 3.15 A) {c.get('screw21', 0)} "
            f"(null {nll.get('screw21', {}).get('mean', 0):.1f}, p95 {nll.get('screw21', {}).get('p95', 0):.0f})"
        )
        lines.append(
            f"             C2 dyad (180 deg, dz~0) {c.get('dyad', 0)} "
            f"(null {nll.get('dyad', {}).get('mean', 0):.1f}, p95 {nll.get('dyad', {}).get('p95', 0):.0f})"
        )
        lines.append(
            f"  Ca-Ca distances: 3.84 A pairs={c.get('com384', 0)}  6.29 A pairs={c.get('com629', 0)}"
        )
        lines.append(f"  operators (fraction of Ca mapped within {MAP_TOL:.2f} A):")
        for op in payload["ops"]:
            lines.append(f"    {op['frac']:.2f}  n={op['n']:3d}  {op['name']}")
        lines.append("")

    lines.append(f"Spatial hotspot clusters (link cutoff {HOTSPOT_CLUSTER_CUT:.1f} A, min {HOTSPOT_MIN_SIZE}): {len(cluster_rows)}")
    for c in cluster_rows:
        lines.append(
            f"  cl {c['id']:2d}  n={c['n']:2d}  d(P)={c['mean_dp']:.1f}  "
            f"r={c['mean_r']:.1f}  z={c['mean_z']:+.1f}  span z={c['z_span']:.1f} A  "
            f"nn={c['nn_min']:.2f}/{c['nn_med']:.2f}  "
            f"DNA1={c['dna1']} DNA2={c['dna2']} 2_1={c['screw21']} dyad={c['dyad']}  "
            f"3.84={c['com384']} 6.29={c['com629']}"
        )
        lines.append(f"       best operator: {c['best_frac']:.2f}  {c['best_op']}")
    lines.append("")
    lines.append("Reading: a motif count above the azimuth-shuffle p95 is evidence")
    lines.append("the DNA screw (or 2_1 / glide) is organizing Ca, not just packing.")
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(OUT_TXT.read_text())

    def _json(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    OUT_JSON.write_text(json.dumps(results, indent=2, default=_json))

    # Figure
    matplotlib_config.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.4))
    ax_un, ax_dz, ax_bar, ax_cl = axes.ravel()

    pxyz = load_phosphate_xyz(PDB)
    if len(pxyz):
        pr, pphi, pz, _ = helix_cyl(pxyz, origin, axis, e_x, e_y)
        ax_un.scatter(pphi, pz, s=18, c="0.55", marker="x", label="P", zorder=2)
    hr, hphi, hz, _ = helix_cyl(ca[hot_ix], origin, axis, e_x, e_y) if len(hot_ix) else (None, None, None, None)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(clusters), 1)))
    assigned = np.full(len(hot_ix), -1)
    for ci, glob in enumerate(clusters):
        for g in glob:
            w = np.where(hot_ix == g)[0]
            if len(w):
                assigned[w[0]] = ci
        ax_un.scatter(
            hphi[assigned == ci],
            hz[assigned == ci],
            s=36,
            color=colors[ci],
            label=f"cl {ci+1} (n={len(glob)})",
            zorder=3,
        )
    loners = assigned < 0
    if hr is not None and loners.any():
        ax_un.scatter(hphi[loners], hz[loners], s=22, c="0.2", label="unclustered hot", zorder=3)
    ax_un.set_xlabel("azimuth φ (deg)")
    ax_un.set_ylabel("helix z (Å)")
    ax_un.set_title("Hotspot clusters in helical unroll")
    ax_un.legend(fontsize=7, loc="best", ncol=2)

    if hr is not None:
        pairs = pair_table(hr, hphi, hz, ca[hot_ix])
        ax_dz.scatter(
            [p["dphi"] for p in pairs],
            [p["dz"] for p in pairs],
            s=12,
            alpha=0.55,
            c="0.25",
            label="hotspot pairs",
        )
    ax_dz.scatter([twist], [rise], s=80, marker="D", c="C3", label=f"DNA 1-step ({twist:.0f}°, {rise:.1f} Å)", zorder=4)
    ax_dz.scatter([2 * twist], [2 * rise], s=70, marker="D", c="C1", label="DNA 2-step", zorder=4)
    ax_dz.scatter([180], [0.5 * COM_A], s=80, marker="s", c="C2", label="2₁ (180°, 3.15 Å)", zorder=4)
    ax_dz.set_xlabel("|Δφ| (deg)")
    ax_dz.set_ylabel("|Δz| (Å)")
    ax_dz.set_xlim(0, 185)
    ax_dz.set_ylim(0, 14)
    ax_dz.set_title("Hotspot-pair screw coordinates")
    ax_dz.legend(fontsize=7)

    if "hotspots" in set_scores:
        names = [o["name"][:22] for o in set_scores["hotspots"]]
        fr = [100 * o["frac"] for o in set_scores["hotspots"]]
        ax_bar.barh(range(len(names)), fr, color="C0")
        ax_bar.set_yticks(range(len(names)), names, fontsize=8)
        ax_bar.set_xlabel("% of hotspot Ca mapped (tol 1.2 Å)")
        ax_bar.set_title("Operators on all hotspots")
        ax_bar.set_xlim(0, max(fr + [5]) * 1.15)

    if cluster_rows:
        ax_cl.bar(
            [c["id"] for c in cluster_rows],
            [c["dna1"] for c in cluster_rows],
            label="DNA 1-step pairs",
            color="C3",
        )
        ax_cl.bar(
            [c["id"] for c in cluster_rows],
            [c["screw21"] for c in cluster_rows],
            bottom=[c["dna1"] for c in cluster_rows],
            label="2₁ pairs",
            color="C2",
        )
        ax_cl.bar(
            [c["id"] for c in cluster_rows],
            [c["com384"] for c in cluster_rows],
            bottom=[c["dna1"] + c["screw21"] for c in cluster_rows],
            label="3.84 Å pairs",
            color="C4",
        )
        ax_cl.set_xlabel("cluster")
        ax_cl.set_ylabel("pair counts")
        ax_cl.set_title("Motifs inside each hotspot cluster")
        ax_cl.legend(fontsize=7)
        ax_cl.set_xticks([c["id"] for c in cluster_rows])

    fig.suptitle("Templating gel: local screw / glide in hotspot clusters", fontsize=12, y=1.01)
    matplotlib_config.savefig(OUT_PNG)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
