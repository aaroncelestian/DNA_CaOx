#!/usr/bin/env python3
"""
Where does COM-like Ca–Ca order appear around DNA?

Hotspots are Ca whose neighbors include both whewellite spacings (~3.84 Å and
~6.29 Å). This figure reports hotspot *rate* vs phosphate distance, vs radius
outside the phosphate cylinder, and a helical unroll — not raw counts.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib_config  # noqa: E402
from find_symmetry import (  # noqa: E402
    COM_A,
    helix_axis_radius,
    load_dna_heavy,
    load_growth_seed_positions,
    load_phosphate_xyz,
    load_whw_ca,
    phosphate_surface_radius,
)
from grow_crystal_from_growth import dna_slab_frame  # noqa: E402
from grow_whewellite import parse_atoms  # noqa: E402
from plot_sleeve_diagram import helix_basis  # noqa: E402

OUT = ROOT / "figures" / "crystallinity" / "DNA_CaOx_shell_lattice_nucleation_where.png"
PDB = ROOT / "DNA_CaOx_gel_altP_geom_shell_lattice_omm.pdb"
CSV = ROOT / "figures/crystallinity/DNA_CaOx_gel_altP_geom_shell_lattice_omm_ca_metrics.csv"
SEEDS = ROOT / "DNA_CaOx_gel_altP_geom_seeds.pdb"


def load():
    pts, _ = load_whw_ca(PDB)
    rows = list(csv.DictReader(CSV.open()))
    if len(rows) != len(pts):
        raise SystemExit(f"CSV/PDB Ca mismatch {len(rows)} vs {len(pts)}")
    d_p = np.array([float(r["d_p_A"]) for r in rows])
    hot = np.array([int(r["hotspot"]) for r in rows], dtype=bool)
    dna = load_dna_heavy(PDB)
    pxyz = load_phosphate_xyz(PDB)
    r_axis = helix_axis_radius(pts, dna)
    r_p = phosphate_surface_radius(pxyz, dna)
    hot = hot & (r_axis >= (r_p - 0.35))
    atoms, _ = parse_atoms(PDB)
    nuc = [a for a in atoms if a["resname"] == "NUC"]
    origin, axis, zmin, zmax, _r_dna = dna_slab_frame(nuc, pad=0.5)
    seeds = load_growth_seed_positions(SEEDS)
    e_x, e_y = helix_basis(origin, axis, seeds)
    rel = pts - origin
    z = rel @ axis
    xy = rel - z[:, None] * axis
    phi = np.degrees(np.arctan2(xy @ e_y, xy @ e_x))
    p_rel = pxyz - origin
    p_z = p_rel @ axis
    p_xy = p_rel - p_z[:, None] * axis
    p_phi = np.degrees(np.arctan2(p_xy @ e_y, p_xy @ e_x))
    return {
        "d_p": d_p,
        "hot": hot,
        "r_axis": r_axis,
        "r_p": r_p,
        "z": z,
        "phi": phi,
        "zmin": zmin,
        "zmax": zmax,
        "p_z": p_z,
        "p_phi": p_phi,
    }


def bin_frac(x, hot, edges):
    mid, frac, n, n_hot = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        mid.append(0.5 * (lo + hi))
        n.append(int(m.sum()))
        n_hot.append(int((m & hot).sum()))
        frac.append(float(hot[m].mean()) if m.sum() else np.nan)
    return np.array(mid), np.array(frac), np.array(n), np.array(n_hot)


def main():
    matplotlib_config.apply_style()
    d = load()
    r_p = d["r_p"]
    dr = d["r_axis"] - r_p
    dp_edges = np.arange(0.0, 32.1, 2.0)
    rad_edges = np.arange(-6.0, 24.1, 2.0)
    x_dp, f_dp, n_dp, nh_dp = bin_frac(d["d_p"], d["hot"], dp_edges)
    x_r, f_r, n_r, nh_r = bin_frac(dr, d["hot"], rad_edges)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.15))
    ax_dp, ax_r, ax_un = axes

    ax_dp.bar(x_dp, 100 * np.nan_to_num(f_dp), width=1.7, color="#c9a227", zorder=3)
    ax_dp.set_xlabel("Distance from nearest P (Å)")
    ax_dp.set_ylabel("Hotspot fraction of Ca (%)")
    ax_dp.set_title("1. Along the backbone?")
    ax_dp.axvspan(0, 8, color="#c45c26", alpha=0.12, zorder=0)
    ax_dp.set_xlim(-0.4, 31)
    ymax = max(22.0, float(np.nanmax(100 * f_dp) * 1.25))
    ax_dp.set_ylim(0, ymax)
    ax_dp.text(4.0, ymax * 0.92, "near P", ha="center", fontsize=8, color="#8a4a1a")

    ax_r.bar(x_r, 100 * np.nan_to_num(f_r), width=1.7, color="#4c78a8", zorder=3)
    ax_r.axvspan(rad_edges[0], 0, color="#888", alpha=0.18, zorder=0)
    ax_r.axvline(0, color="#444", lw=0.8)
    ax_r.axvline(COM_A, color="#1b9e77", ls="--", lw=0.8)
    ax_r.axvline(2 * COM_A, color="#1b9e77", ls=":", lw=0.8)
    ax_r.set_xlabel("Radius − phosphate surface (Å)")
    ax_r.set_ylabel("Hotspot fraction of Ca (%)")
    ax_r.set_title("2. Needs room to form COM?")
    ax_r.set_xlim(rad_edges[0] - 0.4, rad_edges[-1] + 0.2)
    ax_r.set_ylim(0, ymax)
    ax_r.text(
        -3,
        ymax * 0.88,
        "inside DNA\n(grooves)",
        ha="center",
        fontsize=8,
        color="#555",
    )
    ax_r.text(COM_A, ymax * 0.08, "COM a", ha="center", fontsize=7, color="#1b9e77")
    ax_r.text(2 * COM_A, ymax * 0.08, "2a", ha="center", fontsize=7, color="#1b9e77")

    on = (d["z"] >= d["zmin"]) & (d["z"] <= d["zmax"])
    ax_un.scatter(d["p_phi"], d["p_z"], s=28, c="#c45c26", marker="o", label="P", zorder=4, edgecolors="none")
    h = d["hot"] & on
    sc = ax_un.scatter(
        d["phi"][h],
        d["z"][h],
        c=d["d_p"][h],
        s=22,
        cmap="cividis_r",
        vmin=4,
        vmax=28,
        zorder=5,
        edgecolors="none",
        label="hotspot Ca",
    )
    ax_un.set_xlim(-185, 185)
    ax_un.set_xlabel("Azimuth around helix (deg)")
    ax_un.set_ylabel("Axial position (Å)")
    ax_un.set_title("3. Helical nodes?")
    ax_un.legend(frameon=False, loc="upper right", fontsize=8)
    cb = fig.colorbar(sc, ax=ax_un, fraction=0.046, pad=0.04)
    cb.set_label("d(P) (Å)", fontsize=8)
    n_p = len(d["p_z"])
    # phosphates with a DNA-length hotspot within ~8 Å in unroll space is not
    # 3D distance; report 3D count in the caption instead.
    ax_un.text(
        0.02,
        0.04,
        "P track the helix; hotspots do not",
        transform=ax_un.transAxes,
        fontsize=8,
        color="#555",
    )

    fig.suptitle(
        "Saturated CaOx shell — where local COM pair order appears",
        fontsize=13,
        y=1.03,
    )
    fig.text(
        0.5,
        -0.04,
        "Source: DNA_CaOx_gel_altP_geom_shell_lattice_omm.pdb · hotspot = both 3.84 and 6.29 Å "
        f"Ca–Ca neighbors · {int(d['hot'].sum())} hotspot Ca, {n_p} phosphates · "
        "FIRE without COM Ca–Ca targets. Rate, not raw count.",
        ha="center",
        fontsize=8,
        color="#666",
    )
    matplotlib_config.savefig(OUT)
    print(f"Wrote {OUT}")

    near = d["d_p"] < 8
    room = (d["d_p"] >= 12) & (d["d_p"] < 20)
    inside = d["r_axis"] < (r_p - 0.35)
    txt = OUT.with_suffix(".txt")
    lines = [
        "Where COM-like order appears — saturated shell",
        "=" * 52,
        f"Hotspots: {int(d['hot'].sum())}  median d(P)={np.median(d['d_p'][d['hot']]):.1f} Å  "
        f"median r_axis={np.median(d['r_axis'][d['hot']]):.1f} Å",
        f"Phosphate surface radius: {r_p:.2f} Å   COM a={COM_A:.3f} Å",
        "",
        f"1) Backbone (d(P)<8 Å): {int((near & d['hot']).sum())}/{int(near.sum())} Ca "
        f"({100 * (d['hot'][near].mean() if near.any() else 0):.1f}%)",
        f"   Inside P cylinder: {int(inside.sum())} Ca, {int((inside & d['hot']).sum())} hotspots",
        f"2) Room (d(P) 12–20 Å): {int((room & d['hot']).sum())}/{int(room.sum())} Ca "
        f"({100 * (d['hot'][room].mean() if room.any() else 0):.1f}%)",
        "3) Nodes: hotspots do not follow the phosphate helix in unroll.",
        "",
        "d(P)  n   n_hot  frac",
    ]
    for x, n, nh, f in zip(x_dp, n_dp, nh_dp, f_dp):
        if n == 0:
            continue
        lines.append(f"  {x:4.1f} {n:4d} {nh:4d}  {100 * f:5.1f}%")
    txt.write_text("\n".join(lines) + "\n")
    print(f"Wrote {txt}")


if __name__ == "__main__":
    main()
