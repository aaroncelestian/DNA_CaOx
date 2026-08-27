#!/usr/bin/env python3
"""
Diagram of COM order around DNA.

--geometry slab    cylinder (DNA-length coating); default
--geometry sphere  union of 30 Å spheres around the four seed Ca
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib_config  # noqa: E402
from find_symmetry import load_growth_seed_positions, load_whw_ca  # noqa: E402
from grow_crystal_from_growth import dna_slab_frame  # noqa: E402
from grow_whewellite import parse_atoms  # noqa: E402

SEED_PDB = ROOT / "DNA_CaOx_growth.pdb"
SEED_R = 30.0

PHASE_COLORS = {
    "amorphous": "#d95f02",
    "intermediate": "#7570b3",
    "crystalline": "#1b9e77",
}
PHASE_ORDER = ("amorphous", "intermediate", "crystalline")
PHASE_SIZE = {"amorphous": 7, "intermediate": 7, "crystalline": 22}
PHASE_Z = {"amorphous": 1, "intermediate": 2, "crystalline": 3}

PATHS = {
    "slab": {
        "pdb": ROOT / "DNA_CaOx_growth_whewellite30A_dls.pdb",
        "csv": ROOT
        / "figures/crystallinity/DNA_CaOx_growth_whewellite30A_slab_dls_ca_metrics.csv",
        "out": ROOT
        / "figures/crystallinity/DNA_CaOx_growth_whewellite30A_slab_dls_sleeve_diagram.png",
    },
    # Spherical DLS PDB was overwritten by the cylinder; the 1605-Ca spherical
    # envelope is still in the relaxed file, and phases match the spherical DLS CSV.
    "sphere": {
        "pdb": ROOT / "DNA_CaOx_growth_whewellite30A_relaxed.pdb",
        "csv": ROOT
        / "figures/crystallinity/DNA_CaOx_growth_whewellite30A_dls_ca_metrics.csv",
        "out": ROOT
        / "figures/crystallinity/DNA_CaOx_growth_whewellite30A_dls_sleeve_diagram.png",
    },
    "altp": {
        "pdb": ROOT / "DNA_CaOx_growth_whewellite30A_altP_omm.pdb",
        "csv": ROOT
        / "figures/crystallinity/DNA_CaOx_growth_whewellite30A_altP_omm_ca_metrics.csv",
        "out": ROOT
        / "figures/crystallinity/DNA_CaOx_growth_whewellite30A_altP_omm_sleeve_diagram.png",
        "seeds": ROOT / "DNA_CaOx_growth_altP_seeds.pdb",
        "seed_r": 30.0,
    },
}


def load_phases(path: Path) -> np.ndarray:
    phases = []
    with path.open() as f:
        for row in csv.DictReader(f):
            phases.append(row["phase"])
    return np.array(phases)


def load_dna_p(path: Path):
    atoms, _ = parse_atoms(path)
    nuc = [a for a in atoms if a["resname"] == "NUC"]
    p_by_chain: dict[str, list] = {}
    for a in nuc:
        name = a["name"].strip().upper()
        el = a.get("element", name[:1]).upper()
        if name != "P" and el != "P":
            continue
        p_by_chain.setdefault(a["chain"], []).append(a)
    for ch in p_by_chain:
        p_by_chain[ch].sort(key=lambda a: a["resseq"])
    return nuc, p_by_chain


def helix_basis(origin, axis, seeds: dict):
    """Seed-side transverse axis so the nucleating strand sits at +x."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    seed_xyz = np.array(list(seeds.values()), float)
    rel = seed_xyz - origin
    perp = rel - np.outer(rel @ axis, axis)
    mean_perp = perp.mean(axis=0)
    n = np.linalg.norm(mean_perp)
    if n < 1e-6:
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(tmp @ axis) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        mean_perp = np.cross(axis, tmp)
        n = np.linalg.norm(mean_perp)
    e_x = mean_perp / n
    e_y = np.cross(axis, e_x)
    e_y = e_y / np.linalg.norm(e_y)
    return e_x, e_y


def project(pts, origin, axis, e_x, e_y):
    rel = pts - origin
    z = rel @ axis
    x = rel @ e_x
    y = rel @ e_y
    return x, y, z


def scatter_phases(ax, x, y, phase, **kwargs):
    for ph in PHASE_ORDER:
        m = phase == ph
        if not np.any(m):
            continue
        ax.scatter(
            x[m],
            y[m],
            c=PHASE_COLORS[ph],
            s=PHASE_SIZE[ph],
            alpha=0.72 if ph != "crystalline" else 0.95,
            linewidths=0.0,
            zorder=PHASE_Z[ph],
            rasterized=True,
            **kwargs,
        )


def _dna_cartoon(ax, t0=0.16, t1=0.84, y0=0.50):
    t = np.linspace(t0, t1, 200)
    ax.plot(t, y0 + 0.045 * np.sin(18 * np.pi * t), color="#222", lw=1.6, zorder=5)
    ax.plot(
        t, y0 + 0.045 * np.sin(18 * np.pi * t + np.pi), color="#555", lw=1.6, zorder=5
    )
    ax.text(0.50, y0 + 0.075, "DNA", ha="center", va="bottom", fontsize=10, color="#222")


def draw_schematic_slab(ax):
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("C  Open 3D (schematic)", loc="left", pad=8)
    _dna_cartoon(ax)
    ax.add_patch(
        FancyBboxPatch(
            (0.14, 0.28),
            0.72,
            0.44,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor=PHASE_COLORS["intermediate"],
            edgecolor="#333",
            linewidth=1.0,
            linestyle="--",
            alpha=0.22,
            zorder=1,
        )
    )
    ax.text(
        0.50,
        0.30,
        "DNA-length slab\n(helical wrap → intermediate)",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#4a3f7a",
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.22, 0.42),
            0.56,
            0.16,
            boxstyle="round,pad=0.004,rounding_size=0.02",
            facecolor=PHASE_COLORS["amorphous"],
            edgecolor="none",
            alpha=0.28,
            zorder=2,
        )
    )
    lobes = [
        (0.02, 0.50, 0.13, 0.20, "past\nterminus"),
        (0.98, 0.50, 0.13, 0.20, "past\nterminus"),
        (0.50, 0.92, 0.28, 0.12, "off the duplex\ninto bulk 3D"),
        (0.50, 0.08, 0.22, 0.10, ""),
    ]
    for cx, cy, w, h, label in lobes:
        ax.add_patch(
            Ellipse(
                (cx, cy),
                w,
                h,
                facecolor=PHASE_COLORS["crystalline"],
                edgecolor="#0d5c47",
                linewidth=0.8,
                alpha=0.55,
                zorder=3,
            )
        )
        if label:
            ax.text(
                cx,
                cy,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                color="#063d30",
                zorder=6,
            )
    ax.annotate(
        "",
        xy=(0.50, 0.80),
        xytext=(0.50, 0.64),
        arrowprops=dict(arrowstyle="-|>", color="#0d5c47", lw=1.4),
        zorder=6,
    )
    ax.annotate(
        "",
        xy=(0.12, 0.50),
        xytext=(0.20, 0.50),
        arrowprops=dict(arrowstyle="-|>", color="#0d5c47", lw=1.4),
        zorder=6,
    )
    ax.annotate(
        "",
        xy=(0.88, 0.50),
        xytext=(0.80, 0.50),
        arrowprops=dict(arrowstyle="-|>", color="#0d5c47", lw=1.4),
        zorder=6,
    )
    ax.text(
        0.50,
        0.005,
        "Green: coherent COM where the lattice is not forced around the helix",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#333",
    )


def draw_schematic_sphere(ax):
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("C  Sphere cut is open 3D", loc="left", pad=8)
    _dna_cartoon(ax, t0=0.28, t1=0.72, y0=0.42)
    # Four overlapping 30 Å seed spheres (cartoon)
    for i, cx in enumerate((0.34, 0.44, 0.56, 0.66)):
        ax.add_patch(
            Circle(
                (cx, 0.50),
                0.22 + 0.01 * i,
                facecolor=PHASE_COLORS["intermediate"],
                edgecolor="#333",
                lw=0.8,
                ls="--",
                alpha=0.12,
                zorder=1,
            )
        )
    ax.add_patch(
        Ellipse(
            (0.50, 0.78),
            0.42,
            0.22,
            facecolor=PHASE_COLORS["crystalline"],
            edgecolor="#0d5c47",
            lw=0.9,
            alpha=0.55,
            zorder=3,
        )
    )
    ax.add_patch(
        Ellipse(
            (0.12, 0.48),
            0.16,
            0.22,
            facecolor=PHASE_COLORS["crystalline"],
            edgecolor="#0d5c47",
            lw=0.8,
            alpha=0.50,
            zorder=3,
        )
    )
    ax.add_patch(
        Ellipse(
            (0.88, 0.48),
            0.16,
            0.22,
            facecolor=PHASE_COLORS["crystalline"],
            edgecolor="#0d5c47",
            lw=0.8,
            alpha=0.50,
            zorder=3,
        )
    )
    ax.text(
        0.50,
        0.78,
        "coherent COM\nin the seed balloon",
        ha="center",
        va="center",
        fontsize=8,
        color="#063d30",
        zorder=6,
    )
    ax.text(0.12, 0.48, "past\nend", ha="center", va="center", fontsize=7.5, color="#063d30", zorder=6)
    ax.text(0.88, 0.48, "past\nend", ha="center", va="center", fontsize=7.5, color="#063d30", zorder=6)
    ax.text(
        0.50,
        0.18,
        "30 Å spheres around 4 seed Ca\n(not a wrap around the duplex)",
        ha="center",
        va="center",
        fontsize=8,
        color="#4a3f7a",
    )
    ax.plot([0.34, 0.44, 0.56, 0.66], [0.50, 0.50, 0.50, 0.50], "D", ms=5, color="#111", markerfacecolor="white", zorder=7)
    ax.text(
        0.50,
        0.005,
        "Green is in the model: one orientation growing into open 3D",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#333",
    )


def plot_phosphates_and_seeds(ax, p_xy, sx, sy, coord="xy"):
    dna_colors = {"A": "#111", "B": "#444"}
    for i, (ch, pair) in enumerate(p_xy.items()):
        a, b = pair
        ax.plot(
            a,
            b,
            "o",
            ms=3.2,
            color=dna_colors.get(ch, "#222"),
            markeredgecolor="white",
            markeredgewidth=0.3,
            zorder=8,
        )
    ax.plot(
        sx,
        sy,
        "D",
        ms=6.5,
        color="#111",
        markerfacecolor="white",
        markeredgewidth=1.1,
        zorder=9,
    )


def main():
    ap = argparse.ArgumentParser(description="DNA sleeve vs spherical-growth crystallinity diagram")
    ap.add_argument(
        "--geometry",
        choices=("slab", "sphere", "altp"),
        default="slab",
        help="slab = DNA-length cylinder; sphere = 30 Å around seed Ca",
    )
    args = ap.parse_args()
    geom = args.geometry
    paths = PATHS[geom]
    seed_r = float(paths.get("seed_r", SEED_R))
    seed_pdb = paths.get("seeds", SEED_PDB)

    matplotlib_config.apply_style()
    plt.rcParams["axes.grid"] = False

    pts, _ = load_whw_ca(paths["pdb"])
    phase = load_phases(paths["csv"])
    if len(phase) != len(pts):
        raise SystemExit(f"CSV/PDB Ca count mismatch: {len(phase)} vs {len(pts)}")

    nuc, p_by_chain = load_dna_p(paths["pdb"])
    origin, axis, zmin, zmax, r_dna = dna_slab_frame(nuc, pad=0.5)
    r_coat = r_dna + seed_r
    seeds = load_growth_seed_positions(seed_pdb)
    e_x, e_y = helix_basis(origin, axis, seeds)

    cx, cy, cz = project(pts, origin, axis, e_x, e_y)
    seed_xyz = np.array(list(seeds.values()), float)
    sx, sy, sz = project(seed_xyz, origin, axis, e_x, e_y)

    p_xy = {}
    p_zy = {}
    for ch, atoms in p_by_chain.items():
        pxyz = np.array([a["xyz"] for a in atoms], float)
        px, py, pz = project(pxyz, origin, axis, e_x, e_y)
        p_xy[ch] = (px, py)
        p_zy[ch] = (pz, px)

    n_cryst = int((phase == "crystalline").sum())
    n_int = int((phase == "intermediate").sum())
    n_am = int((phase == "amorphous").sum())

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.4, 4.85),
        gridspec_kw={"width_ratios": [1.0, 1.15, 1.05]},
    )
    ax_end, ax_side, ax_sch = axes

    # --- A: end-on ---
    scatter_phases(ax_end, cx, cy, phase)
    plot_phosphates_and_seeds(ax_end, p_xy, sx, sy)
    ax_end.add_patch(Circle((0, 0), r_dna, fill=False, ls=":", lw=0.9, color="#666", zorder=4))
    ax_end.text(
        0.0,
        0.0,
        "DNA",
        ha="center",
        va="center",
        fontsize=8,
        color="#111",
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
    )
    if geom == "slab":
        ax_end.add_patch(
            Circle((0, 0), r_coat, fill=False, ls="--", lw=0.9, color="#666", zorder=4)
        )
        ax_end.text(0.0, r_coat + 1.8, "30 Å coating", ha="center", va="bottom", fontsize=8, color="#444")
        lim = r_coat + 4
        ax_end.set_title("A  Around the duplex", loc="left")
    else:
        for x0, y0 in zip(sx, sy):
            ax_end.add_patch(
                Circle(
                    (x0, y0),
                    seed_r,
                    fill=False,
                    ls="--",
                    lw=0.8,
                    color="#666",
                    alpha=0.7,
                    zorder=4,
                )
            )
        ax_end.set_title("A  Around the duplex (seed-side patch)", loc="left")
        pad = 4.0
        lim = max(np.abs(cx).max(), np.abs(cy).max(), seed_r + np.hypot(sx, sy).max()) + pad
    ax_end.set_aspect("equal")
    ax_end.set_xlabel("Toward seed strand (Å)")
    ax_end.set_ylabel("Perpendicular (Å)")
    ax_end.set_xlim(-lim, lim)
    ax_end.set_ylim(-lim, lim)
    ax_end.axhline(0, color="#ccc", lw=0.5, zorder=0)
    ax_end.axvline(0, color="#ccc", lw=0.5, zorder=0)

    # --- B: side view ---
    scatter_phases(ax_side, cz, cx, phase)
    plot_phosphates_and_seeds(ax_side, p_zy, sz, sx)
    zmid = 0.5 * (zmin + zmax)
    if geom == "slab":
        ax_side.add_patch(
            Rectangle(
                (zmin, -r_coat),
                zmax - zmin,
                2 * r_coat,
                fill=False,
                ls="--",
                lw=1.0,
                color="#333",
                zorder=4,
            )
        )
        for zc in (zmin - 5.5, zmax + 5.5):
            ax_side.add_patch(
                Ellipse(
                    (zc, 0.0),
                    10.5,
                    22,
                    facecolor=PHASE_COLORS["crystalline"],
                    edgecolor="#0d5c47",
                    lw=0.8,
                    alpha=0.28,
                    zorder=0,
                )
            )
        ax_side.add_patch(
            Ellipse(
                (zmid, r_coat + 8.5),
                0.62 * (zmax - zmin),
                14,
                facecolor=PHASE_COLORS["crystalline"],
                edgecolor="#0d5c47",
                lw=0.8,
                alpha=0.28,
                zorder=0,
            )
        )
        ax_side.annotate(
            "open 3D:\ncoherent COM",
            xy=(zmax + 5.2, 0.0),
            xytext=(zmax + 11.5, 30),
            fontsize=8,
            color="#0d5c47",
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-|>", color="#0d5c47", lw=1.1),
        )
        ax_side.annotate(
            "",
            xy=(zmid, r_coat + 8.0),
            xytext=(zmid, r_coat - 4.0),
            arrowprops=dict(arrowstyle="-|>", color="#0d5c47", lw=1.1),
        )
        ax_side.text(zmid, -r_coat - 2.4, "slab crop", ha="center", va="top", fontsize=8, color="#333")
        ax_side.set_title("B  Along the duplex", loc="left")
        ax_side.set_xlim(zmin - 13, zmax + 27)
        ax_side.set_ylim(-r_coat - 8, r_coat + 18)
    else:
        ax_side.axvline(zmin, color="#333", ls="--", lw=0.9, zorder=4)
        ax_side.axvline(zmax, color="#333", ls="--", lw=0.9, zorder=4)
        for x0, z0 in zip(sx, sz):
            ax_side.add_patch(
                Circle(
                    (z0, x0),
                    seed_r,
                    fill=False,
                    ls="--",
                    lw=0.8,
                    color="#666",
                    alpha=0.65,
                    zorder=4,
                )
            )
        n_past = int(((cz < zmin) | (cz > zmax)).sum())
        z_lo, z_hi = cz.min() - 4, cz.max() + 4
        x_lo, x_hi = min(cx.min(), sx.min() - seed_r) - 4, max(cx.max(), sx.max() + seed_r) + 8
        ax_side.set_xlim(z_lo, z_hi + 6)
        ax_side.set_ylim(x_lo, x_hi)
        ax_side.text(
            zmin - 1.2,
            x_lo + 2,
            "DNA end",
            ha="right",
            va="bottom",
            fontsize=8,
            color="#333",
            rotation=90,
        )
        ax_side.annotate(
            f"grows past DNA\n({n_past} Ca beyond termini)",
            xy=(cz.min() + 2, np.median(cx[cz < zmin]) if np.any(cz < zmin) else 0),
            xytext=(z_lo + 2, x_hi - 8),
            fontsize=8,
            color="#0d5c47",
            ha="left",
            arrowprops=dict(arrowstyle="-|>", color="#0d5c47", lw=1.1),
        )
        ax_side.set_title("B  Along the duplex (30 Å seed spheres)", loc="left")

    ax_side.set_xlabel("Along helix (Å)")
    ax_side.set_ylabel("Toward seed strand (Å)")
    ax_side.set_aspect("equal")

    if geom == "slab":
        draw_schematic_slab(ax_sch)
        title = "Helical sleeve around DNA is intermediate; crystal is likelier in open 3D"
        note = (
            f"Cylinder DLS model, strict COM net.  "
            f"Ca: {n_am} amorphous, {n_int} intermediate, {n_cryst} crystalline.  "
            f"Green lobes in B are schematic (not in the cropped model)."
        )
    elif geom == "altp":
        draw_schematic_sphere(ax_sch)
        title = "Every other P, 30 Å wrap after FIRE/OpenMM"
        note = (
            f"12 COM seeds (stride-2 on both strands), union of 30 Å spheres, "
            f"rigid-oxalate FIRE then OpenMM.  "
            f"Ca: {n_am} amorphous, {n_int} intermediate, {n_cryst} crystalline."
        )
    else:
        draw_schematic_sphere(ax_sch)
        title = "Spherical 30 Å cut: one-sided patch, crystal away from DNA and past the ends"
        note = (
            f"Spherical model (union of 30 Å spheres around 4 seed Ca), strict COM net.  "
            f"Ca: {n_am} amorphous, {n_int} intermediate, {n_cryst} crystalline.  "
            f"Coordinates from the spherical 30 Å model; phases from the DLS analysis."
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PHASE_COLORS[ph],
            markersize=8 if ph != "crystalline" else 10,
            label=ph,
        )
        for ph in PHASE_ORDER
    ]
    handles += [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#111",
            markersize=5,
            label="phosphate",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="#111",
            markerfacecolor="white",
            markersize=7,
            label="COM seeds",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.text(0.01, -0.06, note, fontsize=8, color="#444")

    matplotlib_config.savefig(paths["out"])
    print(f"Wrote {paths['out']}")
    print(f"geometry={geom}  DNA z {zmin:.1f}–{zmax:.1f}  Ca z {cz.min():.1f}–{cz.max():.1f}")
    print(f"phase counts  amorphous={n_am}  intermediate={n_int}  crystalline={n_cryst}")


if __name__ == "__main__":
    main()
