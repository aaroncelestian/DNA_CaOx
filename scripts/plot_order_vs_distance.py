#!/usr/bin/env python3
"""
COM order vs distance from phosphate, azimuth-averaged.

Local 20 Å wrap (~2 c-steps) after oxalate/Ca DLS. Packed merge and the
thinner 10 Å DLS wrap are shown as controls.
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
from find_symmetry import load_growth_seed_positions, load_whw_ca  # noqa: E402
from grow_crystal_from_growth import dna_slab_frame  # noqa: E402
from grow_whewellite import parse_atoms  # noqa: E402
from plot_sleeve_diagram import helix_basis  # noqa: E402

OUT_DIR = ROOT / "figures" / "crystallinity"
PHASE_COLORS = {
    "amorphous": "#d95f02",
    "intermediate": "#7570b3",
    "crystalline": "#1b9e77",
}
MODELS = {
    "local": {
        "pdb": ROOT / "DNA_CaOx_growth_whewellite20A_allP_dls.pdb",
        "csv": OUT_DIR / "DNA_CaOx_growth_whewellite20A_allP_dls_ca_metrics.csv",
        "seeds": ROOT / "DNA_CaOx_growth_allP_seeds.pdb",
        "label": "Local 20 Å DLS",
        "radius": 20.0,
    },
    "packed": {
        "pdb": ROOT / "DNA_CaOx_growth_whewellite20A_allP.pdb",
        "csv": OUT_DIR / "DNA_CaOx_growth_whewellite20A_allP_ca_metrics.csv",
        "seeds": ROOT / "DNA_CaOx_growth_allP_seeds.pdb",
        "label": "Local 20 Å packed",
        "radius": 20.0,
    },
    "thin": {
        "pdb": ROOT / "DNA_CaOx_growth_whewellite10A_allP_dls.pdb",
        "csv": OUT_DIR / "DNA_CaOx_growth_whewellite10A_allP_dls_ca_metrics.csv",
        "seeds": ROOT / "DNA_CaOx_growth_allP_seeds.pdb",
        "label": "Local 10 Å DLS",
        "radius": 10.0,
    },
    "altp": {
        "pdb": ROOT / "DNA_CaOx_growth_whewellite30A_altP_omm.pdb",
        "csv": OUT_DIR / "DNA_CaOx_growth_whewellite30A_altP_omm_ca_metrics.csv",
        "seeds": ROOT / "DNA_CaOx_growth_altP_seeds.pdb",
        "label": "Alt-P 30 Å FIRE/OMM",
        "radius": 30.0,
    },
}


def load_model(name: str) -> dict:
    spec = MODELS[name]
    atoms, _ = parse_atoms(spec["pdb"])
    nuc = [a for a in atoms if a["resname"] == "NUC"]
    origin, axis, zmin, zmax, r_dna = dna_slab_frame(nuc, pad=0.5)
    seeds = load_growth_seed_positions(spec["seeds"])
    e_x, e_y = helix_basis(origin, axis, seeds)
    pts, _ = load_whw_ca(spec["pdb"])
    rows = list(csv.DictReader(spec["csv"].open()))
    if len(rows) != len(pts):
        raise SystemExit(f"{name}: CSV/PDB Ca mismatch {len(rows)} vs {len(pts)}")
    rel = pts - origin
    z = rel @ axis
    xy = rel - z[:, None] * axis
    phi = np.degrees(np.arctan2(xy @ e_y, xy @ e_x))
    seed_xyz = np.array(list(seeds.values()), float)
    cover_r = float(spec.get("radius", 30.0))
    d_seed = np.linalg.norm(pts[:, None, :] - seed_xyz[None, :, :], axis=2)
    n_cover = (d_seed <= cover_r + 0.05).sum(axis=1)
    return {
        "label": spec["label"],
        "zmin": zmin,
        "zmax": zmax,
        "on": (z >= zmin) & (z <= zmax),
        "z": z,
        "d_p": np.array([float(r["d_p_A"]) for r in rows]),
        "score": np.array([float(r["crystallinity"]) for r in rows]),
        "phase": np.array([r["phase"] for r in rows]),
        "phi": phi,
        "face": (xy @ e_x) >= 0.0,
        "n_cover": n_cover,
        "n_seeds": len(seeds),
    }


def bin_edges(d_p: np.ndarray, width: float) -> np.ndarray:
    hi = max(float(np.ceil(d_p.max() / width) * width), width)
    return np.arange(0.0, hi + 0.5 * width, width)


def shell_stats(data: dict, edges: np.ndarray, mask: np.ndarray) -> dict:
    d_p, score, phase = data["d_p"], data["score"], data["phase"]
    n_bin = len(edges) - 1
    out = {
        "x": 0.5 * (edges[:-1] + edges[1:]),
        "n": np.zeros(n_bin, int),
        "score": np.full(n_bin, np.nan),
        "sem": np.full(n_bin, np.nan),
        "amorphous": np.full(n_bin, np.nan),
        "intermediate": np.full(n_bin, np.nan),
        "crystalline": np.full(n_bin, np.nan),
    }
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = mask & (d_p >= lo) & (d_p < hi)
        n = int(sel.sum())
        out["n"][i] = n
        if n < 3:
            continue
        s = score[sel]
        out["score"][i] = float(np.mean(s))
        out["sem"][i] = float(np.std(s, ddof=1) / np.sqrt(n))
        ph = phase[sel]
        for name in ("amorphous", "intermediate", "crystalline"):
            out[name][i] = float(np.mean(ph == name))
    return out


def main():
    matplotlib_config.apply_style()
    local = load_model("local")
    packed = load_model("packed")
    thin = load_model("thin")

    local_edges = np.arange(0.0, 24.5, 2.0)
    loc_all = shell_stats(local, local_edges, local["on"])
    loc_a = shell_stats(local, local_edges, local["on"] & local["face"])
    loc_b = shell_stats(local, local_edges, local["on"] & ~local["face"])
    pack_all = shell_stats(packed, local_edges, packed["on"])
    thin_all = shell_stats(thin, local_edges, thin["on"])

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    ax_ph, ax_sc, ax_az, ax_dls = axes.ravel()

    x = loc_all["x"]
    bottoms = np.zeros_like(x, float)
    for name, color in PHASE_COLORS.items():
        y = np.nan_to_num(loc_all[name], nan=0.0)
        ax_ph.bar(
            x,
            y,
            width=1.6,
            bottom=bottoms,
            color=color,
            label=name,
            align="center",
        )
        bottoms = bottoms + y
    ax_ph.set_xlim(-0.4, local_edges[-1] + 0.2)
    ax_ph.set_ylim(0, 1.02)
    ax_ph.set_xlabel("Distance from nearest P (Å)")
    ax_ph.set_ylabel("Ca fraction")
    ax_ph.set_title("Local 20 Å DLS — phase vs d(P)")
    ax_ph.legend(frameon=False, loc="upper right")

    ax_sc.errorbar(
        loc_a["x"],
        loc_a["score"],
        yerr=loc_a["sem"],
        fmt="o-",
        color="#4c78a8",
        label="face A",
        capsize=2,
    )
    ax_sc.errorbar(
        loc_b["x"],
        loc_b["score"],
        yerr=loc_b["sem"],
        fmt="s--",
        color="#f58518",
        label="face B",
        capsize=2,
    )
    ax_sc.axhline(0.14, color="#888", ls=":", lw=0.8, label="intermediate")
    ax_sc.axhline(0.25, color="#1b9e77", ls=":", lw=0.8, label="crystalline")
    ax_sc.set_xlim(-0.4, local_edges[-1] + 0.2)
    ax_sc.set_ylim(0, 0.45)
    ax_sc.set_xlabel("Distance from nearest P (Å)")
    ax_sc.set_ylabel("COM-net score")
    ax_sc.set_title("Local wrap — opposite faces")
    ax_sc.legend(frameon=False, loc="upper left", ncol=2)

    # Interior shell: full 10 Å scoring neighborhood (away from DNA and the cut).
    shell = local["on"] & (local["d_p"] >= 10.0) & (local["d_p"] < 16.0)
    phi_edges = np.linspace(-180, 180, 9)
    phi_c = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    means, sems, ns = [], [], []
    for lo, hi in zip(phi_edges[:-1], phi_edges[1:]):
        sel = shell & (local["phi"] >= lo) & (local["phi"] < hi)
        ns.append(int(sel.sum()))
        if sel.sum() < 3:
            means.append(np.nan)
            sems.append(np.nan)
        else:
            s = local["score"][sel]
            means.append(float(np.mean(s)))
            sems.append(float(np.std(s, ddof=1) / np.sqrt(sel.sum())))
    ax_az.bar(
        phi_c,
        np.nan_to_num(means, nan=0.0),
        width=38,
        color="#7570b3",
        alpha=0.85,
        yerr=np.nan_to_num(sems, nan=0.0),
        capsize=2,
        error_kw={"elinewidth": 0.8},
    )
    ax_az.set_xlim(-190, 190)
    ax_az.set_ylim(0, 0.35)
    ax_az.set_xlabel("Azimuth around helix (deg)")
    ax_az.set_ylabel("COM-net score")
    ax_az.set_title("20 Å DLS, 10–16 Å from P — vs azimuth")
    ax_az.text(
        0.03,
        0.92,
        f"n = {int(shell.sum())} Ca",
        transform=ax_az.transAxes,
        fontsize=9,
        color="#555",
    )

    ax_dls.errorbar(
        pack_all["x"],
        pack_all["score"],
        yerr=pack_all["sem"],
        fmt="s--",
        color="#888888",
        label="packed (rigid merge)",
        capsize=2,
    )
    ax_dls.errorbar(
        loc_all["x"],
        loc_all["score"],
        yerr=loc_all["sem"],
        fmt="o-",
        color="#4c78a8",
        label="20 Å oxalate/Ca DLS",
        capsize=2,
    )
    ax_dls.plot(
        thin_all["x"],
        thin_all["score"],
        "^-",
        color="#54a24b",
        label="10 Å DLS",
        alpha=0.85,
    )
    ax_dls.axhline(0.14, color="#888", ls=":", lw=0.8)
    ax_dls.axhline(0.25, color="#1b9e77", ls=":", lw=0.8)
    ax_dls.set_xlim(-0.4, local_edges[-1] + 0.2)
    ax_dls.set_ylim(0, 0.45)
    ax_dls.set_xlabel("Distance from nearest P (Å)")
    ax_dls.set_ylabel("COM-net score")
    ax_dls.set_title("Packed vs DLS (azimuth average)")
    ax_dls.legend(frameon=False, loc="upper left")

    fig.suptitle(
        "COM order around DNA after oxalate/Ca DLS, 20 Å wrap",
        fontsize=13,
        y=1.01,
    )
    fig.text(
        0.5,
        -0.02,
        "Source: DNA_CaOx_growth_whewellite20A_allP_dls.pdb "
        "(controls: packed 20A_allP.pdb, 10A DLS) · DNA-length Ca only",
        ha="center",
        fontsize=8,
        color="#666",
    )
    out = OUT_DIR / "DNA_CaOx_allP_order_vs_distance.png"
    matplotlib_config.savefig(out)
    print(f"Wrote {out}")

    # Short text companion for the report.
    lines = ["Azimuth-averaged COM order vs d(P)", "=" * 42, ""]
    lines.append("Local 20 Å DLS (DNA length)")
    lines.append("  dP   n   score  am    int   xtal")
    for i, x in enumerate(loc_all["x"]):
        if loc_all["n"][i] == 0:
            continue
        lines.append(
            f"  {x:4.1f} {loc_all['n'][i]:4d}  {loc_all['score'][i]:5.3f}  "
            f"{loc_all['amorphous'][i]:5.2f} {loc_all['intermediate'][i]:5.2f} "
            f"{loc_all['crystalline'][i]:5.2f}"
        )
    lines.append("")
    lines.append("Opposite faces (local wrap score)")
    lines.append("  dP    A      B     |A-B|")
    for i, x in enumerate(loc_all["x"]):
        a, b = loc_a["score"][i], loc_b["score"][i]
        if np.isnan(a) or np.isnan(b):
            continue
        lines.append(f"  {x:4.1f}  {a:5.3f}  {b:5.3f}  {abs(a-b):5.3f}")
    lines.append("")
    lines.append("Packed vs DLS (azimuth-averaged score)")
    lines.append("  dP   packed   DLS")
    for i, x in enumerate(loc_all["x"]):
        p, d = pack_all["score"][i], loc_all["score"][i]
        if np.isnan(p) and np.isnan(d):
            continue
        ps = f"{p:6.3f}" if not np.isnan(p) else "   nan"
        ds = f"{d:6.3f}" if not np.isnan(d) else "   nan"
        lines.append(f"  {x:4.1f} {ps} {ds}")
    txt = OUT_DIR / "DNA_CaOx_allP_order_vs_distance.txt"
    txt.write_text("\n".join(lines) + "\n")
    print(f"Wrote {txt}")

    plot_altp()


def plot_altp():
    alt = load_model("altp")
    local = load_model("local")
    edges = np.arange(0.0, 32.5, 2.0)
    all_s = shell_stats(alt, edges, np.ones(len(alt["d_p"]), bool))
    on_s = shell_stats(alt, edges, alt["on"])
    off_s = shell_stats(alt, edges, ~alt["on"])
    face_a = shell_stats(alt, edges, alt["on"] & alt["face"])
    face_b = shell_stats(alt, edges, alt["on"] & ~alt["face"])
    loc_on = shell_stats(local, np.arange(0.0, 24.5, 2.0), local["on"])

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    ax_ph, ax_sc, ax_cv, ax_cmp = axes.ravel()

    x = all_s["x"]
    bottoms = np.zeros_like(x, float)
    for name, color in PHASE_COLORS.items():
        y = np.nan_to_num(all_s[name], nan=0.0)
        ax_ph.bar(
            x, y, width=1.6, bottom=bottoms, color=color, label=name, align="center"
        )
        bottoms = bottoms + y
    ax_ph.set_xlim(-0.4, edges[-1] + 0.2)
    ax_ph.set_ylim(0, 1.02)
    ax_ph.set_xlabel("Distance from nearest P (Å)")
    ax_ph.set_ylabel("Ca fraction")
    ax_ph.set_title("Alt-P 30 Å FIRE/OMM — phase vs d(P)")
    ax_ph.legend(frameon=False, loc="upper right")

    ax_sc.errorbar(
        face_a["x"], face_a["score"], yerr=face_a["sem"],
        fmt="o-", color="#4c78a8", label="face A (DNA length)", capsize=2,
    )
    ax_sc.errorbar(
        face_b["x"], face_b["score"], yerr=face_b["sem"],
        fmt="s--", color="#f58518", label="face B (DNA length)", capsize=2,
    )
    ax_sc.axhline(0.14, color="#888", ls=":", lw=0.8)
    ax_sc.axhline(0.25, color="#1b9e77", ls=":", lw=0.8)
    ax_sc.set_xlim(-0.4, edges[-1] + 0.2)
    ax_sc.set_ylim(0, 0.45)
    ax_sc.set_xlabel("Distance from nearest P (Å)")
    ax_sc.set_ylabel("COM-net score")
    ax_sc.set_title("Opposite faces (DNA-length Ca only)")
    ax_sc.legend(frameon=False, loc="upper left")

    cover_vals = np.arange(1, int(alt["n_cover"].max()) + 1)
    means, sems, ns, xtal = [], [], [], []
    for k in cover_vals:
        sel = alt["n_cover"] == k
        ns.append(int(sel.sum()))
        if sel.sum() < 3:
            means.append(np.nan)
            sems.append(np.nan)
            xtal.append(np.nan)
        else:
            s = alt["score"][sel]
            means.append(float(np.mean(s)))
            sems.append(float(np.std(s, ddof=1) / np.sqrt(sel.sum())))
            xtal.append(float(np.mean(alt["phase"][sel] == "crystalline")))
    ax_cv.errorbar(
        cover_vals, means, yerr=sems, fmt="o-", color="#4c78a8", capsize=2, label="score"
    )
    ax_cv2 = ax_cv.twinx()
    ax_cv2.plot(cover_vals, xtal, "s--", color="#1b9e77", label="crystal frac")
    ax_cv.set_xlabel("Covering seeds (30 Å spheres)")
    ax_cv.set_ylabel("COM-net score")
    ax_cv2.set_ylabel("Crystalline fraction")
    ax_cv.set_title("Order vs seed overlap")
    ax_cv.set_ylim(0, 0.45)
    ax_cv2.set_ylim(0, 1.02)
    h1, l1 = ax_cv.get_legend_handles_labels()
    h2, l2 = ax_cv2.get_legend_handles_labels()
    ax_cv.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")

    ax_cmp.errorbar(
        on_s["x"], on_s["score"], yerr=on_s["sem"],
        fmt="o-", color="#4c78a8", label="Alt-P DNA-length", capsize=2,
    )
    ax_cmp.errorbar(
        off_s["x"], off_s["score"], yerr=off_s["sem"],
        fmt="^--", color="#e45756", label="Alt-P past termini", capsize=2,
    )
    ax_cmp.plot(
        loc_on["x"], loc_on["score"], "s-", color="#54a24b",
        alpha=0.85, label="22-seed 20 Å DLS",
    )
    ax_cmp.axhline(0.14, color="#888", ls=":", lw=0.8)
    ax_cmp.axhline(0.25, color="#1b9e77", ls=":", lw=0.8)
    ax_cmp.set_xlim(-0.4, edges[-1] + 0.2)
    ax_cmp.set_ylim(0, 0.45)
    ax_cmp.set_xlabel("Distance from nearest P (Å)")
    ax_cmp.set_ylabel("COM-net score")
    ax_cmp.set_title("DNA-length vs overhang")
    ax_cmp.legend(frameon=False, loc="upper left")

    fig.suptitle(
        "COM order after every-other-P 30 Å wrap (FIRE + OpenMM)",
        fontsize=13,
        y=1.01,
    )
    fig.text(
        0.5,
        -0.02,
        "Source: DNA_CaOx_growth_whewellite30A_altP_omm.pdb "
        "(12 COM seeds, stride-2 phosphates) · FIRE/OMM, freeze-beyond=0",
        ha="center",
        fontsize=8,
        color="#666",
    )
    out = OUT_DIR / "DNA_CaOx_altP_order_vs_distance.png"
    matplotlib_config.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")

    xtal = alt["phase"] == "crystalline"
    lines = [
        "Alt-P 30 Å FIRE/OMM — COM order vs d(P) and covering seeds",
        "=" * 58,
        f"Seeds: {alt['n_seeds']}",
        f"Ca: {len(alt['phase'])}  "
        f"amorphous={(alt['phase']=='amorphous').sum()}  "
        f"intermediate={(alt['phase']=='intermediate').sum()}  "
        f"crystalline={int(xtal.sum())}",
        "",
        "Crystal location",
        f"  DNA-length : {int((xtal & alt['on']).sum())}",
        f"  past termini: {int((xtal & ~alt['on']).sum())}",
        "",
        "Covering-seed counts (all Ca / crystal Ca)",
    ]
    for k in cover_vals:
        n_all = int((alt["n_cover"] == k).sum())
        n_x = int(((alt["n_cover"] == k) & xtal).sum())
        if n_all == 0:
            continue
        lines.append(f"  {k:2d} seeds: {n_all:4d} Ca  ({n_x:4d} crystal)")
    lines.append("")
    lines.append("DNA-length shells")
    lines.append("  dP   n   score  am    int   xtal")
    for i, xv in enumerate(on_s["x"]):
        if on_s["n"][i] == 0:
            continue
        lines.append(
            f"  {xv:4.1f} {on_s['n'][i]:4d}  {on_s['score'][i]:5.3f}  "
            f"{on_s['amorphous'][i]:5.2f} {on_s['intermediate'][i]:5.2f} "
            f"{on_s['crystalline'][i]:5.2f}"
        )
    lines.append("")
    lines.append("Opposite faces (DNA-length score)")
    lines.append("  dP    A      B     |A-B|")
    for i, xv in enumerate(on_s["x"]):
        a, b = face_a["score"][i], face_b["score"][i]
        if np.isnan(a) or np.isnan(b):
            continue
        lines.append(f"  {xv:4.1f}  {a:5.3f}  {b:5.3f}  {abs(a - b):5.3f}")
    txt = OUT_DIR / "DNA_CaOx_altP_order_vs_distance.txt"
    txt.write_text("\n".join(lines) + "\n")
    print(f"Wrote {txt}")


if __name__ == "__main__":
    main()
