#!/usr/bin/env python3
"""
Strand Ca–Ca and 3.84 Å hits: honest gel vs planted alt-P vs no-DNA blob.
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
from strand_order import COM_384, COM_A, WIN_A, WIN_384, strand_order_metrics  # noqa: E402
from find_symmetry import load_whw_ca  # noqa: E402

OUT = ROOT / "figures" / "crystallinity" / "DNA_CaOx_templating_strand_order.png"
TXT = OUT.with_suffix(".txt")

MODELS = [
    {
        "name": "honest",
        "label": "Honest gel (no COM targets)",
        "pdb": ROOT / "DNA_CaOx_templating_gel_omm.pdb",
        "color": "#4c78a8",
    },
    {
        "name": "altp",
        "label": "Planted alt-P 30 Å (COM targets)",
        "pdb": ROOT / "DNA_CaOx_growth_whewellite30A_altP_omm.pdb",
        "color": "#f58518",
    },
    {
        "name": "nodna",
        "label": "No-DNA blob (same N, no COM targets)",
        "pdb": ROOT / "DNA_CaOx_templating_gel_nodna_omm.pdb",
        "color": "#888888",
    },
]


def load_sm(spec):
    if not spec["pdb"].exists():
        return None
    pts, _ = load_whw_ca(spec["pdb"])
    sm = strand_order_metrics(spec["pdb"], pts)
    sm["n_ca"] = len(pts)
    return sm


def main():
    matplotlib_config.apply_style()
    loaded = []
    for spec in MODELS:
        sm = load_sm(spec)
        if sm is None:
            print(f"skip {spec['name']}: missing {spec['pdb'].name}")
            continue
        loaded.append((spec, sm))
    if not loaded:
        raise SystemExit("No templating PDBs found")

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.2))
    ax_h, ax_b, ax_u = axes

    bins = np.arange(3.2, 8.2, 0.25)
    for spec, sm in loaded:
        if len(sm["seq_d"]) == 0:
            continue
        ax_h.hist(
            sm["seq_d"],
            bins=bins,
            histtype="step",
            lw=1.6,
            color=spec["color"],
            label=spec["label"],
            density=True,
        )
    ax_h.axvline(COM_384, color="#1b9e77", ls=":", lw=0.9)
    ax_h.axvline(COM_A, color="#1b9e77", ls="--", lw=0.9)
    ax_h.set_xlabel("Sequential Ca–Ca along strand (Å)")
    ax_h.set_ylabel("Density")
    ax_h.set_title("1. Strand Ca–Ca")
    ax_h.legend(frameon=False, fontsize=7, loc="upper right")
    ax_h.text(COM_384, ax_h.get_ylim()[1] * 0.9 if ax_h.get_ylim()[1] else 1, "3.84", ha="center", fontsize=7, color="#1b9e77")
    ax_h.text(COM_A, ax_h.get_ylim()[1] * 0.9 if ax_h.get_ylim()[1] else 1, "6.29", ha="center", fontsize=7, color="#1b9e77")

    labels, f384, f629, nbr = [], [], [], []
    for spec, sm in loaded:
        labels.append(spec["label"].split("(")[0].strip())
        nseq = max(sm["n_seq"], 1)
        f384.append(100 * sm["frac_seq_384"])
        f629.append(100 * sm["frac_seq_629"])
        nbr.append(sm["n_ox_bridge"])
    x = np.arange(len(labels))
    w = 0.36
    ax_b.bar(x - w / 2, f629, w, color="#4c78a8", label="seq. 6.29 Å")
    ax_b.bar(x + w / 2, f384, w, color="#c9a227", label="seq. 3.84 Å")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, rotation=18, ha="right")
    ax_b.set_ylabel("% of sequential pairs")
    ax_b.set_title("2. COM spacings along P")
    ax_b.legend(frameon=False, fontsize=7)

    # Unroll: honest gel phosphate Ca only (if present)
    honest = next((sm for spec, sm in loaded if spec["name"] == "honest"), None)
    gel_pdb = MODELS[0]["pdb"]
    if honest and honest["n_p_ca"] and gel_pdb.exists():
        from grow_whewellite import parse_atoms  # noqa: E402
        from grow_crystal_from_growth import dna_slab_frame  # noqa: E402

        atoms, _ = parse_atoms(gel_pdb)
        nuc = [a for a in atoms if a["resname"] == "NUC"]
        pts, _ = load_whw_ca(gel_pdb)
        origin, axis, zmin, zmax, _r = dna_slab_frame(nuc, pad=0.5)
        pxyz = np.array(
            [a["xyz"] for a in nuc if a["name"].strip().upper() == "P" or a["element"].upper() == "P"]
        )
        if len(pxyz):
            p_rel = pxyz - origin
            p_z = p_rel @ axis
            p_xy = p_rel - p_z[:, None] * axis
            # arbitrary azimuth basis
            ex = p_xy[0]
            ex = ex / max(float(np.linalg.norm(ex)), 1e-8)
            ey = np.cross(axis, ex)
            p_phi = np.degrees(np.arctan2(p_xy @ ey, p_xy @ ex))
            ax_u.scatter(p_phi, p_z, s=22, c="#c45c26", label="P", zorder=3)
        rel = pts - origin
        z = rel @ axis
        xy = rel - z[:, None] * axis
        ex = xy[np.argmax(np.linalg.norm(xy, axis=1))]
        ex = ex / max(float(np.linalg.norm(ex)), 1e-8)
        ey = np.cross(axis, ex)
        phi = np.degrees(np.arctan2(xy @ ey, xy @ ex))
        pb = honest["p_bound"].astype(bool)
        ax_u.scatter(
            phi[pb],
            z[pb],
            c=honest["hit_629"][pb] + 2 * honest["hit_384"][pb],
            s=28,
            cmap="cividis",
            vmin=0,
            vmax=3,
            zorder=4,
            label="P-bound Ca",
        )
        ax_u.set_xlabel("Azimuth (deg)")
        ax_u.set_ylabel("Axial (Å)")
        ax_u.set_title("3. Honest gel: P vs P-bound Ca")
        ax_u.legend(frameon=False, fontsize=7, loc="upper right")
    else:
        ax_u.text(0.5, 0.5, "Honest gel missing", ha="center", va="center", transform=ax_u.transAxes)
        ax_u.set_axis_off()

    fig.suptitle("Phosphate templating test — strand order, not coat pair-corr", fontsize=13, y=1.03)
    fig.text(
        0.5,
        -0.04,
        "Honest gel: thin CaOx at P, FIRE --no-com-targets, gel unfrozen. "
        "Alt-P: planted COM lattice + COM springs (positive control). "
        "No-DNA: same unit count, random sphere.",
        ha="center",
        fontsize=8,
        color="#666",
    )
    matplotlib_config.savefig(OUT)
    print(f"Wrote {OUT}")

    lines = ["Phosphate templating — strand Ca–Ca", "=" * 48, ""]
    for spec, sm in loaded:
        lines.append(spec["label"])
        lines.append(f"  n_Ca={sm['n_ca']}  P-bound={sm['n_p_ca']}  seq n={sm['n_seq']}")
        med = f"{sm['median_seq']:.2f}" if sm["median_seq"] is not None else "nan"
        lines.append(
            f"  median seq Ca–Ca={med} Å   "
            f"6.29 hits={sm['n_seq_629']}  3.84 hits={sm['n_seq_384']}"
        )
        lines.append(
            f"  P-layer pairs 3.84={sm['n_pair_384']}  6.29={sm['n_pair_629']}  "
            f"oxalate bridges={sm['n_ox_bridge']}"
        )
        lines.append("")
    TXT.write_text("\n".join(lines))
    print(f"Wrote {TXT}")


if __name__ == "__main__":
    main()
