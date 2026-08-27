"""Shared matplotlib defaults for DNA_CaOx figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.figsize": (8, 5),
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def savefig(path, tight=True):
    if tight:
        plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
