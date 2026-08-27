#!/usr/bin/env python3
"""
Phosphate-templating experiment (FIRE, not MD):

  1. Gel at every P + disordered second+third-row CaOx + extra waters
     (strand Ca-Ca may be 3.84 A; not a 30 A coat)
  2. Honest FIRE (--no-com-targets, gel unfrozen)
  3. Density-matched no-DNA blob + honest FIRE
  4. Score strand Ca–Ca vs planted alt-P 30 Å (existing positive control)
  5. Plot

Does not pack a 30 Å shell and does not freeze the gel.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
VENV = ROOT / ".venv" / "bin" / "python"
FIRE_PY = str(VENV) if VENV.exists() else PY


def run(cmd: list[str]):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    gel = ROOT / "DNA_CaOx_templating_gel.pdb"
    gel_seeds = ROOT / "DNA_CaOx_templating_gel_seeds.pdb"
    gel_omm = ROOT / "DNA_CaOx_templating_gel_omm.pdb"
    nodna = ROOT / "DNA_CaOx_templating_gel_nodna.pdb"
    nodna_omm = ROOT / "DNA_CaOx_templating_gel_nodna_omm.pdb"

    run(
        [
            PY,
            "scripts/build_gel_first.py",
            "--templating",
            "--orient",
            "geometry",
        ]
    )
    run(
        [
            PY,
            "scripts/build_gel_first.py",
            "--templating",
            "--alt-p",
            "--orient",
            "geometry",
        ]
    )

    fire_common = [
        FIRE_PY,
        "scripts/fire_openmm_caox.py",
        "--no-com-targets",
        "--freeze-beyond",
        "0",
        "--steps",
        "180",
        "--max-move",
        "0.14",
    ]
    run(fire_common + [str(gel), "-o", str(gel_omm)])
    run(
        [
            PY,
            "scripts/build_nodna_blob.py",
            "--gel",
            str(gel_omm if gel_omm.exists() else gel),
            "-o",
            str(nodna),
        ]
    )
    run(fire_common + [str(nodna), "-o", str(nodna_omm)])

    for pdb, seeds, tag in (
        (gel_omm, gel_seeds, "DNA_CaOx_templating_gel_omm"),
        (nodna_omm, nodna, "DNA_CaOx_templating_gel_nodna_omm"),
    ):
        if not pdb.exists():
            print(f"skip score {pdb.name}: missing", flush=True)
            continue
        cmd = [
            PY,
            "scripts/find_symmetry.py",
            "--pdb",
            str(pdb),
            "--tag",
            tag,
            "--skip-sweep",
        ]
        if seeds.exists():
            cmd.extend(["--seeds", str(seeds)])
        run(cmd)

    run([PY, "scripts/plot_strand_order.py"])
    print("\nDone. See figures/crystallinity/DNA_CaOx_templating_strand_order.png")


if __name__ == "__main__":
    main()
