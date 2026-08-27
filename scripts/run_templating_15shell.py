#!/usr/bin/env python3
"""
15-row templating gel: build → honest FIRE → OpenMM → NVT MD → score → viewer.

  .venv/bin/python scripts/run_templating_15shell.py
  .venv/bin/python scripts/run_templating_15shell.py --md-ns 0.5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
VENV_PY = ROOT / ".venv" / "bin" / "python"
FIRE_PY = str(VENV_PY) if VENV_PY.exists() else PY


def run(cmd: list[str], **kw):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, **kw)


def main():
    ap = argparse.ArgumentParser(description="15-row templating gel + MD nucleation test")
    ap.add_argument("--md-ns", type=float, default=0.1, help="NVT MD length (ns)")
    ap.add_argument("--fire-steps", type=int, default=200)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-fire", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    tag = "templating_gel_15shell"
    gel_pdb = ROOT / f"DNA_CaOx_{tag}.pdb"
    gel_omm = ROOT / f"DNA_CaOx_{tag}_omm.pdb"
    gel_seeds = ROOT / f"DNA_CaOx_{tag}_seeds.pdb"
    traj = ROOT / "viewer" / "trajectories" / f"DNA_CaOx_{tag}_fire.trj.json"

    if not args.skip_build:
        run(
            [
                PY,
                "scripts/build_gel_first.py",
                "--templating-15shell",
                "--orient",
                "geometry",
            ]
        )

    if not args.skip_fire:
        run(
            [
                FIRE_PY,
                "scripts/fire_openmm_caox.py",
                str(gel_pdb),
                "--no-com-targets",
                "--freeze-beyond",
                "0",
                "--steps",
                str(args.fire_steps),
                "--max-move",
                "0.14",
                "--traj",
                str(traj),
                "--traj-interval",
                "20",
                "--md-ns",
                str(args.md_ns),
                "--md-traj-interval",
                "1000",
                "-o",
                str(gel_omm),
            ]
        )

    if not args.skip_score:
        run(
            [
                PY,
                "scripts/find_symmetry.py",
                "--pdb",
                str(gel_omm),
                "--seeds",
                str(gel_seeds),
                "--tag",
                f"DNA_CaOx_{tag}_omm",
                "--skip-sweep",
            ]
        )

    if not args.skip_export:
        run([PY, "scripts/export_viewer_data.py", "--geom", tag])

    print("\nDone.", flush=True)
    print(f"  Model : {gel_omm}", flush=True)
    print(f"  Traj  : {traj}", flush=True)
    print(f"  Viewer: load '{tag}' after hard refresh", flush=True)


if __name__ == "__main__":
    main()
