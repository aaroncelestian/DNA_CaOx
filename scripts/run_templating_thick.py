#!/usr/bin/env python3
"""
5-row (thick) templating gel: MD from relaxed OMM → score → viewer.

  .venv/bin/python scripts/run_templating_thick.py --md-only --md-ns 100 --md-temperature 350
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
STEPS_PER_NS = 1_000_000  # 1 fs timestep
STEPS_PER_SEC_EST = 300  # conservative CPU estimate for thick gel


def run(cmd: list[str], **kw):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, **kw)


def md_traj_interval(ns: float, target_frames: int = 200) -> int:
    total_steps = int(round(ns * STEPS_PER_NS))
    return max(1, total_steps // max(1, target_frames))


def main():
    ap = argparse.ArgumentParser(description="Thick (5-row) templating gel MD")
    ap.add_argument("--md-ns", type=float, default=100.0, help="NVT MD length (ns)")
    ap.add_argument("--md-temperature", type=float, default=350.0, help="NVT temperature (K)")
    ap.add_argument("--md-traj-frames", type=int, default=200, help="Target MD trajectory frames")
    ap.add_argument("--md-only", action="store_true", help="MD from existing *_omm.pdb")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    tag = "templating_gel_thick"
    gel_omm = ROOT / f"DNA_CaOx_{tag}_omm.pdb"
    gel_seeds = ROOT / f"DNA_CaOx_{tag}_seeds.pdb"
    traj = ROOT / "viewer" / "trajectories" / f"DNA_CaOx_{tag}_fire.trj.json"
    md_interval = md_traj_interval(args.md_ns, args.md_traj_frames)
    est_h = args.md_ns * STEPS_PER_NS / STEPS_PER_SEC_EST / 3600

    if not gel_omm.exists():
        raise SystemExit(f"Missing {gel_omm.name}; build and relax thick gel first.")

    print(
        f"MD plan: {args.md_ns:.1f} ns at {args.md_temperature:.0f} K, "
        f"~{args.md_traj_frames} frames (every {md_interval} steps ≈ "
        f"{md_interval / STEPS_PER_NS * 1000:.2f} ps), "
        f"est. wall ~{est_h:.1f} h on CPU",
        flush=True,
    )

    run(
        [
            FIRE_PY,
            "scripts/fire_openmm_caox.py",
            str(gel_omm),
            "--md-only",
            "--no-com-targets",
            "--freeze-beyond",
            "0",
            "--steps",
            "0",
            "--skip-openmm",
            "--traj",
            str(traj),
            "--md-ns",
            str(args.md_ns),
            "--md-temperature",
            str(args.md_temperature),
            "--md-traj-interval",
            str(md_interval),
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


if __name__ == "__main__":
    main()
