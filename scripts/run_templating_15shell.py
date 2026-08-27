#!/usr/bin/env python3
"""
15-row templating gel: build → honest FIRE → OpenMM → NVT MD → score → viewer.

  .venv/bin/python scripts/run_templating_15shell.py
  .venv/bin/python scripts/run_templating_15shell.py --md-ns 0.5
  .venv/bin/python scripts/run_templating_15shell.py --md-only --md-us 2 --md-temperature 350
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


def md_traj_interval(ns: float, target_frames: int = 400) -> int:
    """Integration steps between trajectory frames (1 fs timestep)."""
    total_steps = int(round(ns * 1000.0 / 0.001))  # ns → ps → steps
    return max(1, total_steps // max(1, target_frames))


def main():
    ap = argparse.ArgumentParser(description="15-row templating gel + MD nucleation test")
    ap.add_argument("--md-ns", type=float, default=None, help="NVT MD length (ns)")
    ap.add_argument("--md-us", type=float, default=None, help="NVT MD length (µs); overrides --md-ns")
    ap.add_argument("--md-temperature", type=float, default=350.0, help="NVT temperature (K)")
    ap.add_argument("--md-traj-frames", type=int, default=400, help="Target MD trajectory frames")
    ap.add_argument("--fire-steps", type=int, default=200)
    ap.add_argument("--md-only", action="store_true", help="Continue MD from existing *_omm.pdb")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-fire", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    if args.md_us is not None:
        md_ns = args.md_us * 1000.0
    elif args.md_ns is not None:
        md_ns = args.md_ns
    else:
        md_ns = 0.1

    tag = "templating_gel_15shell"
    gel_pdb = ROOT / f"DNA_CaOx_{tag}.pdb"
    gel_omm = ROOT / f"DNA_CaOx_{tag}_omm.pdb"
    gel_seeds = ROOT / f"DNA_CaOx_{tag}_seeds.pdb"
    traj = ROOT / "viewer" / "trajectories" / f"DNA_CaOx_{tag}_fire.trj.json"
    md_interval = md_traj_interval(md_ns, args.md_traj_frames)

    if args.md_only:
        args.skip_build = True
        if not gel_omm.exists():
            raise SystemExit(f"Missing {gel_omm.name}; run a full build first.")

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
        fire_input = gel_omm if args.md_only else gel_pdb
        fire_cmd = [
            FIRE_PY,
            "scripts/fire_openmm_caox.py",
            str(fire_input),
            "--no-com-targets",
            "--freeze-beyond",
            "0",
            "--max-move",
            "0.14",
            "--traj",
            str(traj),
            "--md-ns",
            str(md_ns),
            "--md-temperature",
            str(args.md_temperature),
            "--md-traj-interval",
            str(md_interval),
            "-o",
            str(gel_omm),
        ]
        if args.md_only:
            fire_cmd += ["--md-only", "--steps", "0", "--skip-openmm"]
        else:
            fire_cmd += ["--steps", str(args.fire_steps), "--traj-interval", "20"]
        print(
            f"MD plan: {md_ns:.1f} ns ({md_ns/1000:.3f} µs) at {args.md_temperature:.0f} K, "
            f"~{args.md_traj_frames} traj frames (interval {md_interval} steps)",
            flush=True,
        )
        run(fire_cmd)

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
