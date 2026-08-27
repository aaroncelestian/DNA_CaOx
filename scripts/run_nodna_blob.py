#!/usr/bin/env python3
"""
No-DNA CaOx blob: NVT MD from existing FIRE/OMM coordinates (no rebuild).

  .venv/bin/python scripts/run_nodna_blob.py --md-ps 50
  .venv/bin/python scripts/run_nodna_blob.py --md-ps 50 --md-only
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
    total_steps = int(round(ns * 1000.0 / 0.001))
    return max(1, total_steps // max(1, target_frames))


def main():
    ap = argparse.ArgumentParser(description="No-DNA blob MD from existing *_omm.pdb")
    ap.add_argument("--md-ps", type=float, default=None, help="NVT MD length (ps)")
    ap.add_argument("--md-ns", type=float, default=None, help="NVT MD length (ns)")
    ap.add_argument("--md-temperature", type=float, default=350.0, help="NVT temperature (K)")
    ap.add_argument("--md-traj-frames", type=int, default=400, help="Target MD trajectory frames")
    ap.add_argument(
        "--md-only",
        action="store_true",
        help="MD only from existing *_omm.pdb (default when input exists)",
    )
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    if args.md_ns is not None:
        md_ns = args.md_ns
    elif args.md_ps is not None:
        md_ns = args.md_ps / 1000.0
    else:
        md_ns = 0.05

    tag = "templating_gel_nodna"
    nodna_omm = ROOT / f"DNA_CaOx_{tag}_omm.pdb"
    traj = ROOT / "viewer" / "trajectories" / f"DNA_CaOx_{tag}_fire.trj.json"
    md_interval = md_traj_interval(md_ns, args.md_traj_frames)

    if not nodna_omm.exists():
        raise SystemExit(f"Missing {nodna_omm.name}; build the blob first.")

    fire_cmd = [
        FIRE_PY,
        "scripts/fire_openmm_caox.py",
        str(nodna_omm),
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
        str(nodna_omm),
        "--md-only",
        "--steps",
        "0",
        "--skip-openmm",
        "--traj-include-openmm",
    ]
    print(
        f"MD plan: {md_ns * 1000:.1f} ps ({md_ns:.6f} ns) at {args.md_temperature:.0f} K, "
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
                str(nodna_omm),
                "--tag",
                f"DNA_CaOx_{tag}_omm",
                "--skip-sweep",
            ]
        )

    if not args.skip_export:
        run([PY, "scripts/export_viewer_data.py", "--geom", "templating_nodna"])

    print("\nDone.", flush=True)
    print(f"  Model : {nodna_omm}", flush=True)
    print(f"  Traj  : {traj}", flush=True)
    print("  Viewer: load 'templating_nodna' after hard refresh", flush=True)


if __name__ == "__main__":
    main()
