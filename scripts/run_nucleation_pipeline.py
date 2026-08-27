#!/usr/bin/env python3
"""
Run the gel alt-P nucleation experiment end-to-end:

  1. build_gel_first --alt-p --orient geometry
  2. FIRE/OMM honest gel (--no-com-targets)
  3. find_symmetry on gel
  4. build_gel_shell --mode lattice --saturation saturated (no crystal seed)
  5. FIRE/OMM shell (--gel-outward-com, freeze gel)
  6. find_symmetry on shell
  7. export_viewer_data
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
FIRE_PY = str(ANACONDA) if (ANACONDA := Path("/opt/anaconda3/bin/python")).exists() else PY
# scipy / numpy-heavy steps
SCI_PY = FIRE_PY


def run(cmd: list[str], **kw):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, **kw)


def main():
    tag = "gel_altP_geom"
    gel_pdb = ROOT / f"DNA_CaOx_{tag}.pdb"
    gel_omm = ROOT / f"DNA_CaOx_{tag}_omm.pdb"
    gel_seeds = ROOT / f"DNA_CaOx_{tag}_seeds.pdb"
    shell_pdb = ROOT / f"DNA_CaOx_{tag}_shell_lattice.pdb"
    shell_omm = ROOT / f"DNA_CaOx_{tag}_shell_lattice_omm.pdb"
    shell_seeded_pdb = ROOT / f"DNA_CaOx_{tag}_shell_lattice_seeded.pdb"
    shell_seeded_omm = ROOT / f"DNA_CaOx_{tag}_shell_lattice_seeded_omm.pdb"
    use_seed = "--crystal-seed" in sys.argv
    fire_py = FIRE_PY

    run(
        [
            PY,
            "scripts/build_gel_first.py",
            "--alt-p",
            "--orient",
            "geometry",
        ]
    )

    run(
        [
            fire_py,
            "scripts/fire_openmm_caox.py",
            str(gel_pdb),
            "--no-com-targets",
            "--steps",
            "80",
            "--traj",
            str(ROOT / "viewer" / "trajectories" / f"DNA_CaOx_{tag}_fire.trj.json"),
        ]
    )

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
        ]
    )

    shell_build_cmd = [
        SCI_PY,
        "scripts/build_gel_shell.py",
        "--gel",
        str(gel_omm),
        "--mode",
        "lattice",
        "--saturation",
        "saturated",
    ]
    if use_seed:
        shell_build_cmd.append("--crystal-seed")
        print("Note: --crystal-seed is optional; default is saturated shell without seed.")
    run(shell_build_cmd)

    shell_input = shell_seeded_pdb if use_seed else shell_pdb
    shell_output = shell_seeded_omm if use_seed else shell_omm
    shell_tag = (
        f"DNA_CaOx_{tag}_shell_lattice_seeded_omm"
        if use_seed
        else f"DNA_CaOx_{tag}_shell_lattice_omm"
    )
    traj_name = (
        f"DNA_CaOx_{tag}_shell_lattice_seeded_fire.trj.json"
        if use_seed
        else f"DNA_CaOx_{tag}_shell_lattice_fire.trj.json"
    )

    # Infer gel_max from shell report or gel_omm
    from grow_whewellite import parse_atoms

    atoms, _ = parse_atoms(gel_omm)
    gel_max = max(
        a["resseq"] for a in atoms if a["resname"] in ("WHW", "COM")
    )

    fire_cmd = [
            fire_py,
            "scripts/fire_openmm_caox.py",
            str(shell_input),
            "--freeze-resseq-le",
            str(gel_max),
            "--com-min-resseq",
            str(gel_max),
            "--com-ramp-steps",
            "250",
            "--rotation-free-steps",
            "150",
            "--steps",
            "2500",
            "--max-move",
            "0.16",
            "--traj-interval",
            "50",
            "--shell-pos-weight",
            "0.09",
            "--shell-max-shift",
            "8.5",
            "--w-gel-com",
            "22",
            "--gel-com-decay-len",
            "7",
            "--shell-com-min-dgel",
            "12",
            "--traj",
            str(ROOT / "viewer" / "trajectories" / traj_name),
    ]
    if use_seed:
        fire_cmd.append("--seed-epitax")
    fire_cmd.append("--gel-outward-com")
    run(fire_cmd)

    run(
        [
            PY,
            "scripts/find_symmetry.py",
            "--pdb",
            str(shell_output),
            "--seeds",
            str(gel_seeds),
            "--tag",
            shell_tag,
        ]
    )

    run([SCI_PY, "scripts/export_viewer_data.py"])
    print(
        "\nDone. Open viewer → "
        + ("Seeded shell" if use_seed else "Lattice shell")
        + "; scrub FIRE trajectory."
    )


if __name__ == "__main__":
    main()
