#!/usr/bin/env python3
"""
FIRE (rigid oxalate) then OpenMM minimization on CaOx / whewellite.

DNA is copied unchanged. WHW oxalate groups (C2O4) are rigid bodies;
Ca and water O are free. The restraint potential matches dls_caox.py:

  * Ca–O coordination to the starting ligands
  * one-sided O···O ≥ OO_TARGET
  * one-sided Ca···Ca ≥ CA_MIN, plus soft COM Ca–Ca targets
  * positional anchors and WHW–DNA exclusion

After FIRE, OpenMM LocalEnergyMinimizer relaxes the same potential
(stiff intra-oxalate bonds keep C2O4 nearly rigid). Run with the
project venv that has OpenMM:

  .venv/bin/python scripts/fire_openmm_caox.py DNA_CaOx_growth_whewellite10A_allP_relaxed.pdb
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dls_caox import (  # noqa: E402
    CA_MIN,
    COM_CA,
    DNA_HEAVY,
    EPITAX_RADIUS,
    MAX_SHIFT,
    MERGE_SAME,
    OO_CUT,
    OO_TARGET,
    W_CA,
    W_CAO,
    W_COM,
    W_DNA,
    W_EPITAX,
    W_GEL_COM,
    GEL_COM_DECAY_LEN,
    SHELL_COM_MIN_DGEL,
    W_INTRA,
    W_OO,
    W_POS,
    build_tables,
    el_of,
    energy_grad,
    find_oxalates,
    inter_oo_stats,
    is_seed_atom,
    merge_duplicate_atoms,
    query_pairs,
)
from geom_constraints import is_ca, is_oxygen  # noqa: E402
from grow_whewellite import parse_atoms  # noqa: E402
from relax_whewellite_units import write_pdb  # noqa: E402
from helix_frame import (  # noqa: E402
    basis_from_dna,
    ca_positions_helix,
    oxalate_segments_helix,
)

ROOT = Path(__file__).resolve().parents[1]
CAOX_RESNAMES = frozenset({"WHW", "COM"})
OO_SKIN = 0.50
OMM_W_INTRA = 400.0
NM = 0.1  # Å → nm


def default_pdb() -> Path:
    for name in (
        "DNA_CaOx_growth_whewellite10A_allP_relaxed.pdb",
        "DNA_CaOx_growth_whewellite10A_allP.pdb",
        "DNA_CaOx_growth_whewellite30A.pdb",
    ):
        p = ROOT / name
        if p.exists():
            return p
    return ROOT / "DNA_CaOx_growth_whewellite10A_allP.pdb"


def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def q_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def q_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    w = float(np.linalg.norm(omega))
    if w < 1e-16:
        return q
    half = 0.5 * w * dt
    dq = np.array([math.cos(half), *(math.sin(half) * omega / w)])
    out = q_mul(dq, q)
    n = float(np.linalg.norm(out))
    return out / n if n > 0 else q


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Rotation R and translation t with R @ Q_i + t ≈ P_i. Q is body-frame (COM 0)."""
    cP = P.mean(axis=0)
    H = Q.T @ (P - cP)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, cP


def oo_pair_list(xyz, o_idx, gid, cutoff: float):
    if len(o_idx) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    pairs = query_pairs(xyz[o_idx], cutoff)
    if len(pairs) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    a = o_idx[pairs[:, 0]]
    b = o_idx[pairs[:, 1]]
    keep = gid[a] != gid[b]
    return a[keep], b[keep]


class RigidBody:
    __slots__ = ("idx", "rel", "com", "com0", "q", "I", "M", "frozen")

    def __init__(self, idx: np.ndarray, xyz: np.ndarray):
        self.idx = np.asarray(idx, int)
        pts = xyz[self.idx]
        self.com = pts.mean(axis=0).copy()
        self.com0 = self.com.copy()
        self.rel = pts - self.com
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        rg2 = float(np.square(self.rel).sum(axis=1).mean()) if len(self.rel) else 1.0
        self.I = max(0.8, float(len(self.idx)) * rg2)
        self.M = float(len(self.idx))
        self.frozen = False


def build_bodies(oxalates, xyz):
    return [RigidBody(np.array(g, int), xyz) for g in oxalates if len(g) >= 2]


def apply_bodies(xyz, bodies):
    for b in bodies:
        if b.frozen:
            continue
        xyz[b.idx] = (q_to_R(b.q) @ b.rel.T).T + b.com
    return xyz


def rigidify_xyz(xyz, bodies):
    for b in bodies:
        if b.frozen or len(b.idx) < 2:
            continue
        R, t = kabsch(xyz[b.idx], b.rel)
        xyz[b.idx] = (R @ b.rel.T).T + t
        b.com = t.copy()
        # store R as quaternion
        b.q = rot_to_q(R)
    return xyz


def rot_to_q(R: np.ndarray) -> np.ndarray:
    tr = float(np.trace(R))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array(
            [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        )
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array(
            [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        )
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array(
            [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
        )
    n = float(np.linalg.norm(q))
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def reduce_forces(F, xyz, bodies):
    """Atomic forces → COM force and world-frame torque; rewrite oxalate atom forces."""
    Fcom = np.zeros((len(bodies), 3))
    tau = np.zeros((len(bodies), 3))
    for k, b in enumerate(bodies):
        f = F[b.idx]
        Fcom[k] = f.sum(axis=0)
        r = xyz[b.idx] - b.com
        tau[k] = np.cross(r, f).sum(axis=0)
        if b.frozen:
            Fcom[k] = 0.0
            tau[k] = 0.0
            F[b.idx] = 0.0
        else:
            F[b.idx] = Fcom[k] / max(len(b.idx), 1) + np.cross(tau[k] / b.I, r)
    return Fcom, tau


def freeze_by_resseq(bodies, atoms, free_idx, max_resseq: int):
    """Freeze gel WHW units (oxalate bodies + free Ca/water) through resseq."""
    n_bodies = 0
    frozen = []
    for bdy in bodies:
        if all(atoms[i]["resseq"] <= max_resseq for i in bdy.idx):
            bdy.frozen = True
            n_bodies += 1
    for i in free_idx:
        if atoms[i]["resseq"] <= max_resseq:
            frozen.append(int(i))
    return n_bodies, np.array(frozen, int)


def freeze_crystal_seed(bodies, atoms, free_idx):
    """Freeze embedded whewellite seed (bfac ≈ SEED_CRYSTAL_BFAC)."""
    seed_atoms = {
        i for i, a in enumerate(atoms) if is_seed_atom(a)
    }
    if not seed_atoms:
        return 0, np.array([], int)
    n_bodies = 0
    for bdy in bodies:
        if any(int(i) in seed_atoms for i in bdy.idx):
            bdy.frozen = True
            n_bodies += 1
    frozen = [int(i) for i in free_idx if int(i) in seed_atoms]
    return n_bodies, np.array(frozen, int)


def mark_frozen(bodies, free_idx, xyz, dna_xyz, freeze_beyond: float, o_idx, gid):
    empty = np.zeros(0, int)
    if freeze_beyond <= 0 or len(dna_xyz) == 0:
        return 0, empty
    tree = cKDTree(dna_xyz)
    a, b = oo_pair_list(xyz, o_idx, gid, OO_CUT)
    clash = set()
    if len(a):
        clash.update(int(i) for i in a)
        clash.update(int(i) for i in b)
    n_fr = 0
    dmin, _ = tree.query(xyz, k=1)
    for bdy in bodies:
        if float(dmin[bdy.idx].min()) > freeze_beyond and not any(
            int(i) in clash for i in bdy.idx
        ):
            bdy.frozen = True
            n_fr += 1
    frozen_free = np.array(
        [int(i) for i in free_idx if float(dmin[i]) > freeze_beyond and int(i) not in clash],
        int,
    )
    return n_fr, frozen_free


def make_traj_frame(atoms, xyz, ca_idx, dna_xyz, step: int, energy=None, phase: str = "fire"):
    """One viewer trajectory frame (helix-frame Ca + oxalate sticks)."""
    helix_origin, helix_axis, helix_ex, helix_ey = basis_from_dna(dna_xyz)
    ca_atom_idx = [int(i) for i in ca_idx]
    whw_snap = []
    for i, a in enumerate(atoms):
        b = dict(a)
        b["xyz"] = np.asarray(xyz[i], float).copy()
        whw_snap.append(b)
    return {
        "step": int(step),
        "phase": str(phase),
        "ca": ca_positions_helix(
            whw_snap, ca_atom_idx, helix_origin, helix_ex, helix_axis, helix_ey
        ),
        "oxalate": oxalate_segments_helix(
            whw_snap, helix_origin, helix_ex, helix_axis, helix_ey
        ),
        "energy": energy,
    }


def run_fire(
    atoms,
    dna_xyz,
    steps: int,
    freeze_beyond: float,
    max_move: float,
    oxalates=None,
    ca_idx=None,
    water=None,
    com_targets: bool = True,
    freeze_resseq_le: int = 0,
    com_min_resseq: int = 0,
    com_ramp_steps: int = 0,
    rotation_free_steps: int = 0,
    shell_pos_weight: float = 0.18,
    shell_max_shift: float = 4.5,
    seed_epitax: bool = False,
    epitax_radius: float = EPITAX_RADIUS,
    w_epitax: float = W_EPITAX,
    shell_epitax_pos_weight: float = 0.06,
    shell_epitax_max_shift: float = 10.0,
    seed_epitax_outward_only: bool = True,
    gel_outward_com: bool = False,
    w_gel_com: float = W_GEL_COM,
    gel_com_decay_len: float = GEL_COM_DECAY_LEN,
    shell_com_min_dgel: float = SHELL_COM_MIN_DGEL,
    traj_path: Path | None = None,
    traj_interval: int = 10,
    traj_out: list | None = None,
    step_offset: int = 0,
    traj_phase: str = "fire",
):
    if oxalates is None:
        oxalates, ca_idx, water = find_oxalates(atoms)
    xyz = np.array([a["xyz"] for a in atoms], float)
    o_idx = np.array([i for i, a in enumerate(atoms) if is_oxygen(a)], int)
    gid = np.full(len(atoms), -1, int)
    for gi, grp in enumerate(oxalates):
        for i in grp:
            gid[i] = gi
    w0 = len(oxalates)
    for k, i in enumerate(water):
        gid[i] = w0 + k

    n0, min0 = inter_oo_stats(xyz, o_idx, gid)
    ox_atoms = set()
    for g in oxalates:
        ox_atoms.update(g)
    free_idx = np.array(
        [i for i in range(len(atoms)) if i not in ox_atoms], int
    )
    bodies = build_bodies(oxalates, xyz)
    frozen_free = np.array([], int)
    n_body_fr = 0
    n_gel_fr = 0
    n_seed_fr = 0
    if freeze_resseq_le > 0:
        n_gel_fr, frozen_free = freeze_by_resseq(
            bodies, atoms, free_idx, freeze_resseq_le
        )
    if seed_epitax:
        n_seed_fr, frozen_seed = freeze_crystal_seed(bodies, atoms, free_idx)
        frozen_free = np.unique(
            np.concatenate([frozen_free, frozen_seed])
        ).astype(int)
    if freeze_beyond > 0:
        n_body_fr, frozen_dist = mark_frozen(
            bodies, free_idx, xyz, dna_xyz, freeze_beyond, o_idx, gid
        )
        frozen_free = np.unique(np.concatenate([frozen_free, frozen_dist])).astype(int)
    frozen_free_set = set(frozen_free.tolist())

    tbl = build_tables(
        atoms,
        oxalates,
        ca_idx,
        water,
        dna_xyz,
        com_min_resseq=com_min_resseq,
        shell_pos_weight=shell_pos_weight,
        shell_max_shift=shell_max_shift,
        seed_epitax=seed_epitax,
        epitax_radius=epitax_radius,
        w_epitax=w_epitax,
        shell_epitax_pos_weight=shell_epitax_pos_weight,
        shell_epitax_max_shift=shell_epitax_max_shift,
        seed_epitax_outward_only=seed_epitax_outward_only,
        gel_outward_com=gel_outward_com,
        w_gel_com=w_gel_com,
        gel_com_decay_len=gel_com_decay_len,
        shell_com_min_dgel=shell_com_min_dgel,
    )
    if not com_targets:
        tbl["com"] = (
            np.array([], int),
            np.array([], int),
            np.array([], float),
            np.array([], float),
        )
    n_com = len(tbl["com"][0])
    n_epi = len(tbl["epitax"][0])
    print(
        f"FIRE start: {len(bodies)} rigid oxalate, {len(ca_idx)} Ca, {len(water)} water O, "
        f"frozen bodies={n_body_fr + n_gel_fr + n_seed_fr} "
        f"(gel≤{freeze_resseq_le}={n_gel_fr}, seed={n_seed_fr}), "
        f"COM pairs={n_com}"
        + (f" (resseq>{com_min_resseq})" if com_min_resseq else "")
        + (f", epitax pairs={n_epi}" if seed_epitax else "")
        + (", gel→shell COM" if gel_outward_com else "")
        + f", inter-group O-O < 2.0 Å: n={n0} min={min0}",
        flush=True,
    )

    record_traj = traj_path is not None or traj_out is not None
    traj_frames = traj_out if traj_out is not None else []
    max_shift_arr = tbl.get("max_shift", np.full(len(atoms), MAX_SHIFT, float))

    def record_frame(step: int, energy: float | None = None):
        if not record_traj:
            return
        traj_frames.append(
            make_traj_frame(
                atoms,
                xyz,
                ca_idx,
                dna_xyz,
                step_offset + int(step),
                energy,
                traj_phase,
            )
        )

    def restraint_scales(step: int):
        if rotation_free_steps > 0 and step <= rotation_free_steps:
            t = step / max(rotation_free_steps, 1)
            w_pos = 0.35 + 0.65 * t
        else:
            w_pos = 1.0
        if com_ramp_steps > 0:
            w_com = min(1.0, step / max(com_ramp_steps, 1))
        else:
            w_com = 1.0
        return w_com, w_pos

    v_free = np.zeros((len(free_idx), 3))
    v_com = np.zeros((len(bodies), 3))
    omega = np.zeros((len(bodies), 3))
    dt = 0.04
    dt_min = 0.012
    dt_max = 0.18
    alpha = 0.1
    alpha_start = 0.1
    n_pos = 0
    history = []
    oo_a, oo_b = oo_pair_list(xyz, o_idx, gid, OO_CUT + OO_SKIN)

    def energy_at(xcur, step=1):
        w_com, w_pos = restraint_scales(step)
        return energy_grad(
            xcur,
            tbl,
            oo_pairs=(oo_a, oo_b),
            skip_intra=True,
            w_com_scale=w_com,
            w_pos_scale=w_pos,
        )

    E, g = energy_at(xyz.ravel(), 0)
    n1, min1 = n0, min0
    record_frame(0, float(E))

    for it in range(1, steps + 1):
        if it == 1 or it % 8 == 0:
            oo_a, oo_b = oo_pair_list(xyz, o_idx, gid, OO_CUT + OO_SKIN)
        E, g = energy_at(xyz.ravel(), it)
        F = -g.reshape(-1, 3)
        if frozen_free_set:
            F[frozen_free] = 0.0
        Fcom, tau = reduce_forces(F, xyz, bodies)
        Ff = F[free_idx] if len(free_idx) else np.zeros((0, 3))

        bits = []
        if len(free_idx):
            bits.append(np.dot(Ff.ravel(), v_free.ravel()))
        bits.append(np.dot(Fcom.ravel(), v_com.ravel()))
        bits.append(np.dot(tau.ravel(), omega.ravel()))
        P = float(sum(bits))

        def mix(vel, force):
            nF = float(np.linalg.norm(force))
            nV = float(np.linalg.norm(vel))
            if nF < 1e-16 or nV < 1e-16:
                return vel
            return (1.0 - alpha) * vel + alpha * nV * force / nF

        if P > 0:
            if len(free_idx):
                v_free = mix(v_free, Ff)
            v_com = mix(v_com, Fcom)
            omega = mix(omega, tau)
            n_pos += 1
            if n_pos > 5:
                dt = min(dt * 1.1, dt_max)
                alpha *= 0.99
        else:
            v_free[:] = 0.0
            v_com[:] = 0.0
            omega[:] = 0.0
            alpha = alpha_start
            dt = max(dt * 0.5, dt_min)
            n_pos = 0

        if len(free_idx):
            v_free += dt * Ff
        for k, b in enumerate(bodies):
            if b.frozen:
                v_com[k] = 0.0
                omega[k] = 0.0
                continue
            v_com[k] += dt * Fcom[k] / b.M
            omega[k] += dt * tau[k] / b.I

        # limit Cartesian / angular step
        if len(free_idx):
            step_f = dt * v_free
            m = np.linalg.norm(step_f, axis=1).max() if len(step_f) else 0.0
            if m > max_move:
                scale = max_move / m
                step_f *= scale
                v_free *= scale
            xyz[free_idx] += step_f
            d0 = xyz[free_idx] - tbl["xyz0"][free_idx]
            nrm = np.linalg.norm(d0, axis=1)
            lim = max_shift_arr[free_idx]
            over = nrm > lim
            if np.any(over):
                xyz[free_idx[over]] = (
                    tbl["xyz0"][free_idx[over]]
                    + d0[over] * (lim[over] / nrm[over])[:, None]
                )
                v_free[over] = 0.0
            if len(frozen_free):
                xyz[frozen_free] = tbl["xyz0"][frozen_free]

        for k, b in enumerate(bodies):
            if b.frozen:
                continue
            ds = dt * v_com[k]
            sn = float(np.linalg.norm(ds))
            if sn > max_move:
                ds *= max_move / sn
                v_com[k] *= max_move / sn
            b.com = b.com + ds
            dc = b.com - b.com0
            cn = float(np.linalg.norm(dc))
            body_lim = float(max_shift_arr[b.idx].max()) if len(b.idx) else MAX_SHIFT
            if cn > body_lim:
                b.com = b.com0 + dc * (body_lim / cn)
                v_com[k] = 0.0
            w = float(np.linalg.norm(omega[k]))
            if w * dt > 0.25:
                omega[k] *= 0.25 / (w * dt)
            b.q = q_integrate(b.q, omega[k], dt)
        apply_bodies(xyz, bodies)

        if record_traj and (
            it % max(traj_interval, 1) == 0 or it == steps
        ) and not (step_offset > 0 and it == 0):
            record_frame(it, float(E))

        if it == 1 or it % 20 == 0 or it == steps:
            n1, min1 = inter_oo_stats(xyz, o_idx, gid)
            fmax = float(np.linalg.norm(F, axis=1).max()) if len(F) else 0.0
            print(
                f"  FIRE {it:4d}  E={E:.3e}  |F|max={fmax:.2f}  dt={dt:.3f}  "
                f"O-O<2Å n={n1} min={min1}  P={P:.2e}",
                flush=True,
            )
            history.append((it, float(E), n1 if n1 is not None else 0, fmax))
            if n1 == 0 and it > 20 and steps <= 100:
                print("  FIRE converged (no inter-group O-O < 2 Å)", flush=True)
                break

    apply_bodies(xyz, bodies)
    for i, a in enumerate(atoms):
        a["xyz"] = xyz[i]
    n1, min1 = inter_oo_stats(xyz, o_idx, gid)

    if traj_path is not None and traj_out is None and traj_frames:
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj_path.write_text(
            json.dumps(
                {
                    "source": str(traj_path.stem),
                    "n_ca": len(ca_idx),
                    "frames": traj_frames,
                },
                separators=(",", ":"),
            )
        )
        print(f"Wrote trajectory {traj_path} ({len(traj_frames)} frames)", flush=True)

    return {
        "n_oxalate": len(oxalates),
        "n_ca": len(ca_idx),
        "n_water": len(water),
        "n_atoms": len(atoms),
        "n_frozen_bodies": n_body_fr,
        "n_oo_before": n0,
        "min_oo_before": min0,
        "n_oo_after": n1,
        "min_oo_after": min1,
        "history": history,
        "steps_run": history[-1][0] if history else 0,
        "oxalates": oxalates,
        "ca_idx": ca_idx,
        "water": water,
        "bodies": bodies,
        "tbl": tbl,
        "traj_frames": traj_frames,
    }


def write_traj_json(traj_path: Path, n_ca: int, frames: list):
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.write_text(
        json.dumps(
            {"source": str(traj_path.stem), "n_ca": int(n_ca), "frames": frames},
            separators=(",", ":"),
        )
    )
    print(f"Wrote trajectory {traj_path} ({len(frames)} frames)", flush=True)


def import_openmm():
    try:
        import openmm
        from openmm import (
            Context,
            CustomBondForce,
            CustomExternalForce,
            CustomNonbondedForce,
            LocalEnergyMinimizer,
            Platform,
            System,
            VerletIntegrator,
        )
        from openmm.unit import dalton, kilojoule_per_mole, nanometer, picosecond

        return {
            "openmm": openmm,
            "Context": Context,
            "CustomBondForce": CustomBondForce,
            "CustomExternalForce": CustomExternalForce,
            "CustomNonbondedForce": CustomNonbondedForce,
            "LocalEnergyMinimizer": LocalEnergyMinimizer,
            "LangevinMiddleIntegrator": openmm.LangevinMiddleIntegrator,
            "Platform": Platform,
            "System": System,
            "VerletIntegrator": VerletIntegrator,
            "dalton": dalton,
            "kilojoule_per_mole": kilojoule_per_mole,
            "nanometer": nanometer,
            "picosecond": picosecond,
            "kelvin": openmm.unit.kelvin,
        }
    except ImportError as exc:
        return None, str(exc)


def pick_platform(omm, name: str | None):
    Platform = omm["Platform"]
    available = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
    if name:
        if name not in available:
            raise SystemExit(f"OpenMM platform {name!r} not in {available}")
        return Platform.getPlatformByName(name)
    for pref in ("CPU", "OpenCL", "CUDA", "Reference"):
        if pref in available:
            return Platform.getPlatformByName(pref)
    return Platform.getPlatform(0)


def run_openmm(whw, dna_xyz, tbl, bodies, max_iter: int, platform_name: str | None, freeze_resseq_le: int = 0, seed_epitax: bool = False):
    loaded = import_openmm()
    if not isinstance(loaded, dict):
        err = loaded[1] if isinstance(loaded, tuple) else "unknown import error"
        raise RuntimeError(
            "OpenMM is not installed. Use the project venv:\n"
            "  .venv/bin/python scripts/fire_openmm_caox.py ...\n"
            f"Import error: {err}"
        )
    omm = loaded
    xyz = np.array([a["xyz"] for a in whw], float)
    n = len(whw)
    n_dna = len(dna_xyz)
    n_all = n + n_dna

    System = omm["System"]
    CustomBondForce = omm["CustomBondForce"]
    CustomExternalForce = omm["CustomExternalForce"]
    CustomNonbondedForce = omm["CustomNonbondedForce"]
    nanometer = omm["nanometer"]
    dalton = omm["dalton"]
    kilojoule_per_mole = omm["kilojoule_per_mole"]
    picosecond = omm["picosecond"]

    system = System()
    # WHW first, then frozen DNA (mass 0).
    k_dna = 1.0e7
    for a in whw:
        el = el_of(a)
        if freeze_resseq_le > 0 and int(a.get("resseq", 9999)) <= freeze_resseq_le:
            mass = 0.0
        elif seed_epitax and is_seed_atom(a):
            mass = 0.0
        else:
            mass = 40.0 if el == "CA" else (12.0 if el == "C" else 16.0)
        system.addParticle(mass * dalton)
    for _ in range(n_dna):
        system.addParticle(0.0 * dalton)

    bonds = CustomBondForce("k*(10*r - d0)^2")
    bonds.addPerBondParameter("k")
    bonds.addPerBondParameter("d0")
    bonds.setUsesPeriodicBoundaryConditions(False)
    ii, jj, d0 = tbl["intra"]
    for i, j, d in zip(ii, jj, d0):
        bonds.addBond(int(i), int(j), [OMM_W_INTRA, float(d)])
    ci, cj, cd = tbl["cao"]
    for i, j, d in zip(ci, cj, cd):
        bonds.addBond(int(i), int(j), [W_CAO, float(d)])
    mi, mj, mt = tbl["com"][0], tbl["com"][1], tbl["com"][2]
    mw = tbl["com"][3] if len(tbl["com"]) > 3 else np.full(len(mi), W_COM, float)
    for i, j, d, w in zip(mi, mj, mt, mw):
        bonds.addBond(int(i), int(j), [float(w), float(d)])
    ei, ej, et, ew = tbl.get("epitax", (np.zeros(0, int),) * 4)
    for i, j, d, w in zip(ei, ej, et, ew):
        bonds.addBond(int(i), int(j), [float(w), float(d)])
    if bonds.getNumBonds():
        system.addForce(bonds)

    pos = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    pos.addPerParticleParameter("k")
    pos.addPerParticleParameter("x0")
    pos.addPerParticleParameter("y0")
    pos.addPerParticleParameter("z0")
    k_pos = 100.0 * W_POS  # Å² → nm²
    xyz0 = tbl["xyz0"]
    for i in range(n):
        x0, y0, z0 = xyz0[i] * NM
        k_use = (
            k_dna
            if (
                (freeze_resseq_le > 0 and int(whw[i].get("resseq", 9999)) <= freeze_resseq_le)
                or (seed_epitax and is_seed_atom(whw[i]))
            )
            else k_pos
        )
        pos.addParticle(i, [k_use, float(x0), float(y0), float(z0)])
    k_dna = 1.0e7
    for i, p in enumerate(dna_xyz):
        x0, y0, z0 = np.asarray(p, float) * NM
        pos.addParticle(n + i, [k_dna, float(x0), float(y0), float(z0)])
    system.addForce(pos)

    nb = CustomNonbondedForce(
        "w_oo*(oo_min-10*r)^2*step(oo_min-10*r)*step(is_o1*is_o2-0.5)*step(abs(gid1-gid2)-0.5)"
        "+w_ca*(ca_min-10*r)^2*step(ca_min-10*r)*step(is_ca1*is_ca2-0.5)"
        "+w_dna*(dna_min-10*r)^2*step(dna_min-10*r)*(is_dna1*(1-is_dna2)+is_dna2*(1-is_dna1))"
    )
    for name, val in (
        ("w_oo", W_OO * 1.5),
        ("oo_min", OO_TARGET),
        ("w_ca", W_CA),
        ("ca_min", CA_MIN),
        ("w_dna", W_DNA),
        ("dna_min", DNA_HEAVY),
    ):
        nb.addGlobalParameter(name, val)
    for name in ("is_o", "is_ca", "is_dna", "gid"):
        nb.addPerParticleParameter(name)
    gid = tbl["gid"]
    for i, a in enumerate(whw):
        nb.addParticle(
            [
                1.0 if is_oxygen(a) else 0.0,
                1.0 if is_ca(a) else 0.0,
                0.0,
                float(gid[i]),
            ]
        )
    for i in range(n_dna):
        nb.addParticle([0.0, 0.0, 1.0, -1.0])
    nb.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
    nb.setCutoffDistance(0.42 * nanometer)
    system.addForce(nb)

    # One-sided bonds on current short O···O pairs so clash clearance cannot
    # be traded away against Ca–O / COM harmonics.
    oo_a, oo_b = oo_pair_list(xyz, tbl["o_idx"], tbl["gid"], OO_TARGET + 0.15)
    if len(oo_a):
        oo_bforce = CustomBondForce("w*max(dmin-10*r,0)^2")
        oo_bforce.addPerBondParameter("w")
        oo_bforce.addPerBondParameter("dmin")
        oo_bforce.setUsesPeriodicBoundaryConditions(False)
        for i, j in zip(oo_a, oo_b):
            oo_bforce.addBond(int(i), int(j), [W_OO * 1.5, OO_TARGET])
        system.addForce(oo_bforce)

    xyz_all = np.vstack([xyz, dna_xyz]) if n_dna else xyz.copy()
    positions = (xyz_all * NM).tolist()

    platform = pick_platform(omm, platform_name)
    integrator = omm["VerletIntegrator"](0.001 * picosecond)
    context = omm["Context"](system, integrator, platform)
    context.setPositions(positions)
    state0 = context.getState(getEnergy=True)
    e0 = state0.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    print(
        f"OpenMM start: platform={platform.getName()}  E={e0:.3e} kJ/mol  "
        f"particles={n_all} (WHW {n}, DNA {n_dna})",
        flush=True,
    )
    omm["LocalEnergyMinimizer"].minimize(
        context,
        25 * kilojoule_per_mole / nanometer,
        max_iter,
    )
    state = context.getState(getEnergy=True, getPositions=True)
    e1 = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    pos_nm = state.getPositions(asNumpy=True)
    out = np.array(pos_nm.value_in_unit(nanometer), float) * 10.0
    xyz_whw = out[:n]
    n_oo_cart, min_oo_cart = inter_oo_stats(xyz_whw, tbl["o_idx"], tbl["gid"])
    ox_rmsd = oxalate_rmsd(xyz_whw, bodies)
    xyz_rig = xyz_whw.copy()
    xyz_rig = rigidify_xyz(xyz_rig, bodies)
    n_oo_rig, min_oo_rig = inter_oo_stats(xyz_rig, tbl["o_idx"], tbl["gid"])
    dna_shift = 0.0
    if n_dna:
        dna_shift = float(np.linalg.norm(out[n:] - dna_xyz, axis=1).max())
    # Keep the Cartesian OpenMM coordinates (n_oo typically 0). Re-rigidify
    # snap-back recreates a few shorts; FIRE polish restores rigidity instead.
    for i, a in enumerate(whw):
        a["xyz"] = xyz_whw[i]
    print(
        f"OpenMM done: E={e0:.3e} -> {e1:.3e} kJ/mol  max DNA shift={dna_shift:.4f} Å  "
        f"oxalate RMSD={ox_rmsd:.4f} Å  O-O<2Å cartesian n={n_oo_cart} min={min_oo_cart}  "
        f"after re-rigidify n={n_oo_rig} min={min_oo_rig}",
        flush=True,
    )
    del context
    return {
        "e0": e0,
        "e1": e1,
        "platform": platform.getName(),
        "dna_shift": dna_shift,
        "ox_rmsd": ox_rmsd,
        "n_oo_cart": n_oo_cart,
        "min_oo_cart": min_oo_cart,
        "n_oo_rig": n_oo_rig,
        "min_oo_rig": min_oo_rig,
    }


def run_openmm_nvt(
    whw,
    dna_xyz,
    tbl,
    ca_idx,
    *,
    ns: float = 0.1,
    timestep_ps: float = 0.001,
    temperature_k: float = 300.0,
    friction: float = 1.0,
    traj_interval: int = 500,
    traj_out: list | None = None,
    step_offset: int = 0,
    platform_name: str | None = None,
    freeze_resseq_le: int = 0,
    seed_epitax: bool = False,
    md_pos_scale: float = 0.04,
    max_iter: int = 50,
):
    """NVT Langevin MD on the same OpenMM model (after FIRE/min). Weak anchors."""
    loaded = import_openmm()
    if not isinstance(loaded, dict):
        err = loaded[1] if isinstance(loaded, tuple) else "unknown import error"
        raise RuntimeError(f"OpenMM not available: {err}")

    omm = loaded
    xyz = np.array([a["xyz"] for a in whw], float)
    n = len(whw)
    n_dna = len(dna_xyz)
    tbl = dict(tbl)
    tbl["xyz0"] = xyz.copy()

    System = omm["System"]
    CustomBondForce = omm["CustomBondForce"]
    CustomExternalForce = omm["CustomExternalForce"]
    CustomNonbondedForce = omm["CustomNonbondedForce"]
    nanometer = omm["nanometer"]
    dalton = omm["dalton"]
    kilojoule_per_mole = omm["kilojoule_per_mole"]
    picosecond = omm["picosecond"]
    kelvin = omm["kelvin"]

    system = System()
    k_dna = 1.0e7
    for a in whw:
        el = el_of(a)
        if freeze_resseq_le > 0 and int(a.get("resseq", 9999)) <= freeze_resseq_le:
            mass = 0.0
        elif seed_epitax and is_seed_atom(a):
            mass = 0.0
        else:
            mass = 40.0 if el == "CA" else (12.0 if el == "C" else 16.0)
        system.addParticle(mass * dalton)
    for _ in range(n_dna):
        system.addParticle(0.0 * dalton)

    bonds = CustomBondForce("k*(10*r - d0)^2")
    bonds.addPerBondParameter("k")
    bonds.addPerBondParameter("d0")
    bonds.setUsesPeriodicBoundaryConditions(False)
    ii, jj, d0 = tbl["intra"]
    for i, j, d in zip(ii, jj, d0):
        bonds.addBond(int(i), int(j), [OMM_W_INTRA, float(d)])
    ci, cj, cd = tbl["cao"]
    for i, j, d in zip(ci, cj, cd):
        bonds.addBond(int(i), int(j), [W_CAO, float(d)])
    mi, mj, mt = tbl["com"][0], tbl["com"][1], tbl["com"][2]
    mw = tbl["com"][3] if len(tbl["com"]) > 3 else np.full(len(mi), W_COM, float)
    for i, j, d, w in zip(mi, mj, mt, mw):
        bonds.addBond(int(i), int(j), [float(w), float(d)])
    ei, ej, et, ew = tbl.get("epitax", (np.zeros(0, int),) * 4)
    for i, j, d, w in zip(ei, ej, et, ew):
        bonds.addBond(int(i), int(j), [float(w), float(d)])
    if bonds.getNumBonds():
        system.addForce(bonds)

    pos_force = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    pos_force.addPerParticleParameter("k")
    pos_force.addPerParticleParameter("x0")
    pos_force.addPerParticleParameter("y0")
    pos_force.addPerParticleParameter("z0")
    k_pos = 100.0 * W_POS * float(md_pos_scale)
    xyz0 = tbl["xyz0"]
    for i in range(n):
        x0, y0, z0 = xyz0[i] * NM
        k_use = (
            k_dna
            if (
                (freeze_resseq_le > 0 and int(whw[i].get("resseq", 9999)) <= freeze_resseq_le)
                or (seed_epitax and is_seed_atom(whw[i]))
            )
            else k_pos
        )
        pos_force.addParticle(i, [k_use, float(x0), float(y0), float(z0)])
    for i, p in enumerate(dna_xyz):
        x0, y0, z0 = np.asarray(p, float) * NM
        pos_force.addParticle(n + i, [k_dna, float(x0), float(y0), float(z0)])
    system.addForce(pos_force)

    nb = CustomNonbondedForce(
        "w_oo*(oo_min-10*r)^2*step(oo_min-10*r)*step(is_o1*is_o2-0.5)*step(abs(gid1-gid2)-0.5)"
        "+w_ca*(ca_min-10*r)^2*step(ca_min-10*r)*step(is_ca1*is_ca2-0.5)"
        "+w_dna*(dna_min-10*r)^2*step(dna_min-10*r)*(is_dna1*(1-is_dna2)+is_dna2*(1-is_dna1))"
    )
    for name, val in (
        ("w_oo", W_OO * 1.5),
        ("oo_min", OO_TARGET),
        ("w_ca", W_CA),
        ("ca_min", CA_MIN),
        ("w_dna", W_DNA),
        ("dna_min", DNA_HEAVY),
    ):
        nb.addGlobalParameter(name, val)
    for name in ("is_o", "is_ca", "is_dna", "gid"):
        nb.addPerParticleParameter(name)
    gid = tbl["gid"]
    for i, a in enumerate(whw):
        nb.addParticle(
            [
                1.0 if is_oxygen(a) else 0.0,
                1.0 if is_ca(a) else 0.0,
                0.0,
                float(gid[i]),
            ]
        )
    for i in range(n_dna):
        nb.addParticle([0.0, 0.0, 1.0, -1.0])
    nb.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
    nb.setCutoffDistance(0.42 * nanometer)
    system.addForce(nb)

    oo_a, oo_b = oo_pair_list(xyz, tbl["o_idx"], tbl["gid"], OO_TARGET + 0.15)
    if len(oo_a):
        oo_bforce = CustomBondForce("w*max(dmin-10*r,0)^2")
        oo_bforce.addPerBondParameter("w")
        oo_bforce.addPerBondParameter("dmin")
        oo_bforce.setUsesPeriodicBoundaryConditions(False)
        for i, j in zip(oo_a, oo_b):
            oo_bforce.addBond(int(i), int(j), [W_OO * 1.5, OO_TARGET])
        system.addForce(oo_bforce)

    xyz_all = np.vstack([xyz, dna_xyz]) if n_dna else xyz.copy()
    positions = (xyz_all * NM).tolist()

    platform = pick_platform(omm, platform_name)
    integrator = omm["LangevinMiddleIntegrator"](
        temperature_k * kelvin,
        friction / picosecond,
        timestep_ps * picosecond,
    )
    context = omm["Context"](system, integrator, platform)
    context.setPositions(positions)
    if max_iter > 0:
        omm["LocalEnergyMinimizer"].minimize(
            context,
            25 * kilojoule_per_mole / nanometer,
            max_iter,
        )
    context.setVelocitiesToTemperature(temperature_k * kelvin)

    n_steps = max(0, int(round(ns * 1000.0 / timestep_ps)))
    interval = max(1, int(traj_interval))
    frames = []
    print(
        f"NVT MD: {ns:.3f} ns ({n_steps} steps @ {timestep_ps} ps), "
        f"T={temperature_k} K, pos anchor ×{md_pos_scale:.3f}",
        flush=True,
    )
    for step in range(1, n_steps + 1):
        integrator.step(1)
        if traj_out is not None and (step == 1 or step % interval == 0 or step == n_steps):
            state = context.getState(getPositions=True, getEnergy=True)
            pos_nm = state.getPositions(asNumpy=True)
            xyz_whw = np.array(pos_nm.value_in_unit(nanometer), float)[:n] * 10.0
            e = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
            frame = make_traj_frame(
                whw,
                xyz_whw,
                ca_idx,
                dna_xyz,
                step_offset + step,
                e,
                "md",
            )
            traj_out.append(frame)
            frames.append(frame)
        if step % max(interval * 10, 1000) == 0 or step == n_steps:
            state = context.getState(getEnergy=True)
            e = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
            print(f"  MD step {step}/{n_steps}  E={e:.3e} kJ/mol", flush=True)

    state = context.getState(getPositions=True)
    pos_nm = state.getPositions(asNumpy=True)
    xyz_whw = np.array(pos_nm.value_in_unit(nanometer), float)[:n] * 10.0
    for i, a in enumerate(whw):
        a["xyz"] = xyz_whw[i]
    del context
    return {"n_steps": n_steps, "frames": frames}


def oxalate_rmsd(xyz, bodies) -> float:
    acc = []
    for b in bodies:
        if len(b.idx) < 2:
            continue
        R, t = kabsch(xyz[b.idx], b.rel)
        pred = (R @ b.rel.T).T + t
        acc.append(float(np.sqrt(np.mean(np.square(xyz[b.idx] - pred)))))
    return float(np.mean(acc)) if acc else 0.0


def plot_history(history, out_png: Path, title: str):
    if not history:
        return
    sys.path.insert(0, str(ROOT))
    from matplotlib_config import apply_style, savefig  # noqa: WPS433
    import matplotlib.pyplot as plt

    apply_style()
    it, e, noo, fmax = zip(*history)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(it, e, color="#1b9e77", lw=1.6, label="FIRE energy")
    ax.set_xlabel("FIRE step")
    ax.set_ylabel("Restraint energy")
    ax.set_title(title)
    ax2 = ax.twinx()
    ax2.plot(it, noo, color="#d95f02", lw=1.2, ls="--", label="O···O < 2 Å")
    ax2.set_ylabel("Inter-group O···O < 2 Å")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax2.grid(False)
    savefig(out_png)
    plt.close(fig)


def output_stem(in_pdb: Path) -> str:
    stem = in_pdb.stem
    for tag in ("_relaxed", "_dls", "_fire", "_omm"):
        if stem.endswith(tag):
            stem = stem[: -len(tag)]
    return stem


def main():
    ap = argparse.ArgumentParser(
        description="FIRE (rigid oxalate) then OpenMM min; DNA fixed"
    )
    ap.add_argument("pdb", nargs="?", default=str(default_pdb()), type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=120, help="FIRE steps")
    ap.add_argument("--omm-steps", type=int, default=400, help="OpenMM minimizer iterations")
    ap.add_argument("--polish", type=int, default=60, help="rigid-oxalate FIRE steps after OpenMM")
    ap.add_argument(
        "--merge",
        type=float,
        default=0.0,
        help="Same-element atom merge (Å). 0 keeps intact C2O4 (recommended). "
        f"Old default {MERGE_SAME} Å splits oxalate.",
    )
    ap.add_argument(
        "--freeze-beyond",
        type=float,
        default=-1.0,
        help="Freeze WHW farther than this from DNA (Å). "
        "0 disables. Default: 12 Å when n_atoms>12000, else off.",
    )
    ap.add_argument("--max-move", type=float, default=0.12, help="max FIRE step (Å)")
    ap.add_argument("--platform", type=str, default=None, help="OpenMM platform (CPU/OpenCL/CUDA)")
    ap.add_argument("--skip-openmm", action="store_true")
    ap.add_argument(
        "--no-com-targets",
        action="store_true",
        help="Disable soft whewellite Ca–Ca distance targets (gel-first builds).",
    )
    ap.add_argument(
        "--freeze-resseq-le",
        type=int,
        default=0,
        help="Freeze WHW residues with resseq ≤ N (gel core) during FIRE/OpenMM.",
    )
    ap.add_argument(
        "--com-min-resseq",
        type=int,
        default=0,
        help="Apply COM Ca–Ca targets only when both Ca have resseq > N (shell-only).",
    )
    ap.add_argument(
        "--com-ramp-steps",
        type=int,
        default=0,
        help="Ramp COM target weight from 0→1 over this many FIRE steps.",
    )
    ap.add_argument(
        "--rotation-free-steps",
        type=int,
        default=0,
        help="Early FIRE steps with weaker positional anchors (0.35→1.0).",
    )
    ap.add_argument(
        "--traj",
        type=Path,
        default=None,
        help="Write Ca trajectory JSON for viewer playback.",
    )
    ap.add_argument(
        "--traj-interval",
        type=int,
        default=10,
        help="Record trajectory every N FIRE steps.",
    )
    ap.add_argument(
        "--shell-pos-weight",
        type=float,
        default=0.18,
        help="Positional anchor weight on shell (resseq > freeze-resseq-le).",
    )
    ap.add_argument(
        "--shell-max-shift",
        type=float,
        default=4.5,
        help="Max displacement (Å) for shell atoms / oxalate bodies.",
    )
    ap.add_argument(
        "--seed-epitax",
        action="store_true",
        help="Seed-local COM epitaxy: shell Ca near bfac≈12 seed pull toward "
        "whewellite COM spacings; weaken anchors in epitaxial zone; freeze seed.",
    )
    ap.add_argument(
        "--epitax-radius",
        type=float,
        default=EPITAX_RADIUS,
        help="Radius (Å) around seed Ca for epitaxial coupling.",
    )
    ap.add_argument(
        "--w-epitax",
        type=float,
        default=W_EPITAX,
        help="Base weight for seed–shell COM epitaxial springs.",
    )
    ap.add_argument(
        "--shell-epitax-pos-weight",
        type=float,
        default=0.06,
        help="Positional anchor weight in epitaxial zone (weaker than shell).",
    )
    ap.add_argument(
        "--shell-epitax-max-shift",
        type=float,
        default=10.0,
        help="Max displacement (Å) for atoms in epitaxial zone.",
    )
    ap.add_argument(
        "--gel-outward-com",
        action="store_true",
        help="Template shell from frozen gel Ca only (DNA→outward COM pull); "
        "disables isotropic shell–shell COM.",
    )
    ap.add_argument(
        "--no-seed-epitax-outward-only",
        action="store_true",
        help="Allow seed epitaxy to pull shell Ca inward toward DNA.",
    )
    ap.add_argument(
        "--traj-include-openmm",
        action="store_true",
        help="Append OpenMM + polish frames to trajectory (default: FIRE-only; "
        "avoids a large coordinate jump at the FIRE/OpenMM handoff).",
    )
    ap.add_argument(
        "--md-ns",
        type=float,
        default=0.0,
        help="NVT Langevin MD after OpenMM (ns). 0 = skip. Implies --traj-include-openmm.",
    )
    ap.add_argument(
        "--md-traj-interval",
        type=int,
        default=500,
        help="Record MD trajectory every N integration steps.",
    )
    ap.add_argument(
        "--md-pos-scale",
        type=float,
        default=0.04,
        help="Positional anchor scale during MD (fraction of minimization k).",
    )
    ap.add_argument(
        "--md-temperature",
        type=float,
        default=300.0,
        help="NVT temperature (K).",
    )
    ap.add_argument(
        "--w-gel-com",
        type=float,
        default=W_GEL_COM,
        help="Weight for gel→shell COM templating springs.",
    )
    ap.add_argument(
        "--gel-com-decay-len",
        type=float,
        default=GEL_COM_DECAY_LEN,
        help="Exponential decay length (Å) for gel COM pull beyond ~8 Å from gel.",
    )
    ap.add_argument(
        "--shell-com-min-dgel",
        type=float,
        default=SHELL_COM_MIN_DGEL,
        help="Beyond this d(gel) (Å), add shell–shell COM pairs in the outer annulus.",
    )
    args = ap.parse_args()
    if args.md_ns > 0:
        args.traj_include_openmm = True

    in_pdb = args.pdb
    stem = output_stem(in_pdb)
    out_fire = ROOT / f"{stem}_fire.pdb"
    out_omm = args.output if args.output else ROOT / f"{stem}_omm.pdb"
    report = out_omm.with_name(out_omm.stem + "_report.txt")
    fig_dir = ROOT / "figures" / "crystallinity"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_path = fig_dir / f"{stem}_fire_omm_energy.png"

    atoms, _ = parse_atoms(in_pdb)
    dna = [a for a in atoms if a["resname"] == "NUC"]
    other = [a for a in atoms if a["resname"] not in CAOX_RESNAMES and a["resname"] != "NUC"]
    whw = [dict(a) for a in atoms if a["resname"] in CAOX_RESNAMES]
    for a in whw:
        a["resname"] = "WHW"
    n_whw0 = len(whw)
    if args.merge > 0:
        whw = merge_duplicate_atoms(whw, args.merge)
        print(
            f"Merged same-element WHW atoms < {args.merge:.2f} Å: {n_whw0} -> {len(whw)}",
            flush=True,
        )
    dna_xyz = (
        np.array([a["xyz"] for a in dna if el_of(a) != "H"], float)
        if dna
        else np.zeros((0, 3))
    )
    freeze = args.freeze_beyond
    if freeze < 0:
        freeze = 12.0 if len(whw) > 12000 else 0.0

    traj_path = args.traj
    if traj_path is None and args.pdb:
        traj_path = ROOT / "viewer" / "trajectories" / f"{stem}_fire.trj.json"

    traj_frames: list = []
    traj_fire_frames: list = []
    fire = run_fire(
        whw,
        dna_xyz,
        args.steps,
        freeze,
        args.max_move,
        com_targets=not args.no_com_targets,
        freeze_resseq_le=args.freeze_resseq_le,
        com_min_resseq=args.com_min_resseq,
        com_ramp_steps=args.com_ramp_steps,
        rotation_free_steps=args.rotation_free_steps,
        shell_pos_weight=args.shell_pos_weight,
        shell_max_shift=args.shell_max_shift,
        seed_epitax=args.seed_epitax,
        epitax_radius=args.epitax_radius,
        w_epitax=args.w_epitax,
        shell_epitax_pos_weight=args.shell_epitax_pos_weight,
        shell_epitax_max_shift=args.shell_epitax_max_shift,
        seed_epitax_outward_only=not args.no_seed_epitax_outward_only,
        gel_outward_com=args.gel_outward_com,
        w_gel_com=args.w_gel_com,
        gel_com_decay_len=args.gel_com_decay_len,
        shell_com_min_dgel=args.shell_com_min_dgel,
        traj_path=None,
        traj_interval=args.traj_interval,
        traj_out=traj_fire_frames if traj_path else None,
    )
    if traj_path and not args.traj_include_openmm:
        write_traj_json(traj_path, len(fire["ca_idx"]), traj_fire_frames)
        print(
            f"Trajectory: FIRE-only ({len(traj_fire_frames)} frames); "
            "OpenMM/polish omitted (use --traj-include-openmm for full path)",
            flush=True,
        )
    if args.no_com_targets:
        print("COM Ca–Ca targets disabled (gel-first mode)", flush=True)
    elif args.com_min_resseq:
        print(
            f"COM targets on shell only (resseq > {args.com_min_resseq})",
            flush=True,
        )
    if args.seed_epitax:
        n_epi = len(fire["tbl"]["epitax"][0])
        print(
            f"Seed epitaxy: {n_epi} shell–seed COM pairs within "
            f"{args.epitax_radius:.1f} Å (w={args.w_epitax:.1f}, "
            f"seed frozen, outward-only={not args.no_seed_epitax_outward_only})",
            flush=True,
        )
    if args.gel_outward_com:
        n_gel = len(fire["tbl"]["com"][0])
        print(
            f"Gel-outward COM: {n_gel} pairs (w={args.w_gel_com:.1f}, "
            f"decay={args.gel_com_decay_len:.1f} Å, "
            f"shell COM beyond {args.shell_com_min_dgel:.1f} Å)",
            flush=True,
        )
    write_pdb(
        out_fire,
        dna + other + whw,
        [
            "HEADER    FIRE RIGID-OXALATE WHEWELLITE ON GROWTH MODEL\n",
            "TITLE     RIGID C2O4 FIRE; CA AND WATER FREE; DNA FIXED\n",
            f"REMARK   1 O-O < 2.0 A: {fire['n_oo_before']} -> {fire['n_oo_after']}\n",
        ],
    )
    print(f"Wrote {out_fire.name}", flush=True)

    omm_stats = None
    omm_err = None
    o_idx = np.array([i for i, a in enumerate(whw) if is_oxygen(a)], int)
    gid = fire["tbl"]["gid"]
    xyz_fire = np.array([a["xyz"] for a in whw], float)
    n_oo_fire, min_oo_fire = inter_oo_stats(xyz_fire, o_idx, gid)

    if not args.skip_openmm:
        try:
            omm_stats = run_openmm(
                whw,
                dna_xyz,
                fire["tbl"],
                fire["bodies"],
                args.omm_steps,
                args.platform,
                freeze_resseq_le=args.freeze_resseq_le,
                seed_epitax=args.seed_epitax,
            )
            if args.polish > 0:
                print(f"FIRE polish after OpenMM ({args.polish} steps) ...", flush=True)
                if args.traj_include_openmm and traj_path:
                    traj_frames = list(traj_fire_frames)
                    omm_step = int(fire["steps_run"]) + 1
                    xyz_omm_pre = np.array([a["xyz"] for a in whw], float)
                    traj_frames.append(
                        make_traj_frame(
                            whw,
                            xyz_omm_pre,
                            fire["ca_idx"],
                            dna_xyz,
                            omm_step,
                            omm_stats.get("e1"),
                            "openmm",
                        )
                    )
                polish = run_fire(
                    whw,
                    dna_xyz,
                    args.polish,
                    freeze,
                    args.max_move,
                    oxalates=fire["oxalates"],
                    ca_idx=fire["ca_idx"],
                    water=fire["water"],
                    com_targets=not args.no_com_targets,
                    freeze_resseq_le=args.freeze_resseq_le,
                    com_min_resseq=args.com_min_resseq,
                    com_ramp_steps=0,
                    rotation_free_steps=0,
                    shell_pos_weight=args.shell_pos_weight,
                    shell_max_shift=args.shell_max_shift,
                    seed_epitax=args.seed_epitax,
                    epitax_radius=args.epitax_radius,
                    w_epitax=args.w_epitax,
                    shell_epitax_pos_weight=args.shell_epitax_pos_weight,
                    shell_epitax_max_shift=args.shell_epitax_max_shift,
                    seed_epitax_outward_only=not args.no_seed_epitax_outward_only,
                    gel_outward_com=args.gel_outward_com,
                    w_gel_com=args.w_gel_com,
                    gel_com_decay_len=args.gel_com_decay_len,
                    shell_com_min_dgel=args.shell_com_min_dgel,
                    traj_path=None,
                    traj_interval=max(5, args.traj_interval // 5),
                    traj_out=traj_frames if (traj_path and args.traj_include_openmm) else None,
                    step_offset=int(fire["steps_run"]) + 1,
                    traj_phase="polish",
                )
                offset = fire["steps_run"]
                fire["history"] = fire["history"] + [
                    (it + offset, e, n, f) for it, e, n, f in polish["history"]
                ]
                omm_stats["polish_oo"] = polish["n_oo_after"]
                omm_stats["polish_min"] = polish["min_oo_after"]
                fire_steps = fire["steps_run"] + polish["steps_run"]
            else:
                fire_steps = fire["steps_run"]
                if args.traj_include_openmm and traj_path:
                    traj_frames = list(traj_fire_frames)
                    xyz_omm_pre = np.array([a["xyz"] for a in whw], float)
                    traj_frames.append(
                        make_traj_frame(
                            whw,
                            xyz_omm_pre,
                            fire["ca_idx"],
                            dna_xyz,
                            int(fire["steps_run"]) + 1,
                            omm_stats.get("e1"),
                            "openmm",
                        )
                    )
            xyz_omm = np.array([a["xyz"] for a in whw], float)
            if traj_path and traj_frames and args.traj_include_openmm:
                traj_frames.append(
                    make_traj_frame(
                        whw,
                        xyz_omm,
                        fire["ca_idx"],
                        dna_xyz,
                        fire_steps + 1,
                        omm_stats.get("e1"),
                        "final",
                    )
                )
            write_pdb(
                out_omm,
                dna + other + whw,
                [
                    "HEADER    OPENMM-MINIMIZED WHEWELLITE AFTER RIGID-OXALATE FIRE\n",
                    "TITLE     FIRE -> OPENMM -> RIGID-OXALATE POLISH; DNA FIXED\n",
                    f"REMARK   1 OpenMM {omm_stats['platform']}  "
                    f"E {omm_stats['e0']:.3e} -> {omm_stats['e1']:.3e} kJ/mol\n",
                ],
            )
            print(f"Wrote {out_omm.name}", flush=True)
            if args.md_ns > 0:
                if traj_path and not traj_frames:
                    traj_frames = list(traj_fire_frames)
                md_step0 = int(fire["steps_run"]) + int(args.polish or 0) + 2
                md_stats = run_openmm_nvt(
                    whw,
                    dna_xyz,
                    fire["tbl"],
                    fire["ca_idx"],
                    ns=args.md_ns,
                    traj_interval=args.md_traj_interval,
                    traj_out=traj_frames if traj_path else None,
                    step_offset=md_step0,
                    platform_name=args.platform,
                    freeze_resseq_le=args.freeze_resseq_le,
                    seed_epitax=args.seed_epitax,
                    md_pos_scale=args.md_pos_scale,
                    temperature_k=args.md_temperature,
                )
                omm_stats["md_steps"] = md_stats["n_steps"]
                omm_stats["md_frames"] = len(md_stats["frames"])
                write_pdb(
                    out_omm,
                    dna + other + whw,
                    [
                        "HEADER    OPENMM+MD WHEWELLITE AFTER RIGID-OXALATE FIRE\n",
                        "TITLE     FIRE -> OPENMM -> NVT MD; DNA FIXED\n",
                        f"REMARK   1 MD {args.md_ns:.3f} ns, {md_stats['n_steps']} steps\n",
                    ],
                )
                print(f"Updated {out_omm.name} after MD", flush=True)
        except Exception as exc:
            omm_err = str(exc)
            print(f"OpenMM failed: {omm_err}", flush=True)
            write_pdb(
                out_omm,
                dna + other + whw,
                [
                    "HEADER    FIRE RIGID-OXALATE (OPENMM DID NOT RUN)\n",
                    f"REMARK   1 {omm_err[:70]}\n",
                ],
            )
    else:
        write_pdb(
            out_omm,
            dna + other + whw,
            ["HEADER    FIRE RIGID-OXALATE (--skip-openmm)\n"],
        )

    xyz_final = np.array([a["xyz"] for a in whw], float)
    n_oo, min_oo = inter_oo_stats(xyz_final, o_idx, gid)

    if traj_path and traj_frames and args.traj_include_openmm:
        write_traj_json(traj_path, len(fire["ca_idx"]), traj_frames)

    plot_history(
        fire["history"],
        plot_path,
        "Rigid-oxalate FIRE on CaOx (DNA fixed)",
    )

    lines = [
        "FIRE (rigid oxalate) + OpenMM minimization — CaOx / WHW only (DNA fixed)",
        "=" * 72,
        f"Input     : {in_pdb.name}",
        f"FIRE PDB  : {out_fire.name}",
        f"OpenMM PDB: {out_omm.name}",
        f"WHW atoms : {n_whw0} -> {fire['n_atoms']} (same-element merge {args.merge} Å)",
        f"Groups    : {fire['n_oxalate']} rigid oxalate, {fire['n_ca']} Ca, {fire['n_water']} water O",
        f"FIRE      : {fire['steps_run']} steps, freeze-beyond={freeze:.1f} Å, "
        f"frozen bodies={fire['n_frozen_bodies']}",
        "",
        "Restraints (same as DLS, oxalate rigid instead of intra LS):",
        f"  O···O one-sided target {OO_TARGET:.2f} Å  (weight {W_OO})",
        f"  Ca···Ca one-sided min {CA_MIN:.2f} Å"
        + (
            f", COM targets {COM_CA}"
            if not args.no_com_targets
            else ", COM targets off (gel-first)"
        ),
        f"  Ca–O keep starting coordination (weight {W_CAO})",
        f"  Positional anchors ±{MAX_SHIFT:.2f} Å from start",
        "  DNA coordinates unchanged; WHW–DNA exclusion 2.20 Å",
        "",
        "Inter-group O···O < 2.00 Å:",
        f"  start : n={fire['n_oo_before']}  min={fire['min_oo_before']}",
        f"  FIRE  : n={n_oo_fire}  min={min_oo_fire}",
        f"  final : n={n_oo}  min={min_oo}",
        "",
    ]
    if omm_stats:
        lines += [
            "OpenMM LocalEnergyMinimizer:",
            f"  platform : {omm_stats['platform']}",
            f"  energy   : {omm_stats['e0']:.4e} -> {omm_stats['e1']:.4e} kJ/mol",
            f"  DNA max shift : {omm_stats['dna_shift']:.4f} Å (should be ~0)",
            f"  oxalate RMSD vs rigid : {omm_stats.get('ox_rmsd', 0):.4f} Å",
            f"  O-O after Cartesian min : n={omm_stats.get('n_oo_cart')}  min={omm_stats.get('min_oo_cart')}",
            f"  O-O after re-rigidify   : n={omm_stats.get('n_oo_rig')}  min={omm_stats.get('min_oo_rig')}",
            f"  intra-oxalate k = {OMM_W_INTRA} (keeps C2O4 rigid during Cartesian min)",
        ]
        if "polish_oo" in omm_stats:
            lines.append(
                f"  FIRE polish O-O        : n={omm_stats['polish_oo']}  min={omm_stats['polish_min']}"
            )
        if "md_steps" in omm_stats:
            lines += [
                "NVT Langevin MD:",
                f"  length   : {args.md_ns:.3f} ns ({omm_stats['md_steps']} steps)",
                f"  frames   : {omm_stats.get('md_frames', 0)}",
                f"  T        : {args.md_temperature:.0f} K",
                f"  pos anchor scale : {args.md_pos_scale:.3f}",
                "",
            ]
        lines += ["", ]
    elif omm_err:
        lines += ["OpenMM:", f"  FAILED: {omm_err}", ""]
    else:
        lines += ["OpenMM: skipped (--skip-openmm)", ""]
    if plot_path.exists():
        lines.append(f"Energy plot: {plot_path.relative_to(ROOT)}")
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
