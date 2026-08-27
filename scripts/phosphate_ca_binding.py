"""
Bond-valence guidance for Ca2+ at DNA phosphates.

Ca2+ (formal charge +2) in oxalate/phosphate environments is usually
6–8 coordinate. With Brown–Altermatt parameters for Ca–O (R0 = 1.969 Å,
b = 0.37), a 6-coordinate ion wants ~0.33 valence units (v.u.) per Ca–O
bond, which corresponds to d(Ca–O) ≈ 2.37 Å.

At a phosphate, the preferred inner-sphere motif is bidentate chelation
to the two non-bridging oxygens (OP1 / OP2), not placement on the P atom
or far along the OP1–OP2 bisector toward bulk solvent (that inflates Ca–Ca
along the helix to ~8 Å).

This module scores and places Ca in the O–P–O plane, biased slightly
outward from the helix axis (minor-groove / solvent side) while keeping
both Ca–OP distances near the BV target.
"""

from __future__ import annotations

import math

import numpy as np

# Brown & Altermatt (1985), Ca2+–O2-
R0_CA_O = 1.969
B_CA_O = 0.37
V_CA = 2.0
N_INNER = 6  # expected first-shell O count (2×P + ~4 oxalate/water)

D_TARGET = R0_CA_O - B_CA_O * math.log(V_CA / N_INNER)
D_MIN, D_MAX = 2.20, 2.55  # chemically usable Ca–OP window


def bond_valence(d: float, r0: float = R0_CA_O, b: float = B_CA_O) -> float:
    return math.exp((r0 - float(d)) / b)


def distance_for_valence(v: float, r0: float = R0_CA_O, b: float = B_CA_O) -> float:
    return r0 - b * math.log(max(v, 1e-6))


def norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v * 0.0 if n < 1e-10 else v / n


def radial_frame(xyz: np.ndarray, origin: np.ndarray, axis: np.ndarray):
    w = xyz - origin
    z = float(np.dot(w, axis))
    rho = w - axis * z
    r = float(np.linalg.norm(rho))
    rhat = norm(rho)
    that = axis
    bhat = np.cross(that, rhat)
    return r, rhat, that, bhat, z


def optimal_ca_at_phosphate(
    p_xyz: np.ndarray,
    op_xyzs: list[np.ndarray],
    origin: np.ndarray,
    axis: np.ndarray,
    *,
    d_target: float = D_TARGET,
) -> tuple[np.ndarray, dict]:
    """
    Place Ca2+ for bidentate OP chelation in the phosphate plane.

    Returns (ca_xyz, metrics dict with distances and bond valences).
    """
    p_xyz = np.asarray(p_xyz, float)
    ops = [np.asarray(o, float) for o in op_xyzs if o is not None]
    _, rhat, that, bhat, _ = radial_frame(p_xyz, origin, axis)

    if len(ops) == 0:
        ca = p_xyz + 2.6 * bhat
        return ca, {"mode": "fallback", "d_op": [], "bv_op": [], "bv_sum": 0.0}

    if len(ops) == 1:
        op = ops[0]
        vec = norm(op - p_xyz)
        ca = op + vec * d_target
        d = float(np.linalg.norm(ca - op))
        bv = bond_valence(d)
        return ca, {
            "mode": "monodentate",
            "d_op": [d],
            "bv_op": [bv],
            "bv_sum": bv,
            "d_p": float(np.linalg.norm(ca - p_xyz)),
        }

    op1, op2 = ops[0], ops[1]
    mid = 0.5 * (op1 + op2)
    in_plane = norm(np.cross(op1 - p_xyz, op2 - p_xyz))
    if np.linalg.norm(in_plane) < 0.05:
        in_plane = bhat

    # Bisector in the O–P–O plane, pointing away from P.
    bise = norm(mid - p_xyz)
    if float(np.dot(bise, rhat)) < 0:
        bise = -bise

    best = None
    best_sc = 1e99
    for t in np.linspace(-0.8, 2.4, 33):
        for side in (1.0, -1.0):
            ca = mid + t * bise + side * 0.35 * in_plane * float(np.dot(in_plane, rhat))
            d1 = float(np.linalg.norm(ca - op1))
            d2 = float(np.linalg.norm(ca - op2))
            if d1 < D_MIN or d2 < D_MIN or d1 > D_MAX + 0.15 or d2 > D_MAX + 0.15:
                continue
            bv1, bv2 = bond_valence(d1), bond_valence(d2)
            sc = (d1 - d_target) ** 2 + (d2 - d_target) ** 2
            sc += 0.08 * (d1 - d2) ** 2
            sc += 0.15 * max(0.0, 3.15 - float(np.linalg.norm(ca - p_xyz))) ** 2
            sc += 0.10 * max(0.0, float(np.linalg.norm(ca - p_xyz)) - 3.55) ** 2
            # Prefer solvent-exposed (radial) without moving far off the helix cylinder.
            dr = float(np.dot(ca - p_xyz, rhat))
            sc += 0.06 * max(0.0, 0.15 - dr) ** 2
            sc += 0.04 * dr**2
            if sc < best_sc:
                best_sc = sc
                best = ca
                metrics = {
                    "mode": "bidentate",
                    "d_op": [d1, d2],
                    "bv_op": [bv1, bv2],
                    "bv_sum": bv1 + bv2,
                    "d_p": float(np.linalg.norm(ca - p_xyz)),
                    "radial_A": dr,
                }

    if best is None:
        # Analytic two-point placement on bisector.
        op1, op2 = ops[0], ops[1]
        ang = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(norm(op1 - p_xyz), norm(op2 - p_xyz)),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        half = math.radians(ang) / 2.0
        if half < 0.05:
            half = math.radians(60.0)
        along = d_target / math.sin(half)
        best = mid + along * bise
        d1 = float(np.linalg.norm(best - op1))
        d2 = float(np.linalg.norm(best - op2))
        metrics = {
            "mode": "bidentate-analytic",
            "d_op": [d1, d2],
            "bv_op": [bond_valence(d1), bond_valence(d2)],
            "bv_sum": bond_valence(d1) + bond_valence(d2),
            "d_p": float(np.linalg.norm(best - p_xyz)),
            "radial_A": float(np.dot(best - p_xyz, rhat)),
        }

    return best, metrics


def binding_site_summary() -> str:
    """Human-readable summary of the BV placement model."""
    lines = [
        "Bond-valence model for Ca2+ at DNA phosphate (non-bridging OP1/OP2)",
        f"  Ca–O target distance (6-fold): {D_TARGET:.3f} Å",
        f"  Acceptable Ca–OP window: {D_MIN:.2f}–{D_MAX:.2f} Å",
        "  Placement: bidentate chelation in the O–P–O plane,",
        "  offset slightly outward from the helix axis (solvent-exposed).",
        "  Remaining valence (~1.3 v.u.) is satisfied by oxalate oxygens",
        "  and waters in the first shell — not preset to whewellite geometry.",
    ]
    return "\n".join(lines)
