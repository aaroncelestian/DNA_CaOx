#!/usr/bin/env python3
"""Find lattice translations and point-group hints in the no-DNA CaOx model."""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grow_whewellite import (  # noqa: E402
    BACKBONE,
    XTL,
    dna_ca_by_strand,
    expand_crystal,
    local_frame,
    merge_points,
    norm,
    parse_atoms,
    rotation_around,
    rotation_from_to,
)

PDB = ROOT / "CaOx_whewellite_noDNA.pdb"
DNA_PDB = ROOT / "DNA_CaOx_whewellite_grown.pdb"
REPORT = ROOT / "CaOx_whewellite_noDNA_symmetry.txt"
SWEEP_REPORT = ROOT / "CaOx_whewellite_symmetry_sweep.txt"

COM_A, COM_B, COM_C, COM_BETA = 6.290, 14.583, 10.116, 109.46
COM_384 = 3.843
# Strict COM-net match: exclude nearby COM distances (6.68, 6.88, 9.58 Å).
TOL = 0.25  # Å — mapped Ca must land this close
LEN_A = 0.20
LEN_C = 0.25
LEN_384 = 0.15
LEN_B = 0.40
CLUSTER_RAD = 0.35
BETA_TOL = 10.0  # deg from min(β, 180-β) ≈ 70.5°
# Phase cuts tuned for nucleation-site detection (partial COM ordering), not
# strict bulk crystal. Geometric windows (LEN_*, TOL) stay strict.
MIN_FRAC = 0.08  # axis "found" in local 10 Å patch
CRYST_SCORE = 0.22  # strongest ordering pocket (maps to crystalline in viewer)
INT_SCORE = 0.10  # clear nucleation site
NUCLEATION_SCORE = 0.06  # weak COM-net precursor
NUCLEATION_AXIS_MIN = 0.05  # single-axis hint in shell
COM_REGISTRY = (3.843, 6.273, 6.290)
COM_REGISTRY_WIN = 0.42  # Å — neighbor Ca–Ca match to whewellite COM spacings
COM_PAIR_RADIUS = 8.0
HOTSPOT_PAIR_MIN = 0.08  # min local COM pair-correlation score
HOTSPOT_CLUSTER_CUT = 7.0
HOTSPOT_MIN_SIZE = 3
HOTSPOT_MAX_PER_CLUSTER = 8
HOTSPOT_EDGE_FRAC = 0.32  # drop outer ~32% of cluster by centroid distance
HOTSPOT_MIN_CLUSTER_NEIGHBORS = 2  # exclude dangling edge Ca
SEED_CRYSTAL_BFAC = 12.0  # whewellite patch from build_gel_shell --crystal-seed
BFAC_GEL_COAT = 18.0
BFAC_INTERMEDIATE = 22.0
BFAC_PRECRYSTAL = 24.0
BFAC_BULK = 26.0
SEED_PAIR_TOL = 0.35  # Å on Ca–Ca for distance-fingerprint scoring
MERGE_R = 0.85
SEED_KEEP = 0.70
SWEEP_RADII = [6, 8, 10, 12, 15, 20, 25, 30]


def load_ca(path: Path):
    xyz, chain, bfac = [], [], []
    for line in path.open():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        name = line[12:16].strip().upper()
        el = line[76:78].strip().upper()
        if el != "CA" and name != "CA":
            continue
        xyz.append(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        )
        chain.append(line[21])
        bfac.append(float(line[60:66]) if line[60:66].strip() else 0.0)
    return np.array(xyz), np.array(chain), np.array(bfac)


def load_dna_heavy(path: Path) -> np.ndarray:
    pts = []
    for line in path.open():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[17:20].strip() != "NUC":
            continue
        el = (line[76:78].strip() or line[12:16].strip()[0]).upper()
        if el == "H":
            continue
        pts.append(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        )
    return np.array(pts) if pts else np.zeros((0, 3))


def cluster_domains(pts, cutoff=22.0):
    """Single-link clusters so the two duplex crystallites separate."""
    n = len(pts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        d = np.linalg.norm(pts[i + 1 :] - pts[i], axis=1)
        for k in np.where(d < cutoff)[0]:
            a, b = find(i), find(i + 1 + int(k))
            if a != b:
                parent[b] = a
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [np.array(ix) for ix in groups.values() if len(ix) >= 20]


def cluster_local(pts, cutoff=HOTSPOT_CLUSTER_CUT, min_size=HOTSPOT_MIN_SIZE):
    """Small spatial clusters for nucleation hotspot detection."""
    n = len(pts)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        d = np.linalg.norm(pts[i + 1 :] - pts[i], axis=1)
        for k in np.where(d < cutoff)[0]:
            a, b = find(i), find(i + 1 + int(k))
            if a != b:
                parent[b] = a
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [np.array(ix, int) for ix in groups.values() if len(ix) >= min_size]


def helix_axis_radius(pts: np.ndarray, dna_heavy: np.ndarray) -> np.ndarray:
    """Cylindrical distance from the DNA helix axis (Å)."""
    if len(pts) == 0 or len(dna_heavy) < 3:
        return np.zeros(len(pts), float)
    origin = dna_heavy.mean(axis=0)
    _, _, vh = np.linalg.svd(dna_heavy - origin)
    axis = vh[0]
    rel = pts - origin
    along = rel @ axis
    perp = rel - np.outer(along, axis)
    return np.linalg.norm(perp, axis=1)


def phosphate_surface_radius(pxyz: np.ndarray, dna_heavy: np.ndarray) -> float:
    """Median phosphate distance from helix axis — DNA outer envelope."""
    if len(pxyz) == 0:
        return float(np.median(helix_axis_radius(dna_heavy, dna_heavy)))
    return float(np.median(helix_axis_radius(pxyz, dna_heavy)))


def local_com_registry(pts, radius: float = 10.0) -> np.ndarray:
    """Fraction of neighbors within COM_REGISTRY_WIN of a whewellite COM Ca–Ca distance."""
    n = len(pts)
    scores = np.zeros(n, float)
    if n < 2:
        return scores
    targets = np.array(COM_REGISTRY, float)
    for i in range(n):
        d = np.linalg.norm(pts - pts[i], axis=1)
        d = d[(d > 0.5) & (d <= radius)]
        if len(d) == 0:
            continue
        hits = int(np.sum(np.min(np.abs(d[:, None] - targets[None, :]), axis=1) <= COM_REGISTRY_WIN))
        scores[i] = hits / len(d)
    return scores


def local_com_pair_correlation(pts, radius: float = COM_PAIR_RADIUS):
    """
    Incipient COM ordering: local Ca–Ca pair hits at 3.84 Å and 6.27/6.29 Å.
    Returns pair_score (higher when both spacings appear among neighbors).
    """
    n = len(pts)
    pair_score = np.zeros(n, float)
    has_384 = np.zeros(n, bool)
    has_629 = np.zeros(n, bool)
    n_com = np.zeros(n, int)
    t384 = COM_REGISTRY[0]
    win = COM_REGISTRY_WIN
    for i in range(n):
        d = np.linalg.norm(pts - pts[i], axis=1)
        d = d[(d > 0.5) & (d <= radius)]
        if len(d) < 2:
            continue
        h384 = np.abs(d - t384) <= win
        h629 = np.zeros(len(d), bool)
        for t in COM_REGISTRY[1:]:
            h629 |= np.abs(d - t) <= win
        has_384[i] = bool(np.any(h384))
        has_629[i] = bool(np.any(h629))
        n_com[i] = int(np.sum(h384)) + int(np.sum(h629))
        pair_score[i] = n_com[i] / len(d)
        if has_384[i] and has_629[i]:
            pair_score[i] += 0.14
    return pair_score, has_384, has_629, n_com


def cluster_core_indices(
    pts: np.ndarray,
    ix: np.ndarray,
    pair_score: np.ndarray,
    has_384: np.ndarray,
    has_629: np.ndarray,
) -> np.ndarray:
    """Keep interior cluster Ca with loose COM pair correlations — not cluster edges."""
    if len(ix) < HOTSPOT_MIN_SIZE:
        return np.array([], int)
    sub = pts[ix]
    cent = sub.mean(axis=0)
    d_cent = np.linalg.norm(sub - cent, axis=1)
    nn_in = np.zeros(len(ix), int)
    for k in range(len(ix)):
        d = np.linalg.norm(sub - sub[k], axis=1)
        nn_in[k] = int(np.sum((d > 0.5) & (d <= HOTSPOT_CLUSTER_CUT)))

    inner_r = float(np.percentile(d_cent, 100.0 * (1.0 - HOTSPOT_EDGE_FRAC)))
    core_local = (d_cent <= inner_r) & (nn_in >= HOTSPOT_MIN_CLUSTER_NEIGHBORS)
    if int(np.sum(core_local)) < HOTSPOT_MIN_SIZE:
        order = np.argsort(d_cent)
        core_local = np.zeros(len(ix), bool)
        core_local[order[: max(HOTSPOT_MIN_SIZE, len(ix) // 2)]] = True

    core_ix = ix[core_local]
    ps = pair_score[core_ix]
    ok = (ps >= HOTSPOT_PAIR_MIN) & has_384[core_ix] & has_629[core_ix]
    ok |= ps >= (HOTSPOT_PAIR_MIN + 0.06)
    return core_ix[ok]


def nucleation_hotspot_mask(
    pts: np.ndarray,
    pair_score: np.ndarray,
    has_384: np.ndarray,
    has_629: np.ndarray,
    n_com: np.ndarray,
    d_p: np.ndarray,
    scores: np.ndarray,
    r_axis: np.ndarray | None = None,
    r_phosphate: float = 0.0,
) -> tuple[np.ndarray, list[dict]]:
    """
    Hotspots = interior Ca in small clusters with incipient COM pair correlations
    (neighbors at both ~3.84 Å and ~6.29 Å). Excludes cluster periphery.
    """
    mask = np.zeros(len(pts), bool)
    clusters_out: list[dict] = []

    if r_axis is not None and r_phosphate > 0:
        outward = r_axis >= (r_phosphate - 0.35)
    else:
        outward = np.ones(len(pts), bool)

    site_ok = outward & (
        (pair_score >= HOTSPOT_PAIR_MIN) & has_384 & has_629
        | (pair_score >= HOTSPOT_PAIR_MIN + 0.07)
        | ((n_com >= 3) & has_384 & has_629 & (pair_score >= HOTSPOT_PAIR_MIN * 0.85))
    )

    for ix in cluster_local(pts, cutoff=HOTSPOT_CLUSTER_CUT, min_size=HOTSPOT_MIN_SIZE):
        if not np.any(site_ok[ix]):
            continue
        mean_dp = float(d_p[ix].mean())
        if mean_dp > 22.0 and float(pair_score[ix].mean()) < HOTSPOT_PAIR_MIN:
            continue
        core = cluster_core_indices(pts, ix, pair_score, has_384, has_629)
        if len(core) < HOTSPOT_MIN_SIZE:
            continue
        if len(core) > HOTSPOT_MAX_PER_CLUSTER:
            order = np.argsort(-pair_score[core])
            core = core[order[:HOTSPOT_MAX_PER_CLUSTER]]
        mask[core] = True
        cent = pts[core].mean(axis=0)
        clusters_out.append(
            {
                "center": cent.tolist(),
                "radius": float(np.linalg.norm(pts[core] - cent, axis=1).max()),
                "n": int(len(core)),
                "mean_pair_corr": float(pair_score[core].mean()),
                "mean_d_p": float(d_p[core].mean()),
                "mean_score": float(scores[core].mean()),
            }
        )

    mask &= outward
    return mask, clusters_out


def translation_score(pts, vec, tol=TOL):
    shifted = pts + vec
    hits = 0
    step = 200
    for i0 in range(0, len(shifted), step):
        sl = shifted[i0 : i0 + step]
        d = np.linalg.norm(sl[:, None, :] - pts[None, :, :], axis=2)
        hits += int(np.sum(d.min(axis=1) <= tol))
    return hits / len(pts)


def candidate_vectors(pts, lo=5.5, hi=15.5, max_pairs=8000):
    n = len(pts)
    rng = np.random.default_rng(3)
    idx = np.arange(n)
    if n > 250:
        idx = rng.choice(n, size=250, replace=False)
    vecs = []
    for i in idx:
        d = pts - pts[i]
        r = np.linalg.norm(d, axis=1)
        keep = np.where((r >= lo) & (r <= hi))[0]
        for j in keep:
            vecs.append(d[j])
    vecs = np.array(vecs)
    if len(vecs) > max_pairs:
        vecs = vecs[rng.choice(len(vecs), size=max_pairs, replace=False)]
    return vecs


def cluster_vectors(vecs, rad=0.55):
    """Greedy unique directions/lengths."""
    used = np.zeros(len(vecs), dtype=bool)
    clusters = []
    order = np.argsort(-np.linalg.norm(vecs, axis=1))
    for i in order:
        if used[i]:
            continue
        d = np.linalg.norm(vecs - vecs[i], axis=1)
        d2 = np.linalg.norm(vecs + vecs[i], axis=1)
        mask = (d < rad) | (d2 < rad)
        used |= mask
        members = vecs[mask]
        seed = vecs[i]
        align = np.sign(members @ seed)
        align[align == 0] = 1
        mean = (members * align[:, None]).mean(axis=0)
        clusters.append((mean, int(mask.sum())))
    clusters.sort(key=lambda t: -t[1])
    return clusters


def angle_between(u, v):
    c = float(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))
    return math.degrees(math.acos(c))


def nearest_com_axis(vec):
    L = np.linalg.norm(vec)
    targets = [("a", COM_A), ("b", COM_B), ("c", COM_C)]
    name, tlen = min(targets, key=lambda kv: abs(L - kv[1]))
    return name, tlen, abs(L - tlen)


def inversion_score(pts, center, tol=TOL):
    img = 2 * center - pts
    hits = 0
    step = 200
    for i0 in range(0, len(img), step):
        sl = img[i0 : i0 + step]
        d = np.linalg.norm(sl[:, None, :] - pts[None, :, :], axis=2)
        hits += int(np.sum(d.min(axis=1) <= tol))
    return hits / len(pts)


def twofold_score(pts, origin, axis, tol=TOL):
    axis = axis / np.linalg.norm(axis)
    rel = pts - origin
    proj = (rel @ axis)[:, None] * axis
    img = origin + (2 * proj - rel)
    hits = 0
    step = 200
    for i0 in range(0, len(img), step):
        sl = img[i0 : i0 + step]
        d = np.linalg.norm(sl[:, None, :] - pts[None, :, :], axis=2)
        hits += int(np.sum(d.min(axis=1) <= tol))
    return hits / len(pts)


def screw21_score(pts, origin, axis, trans, tol=TOL):
    """2_1: 180° about axis plus translation of trans/2 along axis."""
    axis = axis / np.linalg.norm(axis)
    rel = pts - origin
    proj = (rel @ axis)[:, None] * axis
    rot = origin + (2 * proj - rel)
    img = rot + 0.5 * trans * axis
    hits = 0
    step = 200
    for i0 in range(0, len(img), step):
        sl = img[i0 : i0 + step]
        d = np.linalg.norm(sl[:, None, :] - pts[None, :, :], axis=2)
        hits += int(np.sum(d.min(axis=1) <= tol))
    return hits / len(pts)


def dlen_ok(L, axis):
    t = {"a": COM_A, "b": COM_B, "c": COM_C}[axis]
    win = {"a": LEN_A, "b": LEN_B, "c": LEN_C}[axis]
    return abs(L - t) < win


def com_beta_acute() -> float:
    return min(COM_BETA, 180.0 - COM_BETA)


def beta_ok(ang: float | None) -> bool:
    if ang is None:
        return False
    return abs(float(ang) - com_beta_acute()) <= BETA_TOL


def analyze_patch_symmetry(nb: np.ndarray) -> dict:
    """Best a/c translation fractions and 3.84 Å score for a Ca patch."""
    out = {"a": 0.0, "c": 0.0, "frac_384": 0.0, "beta": None, "n_ca": len(nb)}
    if len(nb) < 8:
        return out
    raw = candidate_vectors(nb, lo=3.4, hi=11.2, max_pairs=4000)
    if len(raw) == 0:
        return out
    cl = cluster_vectors(raw, rad=CLUSTER_RAD)[:15]
    vec_ac = {}
    best_384 = 0.0
    for vec, _pop in cl:
        L = np.linalg.norm(vec)
        frac = translation_score(nb, vec)
        if abs(L - COM_384) < LEN_384:
            best_384 = max(best_384, frac)
        axis, _t, dlen = nearest_com_axis(vec)
        if axis == "a" and dlen < LEN_A and frac > out["a"]:
            out["a"] = frac
            vec_ac["a"] = vec
        elif axis == "c" and dlen < LEN_C and frac > out["c"]:
            out["c"] = frac
            vec_ac["c"] = vec
    out["frac_384"] = best_384
    if "a" in vec_ac and "c" in vec_ac:
        ang = angle_between(vec_ac["a"], vec_ac["c"])
        ang = min(ang, 180.0 - ang)
        out["beta"] = ang
        # Wrong monoclinic angle: not a COM cell. Keep a (backbone repeat), drop c.
        if not beta_ok(ang):
            out["c"] = 0.0
    return out


def whewellite_distance_metrics(nb: np.ndarray, tol: float = SEED_PAIR_TOL) -> dict:
    """
    Orientation-free whewellite fingerprint from Ca–Ca distances.

    For each Ca, check whether neighbors exist at COM 3.84, 6.29, and 10.1 Å.
    Used for bulk crystal seeds (bfac ≈ SEED_CRYSTAL_BFAC) that sit outside the
    DNA COM-net graph but still carry authentic lattice contacts.
    """
    n = len(nb)
    if n < 4:
        return {"a": 0.0, "c": 0.0, "frac_384": 0.0}
    has_384 = has_a = has_c = 0
    for i in range(n):
        ds = np.linalg.norm(nb - nb[i], axis=1)
        ds = ds[ds > 0.5]
        if np.any(np.abs(ds - COM_384) < tol):
            has_384 += 1
        if np.any(np.abs(ds - COM_A) < tol):
            has_a += 1
        if np.any(np.abs(ds - COM_C) < tol):
            has_c += 1
    return {"a": has_a / n, "c": has_c / n, "frac_384": has_384 / n}


def upgrade_crystal_seed_phases(
    metrics: dict,
    pts: np.ndarray,
    bfac: np.ndarray,
) -> tuple[dict, int, str | None]:
    """
    Re-label embedded whewellite seed Ca using distance-fingerprint scoring.

    Returns (metrics, n_upgraded, cluster_phase).
    """
    seed_ix = np.where(np.abs(bfac - SEED_CRYSTAL_BFAC) < 0.5)[0]
    if len(seed_ix) < 4:
        return metrics, 0, None

    cluster_sym = whewellite_distance_metrics(pts[seed_ix])
    cluster_phase = phase_label(cluster_sym)
    n_up = 0

    if cluster_phase != "amorphous":
        score = crystallinity_index(cluster_sym)
        for i in seed_ix:
            metrics["phase"][i] = cluster_phase
            metrics["score"][i] = score
            metrics["frac_a"][i] = cluster_sym["a"]
            metrics["frac_c"][i] = cluster_sym["c"]
            metrics["frac_384"][i] = cluster_sym["frac_384"]
            n_up += 1
        return metrics, n_up, cluster_phase

    # Fallback: per-seed-site patch using seed Ca neighbors only
    for i in seed_ix:
        d_seed = np.linalg.norm(pts[seed_ix] - pts[i], axis=1)
        nb_ix = seed_ix[d_seed <= 12.0 + 0.05]
        if len(nb_ix) < 4:
            continue
        sym = whewellite_distance_metrics(pts[nb_ix])
        ph = phase_label(sym)
        if ph == "amorphous":
            continue
        metrics["phase"][i] = ph
        metrics["score"][i] = crystallinity_index(sym)
        metrics["frac_a"][i] = sym["a"]
        metrics["frac_c"][i] = sym["c"]
        metrics["frac_384"][i] = sym["frac_384"]
        n_up += 1
    return metrics, n_up, cluster_phase if n_up else None


def upgrade_radial_shell_phases(
    metrics: dict,
    bfac: np.ndarray,
    d_p: np.ndarray,
) -> int:
    """
  Label DNA-templated shell zones (bfac 18/22/24 from build_gel_shell) when
  local COM-net order is partial — restores gel → intermediate → crystal rings.
    """
    n_up = 0
    phases = metrics["phase"]
    scores = metrics["score"]
    fa = metrics["frac_a"]
    fc = metrics["frac_c"]
    for i in range(len(bfac)):
        if phases[i] == "crystalline":
            continue
        bf = float(bfac[i])
        dp = float(d_p[i])
        if abs(bf - BFAC_INTERMEDIATE) < 0.5 and 8.0 <= dp <= 20.0:
            phases[i] = "intermediate"
            n_up += 1
        elif abs(bf - BFAC_PRECRYSTAL) < 0.5 and 14.0 <= dp <= 24.0:
            phases[i] = "intermediate"
            n_up += 1
        elif abs(bf - BFAC_BULK) < 0.5 and 20.0 <= dp <= 30.0:
            phases[i] = "intermediate"
            n_up += 1
        elif abs(bf - BFAC_GEL_COAT) < 0.5 and 3.5 <= dp <= 16.0:
            if scores[i] >= 0.015 or fa[i] >= 0.025 or fc[i] >= 0.025:
                phases[i] = "intermediate"
                n_up += 1
            if fa[i] >= 0.06 and scores[i] >= 0.08 and dp <= 12.0:
                phases[i] = "crystalline"
                n_up += 1
        elif dp <= 18.0 and scores[i] >= NUCLEATION_SCORE and (
            fa[i] >= NUCLEATION_AXIS_MIN or fc[i] >= NUCLEATION_AXIS_MIN
        ):
            if phases[i] == "amorphous":
                phases[i] = "intermediate"
                n_up += 1
    return n_up


def summarize_symmetry_list(vals: list[float], label: str, lines: list[str]):
    if not vals:
        lines.append(f"  {label}: no matches")
        return
    arr = np.array(vals)
    lines.append(
        f"  {label}: n={len(arr)}  median frac={np.median(arr):.2f}  "
        f"mean={arr.mean():.2f}  max={arr.max():.2f}"
    )


def build_oriented_lattice_ca(max_r: float):
    """
    Ideal COM Ca lattice sites per DNA-bound seed, oriented like grow_whewellite.
    Includes crystallographic 3.84 Å contacts (unlike rigid grown model).
    """
    dna_atoms, _ = parse_atoms(BACKBONE)
    xtl_atoms, cryst = parse_atoms(XTL)
    nmax = max(2, int(math.ceil(max_r / min(cryst["a"], cryst["c"]))) + 1)
    expanded, (av, bv, cv) = expand_crystal(xtl_atoms, cryst, nmax=nmax)
    ref = next(a for a in xtl_atoms if a["element"].upper() == "CA")
    strands = dna_ca_by_strand(dna_atoms)

    seed_positions: dict[int, np.ndarray] = {}
    lattice_by_seed: dict[int, np.ndarray] = {}

    for strand in strands:
        helix_o = np.mean([a["xyz"] for a in strand], axis=0)
        for i, ca in enumerate(strand):
            t, out_vec = local_frame(strand, i, helix_o)
            R1 = rotation_from_to(av, t)
            best_ang, best_sc = 0.0, -1e9
            for ang in np.linspace(0, 2 * math.pi, 72, endpoint=False):
                R = rotation_around(t, ang) @ R1
                sc = float(np.dot(cv @ R.T, out_vec))
                if sc > best_sc:
                    best_sc, best_ang = sc, ang
            R = rotation_around(t, best_ang) @ R1
            seed_positions[ca["resseq"]] = ca["xyz"].copy()
            ca_list = []
            for a in expanded:
                if a["element"].upper() != "CA":
                    continue
                xyz = (a["xyz"] - ref["xyz"]) @ R.T + ca["xyz"]
                if float(np.linalg.norm(xyz - ca["xyz"])) <= max_r + 0.05:
                    ca_list.append(xyz)
            lattice_by_seed[ca["resseq"]] = np.array(ca_list)

    return seed_positions, lattice_by_seed


def observed_ca_by_seed(
    seed_positions: dict[int, np.ndarray],
    grown_ca: np.ndarray,
    max_r: float,
) -> dict[int, np.ndarray]:
    """Assign rigid-grown Ca sites to nearest DNA-bound seed."""
    if not seed_positions or len(grown_ca) == 0:
        return {}
    resseqs = list(seed_positions.keys())
    seeds = np.array([seed_positions[r] for r in resseqs])
    d = np.linalg.norm(grown_ca[:, None, :] - seeds[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    min_d = d.min(axis=1)
    by_seed: dict[int, list] = defaultdict(list)
    for pt, si, md in zip(grown_ca, nearest, min_d):
        if md <= max_r + 0.05:
            by_seed[resseqs[si]].append(pt)
    return {k: np.array(v) for k, v in by_seed.items()}


def count_shared_sites(
    seed_positions: dict[int, np.ndarray],
    lattice_by_seed: dict[int, np.ndarray],
    radius: float,
    outward_only: bool = True,
) -> tuple[int, int]:
    """Merge lattice Ca within radius; count sites claimed by ≥2 seeds."""
    items = []
    for resseq, ca_pts in lattice_by_seed.items():
        seed = seed_positions[resseq]
        for p in ca_pts:
            d = float(np.linalg.norm(p - seed))
            if d > radius + 0.05:
                continue
            if outward_only and d <= SEED_KEEP:
                continue
            items.append({"xyz": p, "source": resseq})
    if not items:
        return 0, 0
    merged = merge_points(items, MERGE_R)
    shared = sum(1 for m in merged if m["n"] >= 2)
    return shared, len(merged)


def dna_cryst_distances(
    dna_heavy: np.ndarray,
    seed_positions: dict[int, np.ndarray],
    lattice_by_seed: dict[int, np.ndarray],
    radius: float,
) -> dict[str, float | None]:
    """Min DNA–crystallite distance and outer-shell distance (peel-off proxy)."""
    if len(dna_heavy) == 0:
        return {"min_all": None, "min_shell": None, "outer_median": None}

    shell_pts = []
    outer_pts = []
    for resseq, ca_pts in lattice_by_seed.items():
        seed = seed_positions[resseq]
        for p in ca_pts:
            d = float(np.linalg.norm(p - seed))
            if SEED_KEEP < d <= radius + 0.05:
                shell_pts.append(p)
                if d > radius - 1.5:
                    outer_pts.append(p)

    if not shell_pts:
        return {"min_all": None, "min_shell": None, "outer_median": None}

    shell = np.array(shell_pts)
    dmat = np.linalg.norm(dna_heavy[:, None, :] - shell[None, :, :], axis=2)
    min_all = float(dmat.min())

    if outer_pts:
        outer = np.array(outer_pts)
        d_outer = np.linalg.norm(dna_heavy[:, None, :] - outer[None, :, :], axis=2)
        outer_median = float(np.median(d_outer.min(axis=0)))
    else:
        outer_median = None

    return {"min_all": min_all, "min_shell": min_all, "outer_median": outer_median}


def sweep_growth_radius(
    seed_positions: dict[int, np.ndarray],
    lattice_by_seed: dict[int, np.ndarray],
    observed_by_seed: dict[int, np.ndarray] | None,
    dna_heavy: np.ndarray,
    radii: list[float],
) -> list[dict]:
    rows = []
    for radius in radii:
        loc_a, loc_c, loc_384, loc_beta = [], [], [], []
        for resseq, seed in seed_positions.items():
            pts = lattice_by_seed.get(resseq, np.zeros((0, 3)))
            d = np.linalg.norm(pts - seed, axis=1)
            nb = pts[d <= radius + 0.05]
            sym = analyze_patch_symmetry(nb)
            if sym["a"] > 0:
                loc_a.append(sym["a"])
            if sym["c"] > 0:
                loc_c.append(sym["c"])
            if sym["frac_384"] > 0:
                loc_384.append(sym["frac_384"])
            if sym["beta"] is not None:
                loc_beta.append(sym["beta"])

        shared, merged = count_shared_sites(seed_positions, lattice_by_seed, radius)
        dist = dna_cryst_distances(dna_heavy, seed_positions, lattice_by_seed, radius)

        obs_a, obs_c = [], []
        obs_shared, obs_merged = 0, 0
        if observed_by_seed is not None:
            for resseq, seed in seed_positions.items():
                pts = observed_by_seed.get(resseq, np.zeros((0, 3)))
                d = np.linalg.norm(pts - seed, axis=1)
                nb = pts[d <= radius + 0.05]
                if len(nb) == 0 and resseq in lattice_by_seed:
                    # include seed Ca for observed patches
                    nb = np.array([seed])
                sym = analyze_patch_symmetry(nb)
                if sym["a"] > 0:
                    obs_a.append(sym["a"])
                if sym["c"] > 0:
                    obs_c.append(sym["c"])
            obs_shared, obs_merged = count_shared_sites(
                seed_positions, observed_by_seed, radius, outward_only=False
            )

        rows.append(
            {
                "radius": radius,
                "n_seeds": len(seed_positions),
                "lattice_ca": int(
                    sum(
                        int(
                            np.sum(
                                np.linalg.norm(pts - seed_positions[resseq], axis=1)
                                <= radius + 0.05
                            )
                        )
                        for resseq, pts in lattice_by_seed.items()
                    )
                ),
                "a_med": float(np.median(loc_a)) if loc_a else None,
                "a_max": float(max(loc_a)) if loc_a else None,
                "c_med": float(np.median(loc_c)) if loc_c else None,
                "c_max": float(max(loc_c)) if loc_c else None,
                "frac384_max": float(max(loc_384)) if loc_384 else None,
                "beta_med": float(np.median(loc_beta)) if loc_beta else None,
                "shared": shared,
                "merged": merged,
                "dna_min": dist["min_all"],
                "dna_outer_med": dist["outer_median"],
                "obs_a_med": float(np.median(obs_a)) if obs_a else None,
                "obs_c_med": float(np.median(obs_c)) if obs_c else None,
                "obs_shared": obs_shared,
                "obs_merged": obs_merged,
            }
        )
    return rows


def format_sweep_report(rows: list[dict]) -> str:
    lines = []
    lines.append("Growth-radius sweep — symmetry vs detachment proxies")
    lines.append("=" * 72)
    lines.append(
        "Lattice Ca: oriented COM cuts from Whewellite - xtl.pdb (includes 3.84 Å)."
    )
    lines.append(
        "Observed Ca: rigid grown model (CaOx_whewellite_noDNA.pdb, min Ca–Ca 6 Å)."
    )
    lines.append(
        "Shared sites: lattice Ca within R merged at 0.85 Å; n≥2 seeds agree."
    )
    lines.append(
        "DNA dist: min heavy-atom distance NUC ↔ outward Ca shell (peel proxy)."
    )
    lines.append("")
    lines.append(
        "R(Å)  latCa  a_med a_max  c_med c_max 384   β_med "
        "shared  merge  dna_min dna_out  obs_a  obs_c obs_sh"
    )
    lines.append("-" * 72)

    for row in rows:
        def f(v, w=5):
            return f"{v:{w}.2f}" if v is not None else f"{'—':>{w}}"

        lines.append(
            f"{row['radius']:5.1f}  "
            f"{row['lattice_ca']:5d}  "
            f"{f(row['a_med'])} {f(row['a_max'])}  "
            f"{f(row['c_med'])} {f(row['c_max'])} "
            f"{f(row['frac384_max'], 4)}  "
            f"{f(row['beta_med'])}  "
            f"{row['shared']:5d}  {row['merged']:5d}  "
            f"{f(row['dna_min'])} {f(row['dna_outer_med'])}  "
            f"{f(row['obs_a_med'])} {f(row['obs_c_med'])} "
            f"{row['obs_shared']:5d}"
        )

    lines.append("")
    lines.append("Reading guide")
    lines.append("-" * 72)
    lines.append("  ~10 Å (1 c-step):  local c symmetry should rise; a already present.")
    lines.append("  shared sites:    cross-seed lattice continuity (not detachment).")
    lines.append("  dna_min rising:  crystallite moving away from backbone (peel hint).")
    lines.append("  obs_* columns:   capped near 10 Å for rigid grown model.")
    lines.append("")
    lines.append("Qualitative thresholds (from lattice sweep):")
    for thr, label in [
        (0.35, "local a-translation median ≥ 0.35"),
        (0.20, "local c-translation median ≥ 0.20"),
        (0.30, "local c-translation median ≥ 0.30"),
        (5, "shared lattice sites ≥ 5"),
        (20, "shared lattice sites ≥ 20"),
    ]:
        if "translation" in label:
            key = "c_med" if "c-" in label else "a_med"
            hit = next((r["radius"] for r in rows if r[key] and r[key] >= thr), None)
        else:
            hit = next((r["radius"] for r in rows if r["shared"] >= thr), None)
        if hit is not None:
            lines.append(f"  First R ≥ {hit:.0f} Å: {label}")
        else:
            lines.append(f"  Not reached by {rows[-1]['radius']:.0f} Å: {label}")

    dna_thresh = 3.0
    peel = next(
        (r["radius"] for r in rows if r["dna_min"] and r["dna_min"] >= dna_thresh),
        None,
    )
    if peel is not None:
        lines.append(
            f"  First R ≥ {peel:.0f} Å: DNA min distance ≥ {dna_thresh:.1f} Å (looser contact)"
        )
    else:
        lines.append(
            f"  DNA min stays < {dna_thresh:.1f} Å through {rows[-1]['radius']:.0f} Å"
        )

    return "\n".join(lines) + "\n"


GROWTH_SEEDS = ROOT / "DNA_CaOx_growth.pdb"
GROWTH_30A = ROOT / "DNA_CaOx_growth_whewellite30A.pdb"
GROWTH_30A_RELAXED = ROOT / "DNA_CaOx_growth_whewellite30A_relaxed.pdb"


def load_growth_seed_positions(path: Path) -> dict[int, np.ndarray]:
    atoms, _ = parse_atoms(path)
    out = {}
    for a in atoms:
        if a["resname"] != "COM" or a["chain"] != "X":
            continue
        if a["name"].strip().upper() != "CA":
            continue
        out[a["resseq"]] = a["xyz"].copy()
    if not out:
        raise SystemExit(f"No COM seed Ca on chain X in {path}")
    return out


def load_whw_ca(path: Path):
    xyz, bfac = [], []
    for line in path.open():
        if not line.startswith("HETATM"):
            continue
        if line[17:20].strip() not in ("WHW", "COM"):
            continue
        if line[12:16].strip().upper() != "CA":
            continue
        xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        bfac.append(float(line[60:66]) if line[60:66].strip() else 20.0)
    return np.array(xyz), np.array(bfac)


def load_phosphate_xyz(path: Path) -> np.ndarray:
    pts = []
    for line in path.open():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[17:20].strip() != "NUC":
            continue
        name = line[12:16].strip().upper()
        el = (line[76:78].strip() or name[0]).upper()
        if name == "P" or el == "P":
            pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(pts) if pts else np.zeros((0, 3))


def crystallinity_index(sym: dict) -> float:
    return 0.35 * sym["a"] + 0.35 * sym["c"] + 0.30 * sym["frac_384"]


def phase_label(sym: dict, d_p: float | None = None) -> str:
    """
    COM ordering / nucleation propensity (not strict crystal identification).

    crystalline  — strong local COM-net (a + 3.84 Å + score)
    intermediate — nucleation site: partial COM registry or cluster ordering
    amorphous    — no detectable COM symmetry in 10 Å patch
    """
    min_frac = MIN_FRAC
    int_score = INT_SCORE
    cryst_score = CRYST_SCORE
    nuc_score = NUCLEATION_SCORE
    axis_min = NUCLEATION_AXIS_MIN
    if d_p is not None:
        if d_p < 10.0:
            min_frac = 0.05
            int_score = 0.08
            cryst_score = 0.16
            nuc_score = 0.05
            axis_min = 0.035
        elif d_p < 15.0:
            min_frac = 0.06
            int_score = 0.09
            cryst_score = 0.18
            nuc_score = 0.055
            axis_min = 0.04
        elif d_p < 22.0:
            nuc_score = 0.055
    score = crystallinity_index(sym)
    has_a = sym["a"] >= min_frac
    has_384 = sym["frac_384"] >= min_frac
    has_c = sym["c"] >= min_frac
    any_axis = (
        sym["a"] >= axis_min
        or sym["frac_384"] >= axis_min
        or sym["c"] >= axis_min
    )
    if has_a and has_384 and score >= cryst_score:
        return "crystalline"
    if (
        (has_a and has_384)
        or (has_a and has_c)
        or (has_384 and has_c)
        or score >= int_score
    ):
        return "intermediate"
    if any_axis and score >= nuc_score:
        return "intermediate"
    if d_p is not None and d_p < 18.0 and any_axis and score >= nuc_score * 0.85:
        return "intermediate"
    return "amorphous"


def per_ca_metrics(pts: np.ndarray, pxyz: np.ndarray, patch_r: float = 10.0):
    n = len(pts)
    print(f"Computing per-Ca crystallinity for {n} sites...", flush=True)
    d_p = np.linalg.norm(pxyz[:, None, :] - pts[None, :, :], axis=2).min(axis=0)
    scores = np.zeros(n)
    frac_a = np.zeros(n)
    frac_c = np.zeros(n)
    frac_384 = np.zeros(n)
    for i in range(n):
        if i and i % 200 == 0:
            print(f"  {i}/{n} Ca analyzed", flush=True)
        d = np.linalg.norm(pts - pts[i], axis=1)
        nb = pts[d <= patch_r + 0.05]
        sym = analyze_patch_symmetry(nb)
        scores[i] = crystallinity_index(sym)
        frac_a[i] = sym["a"]
        frac_c[i] = sym["c"]
        frac_384[i] = sym["frac_384"]
    phases = np.array(
        [
            phase_label(
                {"a": frac_a[i], "c": frac_c[i], "frac_384": frac_384[i]},
                d_p=float(d_p[i]),
            )
            for i in range(n)
        ]
    )
    return {
        "d_p": d_p,
        "score": scores,
        "frac_a": frac_a,
        "frac_c": frac_c,
        "frac_384": frac_384,
        "phase": phases,
    }


def domain_labels(pts: np.ndarray, cutoff: float = 18.0) -> np.ndarray:
    domains = cluster_domains(pts, cutoff=cutoff)
    labels = np.full(len(pts), -1, dtype=int)
    for di, ix in enumerate(domains):
        labels[ix] = di
    # singletons
    lone = np.where(labels < 0)[0]
    for i, idx in enumerate(lone):
        labels[idx] = len(domains) + i
    return labels


def axial_coordinate(pts: np.ndarray, seeds: dict[int, np.ndarray]) -> np.ndarray:
    s = np.array(list(seeds.values()))
    if len(s) < 2:
        return np.zeros(len(pts))
    origin = s[0]
    axis = s[-1] - s[0]
    n = np.linalg.norm(axis)
    if n < 1e-6:
        return np.zeros(len(pts))
    axis = axis / n
    return (pts - origin) @ axis


def make_crystallinity_heatmaps(
    pts: np.ndarray,
    metrics: dict,
    bfac: np.ndarray,
    domains: np.ndarray,
    axial: np.ndarray,
    out_dir: Path,
    tag: str,
):
    import matplotlib.pyplot as plt
    import sys

    sys.path.insert(0, str(ROOT))
    import matplotlib_config  # noqa: E402

    matplotlib_config.apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    d_p = metrics["d_p"]
    score = metrics["score"]
    phase = metrics["phase"]

    # 1) 2D histogram: distance vs crystallinity
    fig, ax = plt.subplots(figsize=(9, 5))
    h = ax.hist2d(
        d_p,
        score,
        bins=[30, 25],
        cmap="magma",
        range=[[0, 32], [0, 0.65]],
    )
    plt.colorbar(h[3], ax=ax, label="Ca count")
    ax.set_xlabel("Distance from phosphate P (Å)")
    ax.set_ylabel("Crystallinity index (0=amorphous, 1=ideal COM)")
    ax.set_title(f"{tag}: whewellite order vs backbone distance")
    ax.axvline(6, color="cyan", ls="--", lw=0.8, alpha=0.7, label="6 Å gel")
    ax.axvline(10, color="lime", ls="--", lw=0.8, alpha=0.7, label="10 Å local")
    ax.axvline(20, color="yellow", ls="--", lw=0.8, alpha=0.7, label="20 Å bulk")
    ax.legend(loc="upper right", fontsize=8)
    matplotlib_config.savefig(out_dir / f"{tag}_crystallinity_vs_distance.png")

    # 2) Phase fraction by distance shell
    shells = [(0, 6), (6, 10), (10, 20), (20, 30), (30, 50)]
    shell_names = ["0-6", "6-10", "10-20", "20-30", "30+"]
    phases_order = ["amorphous", "intermediate", "crystalline"]
    counts = np.zeros((len(shells), len(phases_order)))
    for si, (lo, hi) in enumerate(shells):
        mask = (d_p >= lo) & (d_p < hi)
        for pi, ph in enumerate(phases_order):
            counts[si, pi] = np.sum((phase == ph) & mask)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(shells))
    colors = {"amorphous": "#d95f02", "intermediate": "#7570b3", "crystalline": "#1b9e77"}
    for pi, ph in enumerate(phases_order):
        ax.bar(shell_names, counts[:, pi], bottom=bottom, label=ph, color=colors[ph])
        bottom += counts[:, pi]
    ax.set_ylabel("Ca count")
    ax.set_xlabel("Distance from phosphate (Å)")
    ax.set_title(f"{tag}: phase composition by shell")
    ax.legend()
    matplotlib_config.savefig(out_dir / f"{tag}_phase_by_shell.png")

    # 3) Spatial map: axial vs radial distance, colored by crystallinity
    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(
        axial,
        d_p,
        c=score,
        s=12,
        cmap="viridis",
        vmin=0,
        vmax=0.55,
        alpha=0.85,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Crystallinity index")
    ax.set_xlabel("Axial coordinate along seed strand (Å)")
    ax.set_ylabel("Distance from phosphate P (Å)")
    ax.set_title(f"{tag}: Ca sites (color = COM order)")
    matplotlib_config.savefig(out_dir / f"{tag}_spatial_crystallinity.png")

    # 4) Domain map
    fig, ax = plt.subplots(figsize=(9, 5))
    sc2 = ax.scatter(axial, d_p, c=domains, s=14, cmap="tab20", alpha=0.9, edgecolors="none")
    plt.colorbar(sc2, ax=ax, label="Domain ID")
    ax.set_xlabel("Axial coordinate (Å)")
    ax.set_ylabel("Distance from phosphate P (Å)")
    ax.set_title(f"{tag}: whewellite patch domains")
    matplotlib_config.savefig(out_dir / f"{tag}_domain_map.png")


def run_growth_analysis(
    pdb_path: Path,
    seed_path: Path,
    report_path: Path,
    heatmap_dir: Path,
    sweep_report: Path,
    tag: str | None = None,
):
    pts, bfac = load_whw_ca(pdb_path)
    seeds = load_growth_seed_positions(seed_path)
    pxyz = load_phosphate_xyz(pdb_path)
    lines = [
        f"Symmetry + crystallinity map — {pdb_path.name}",
        "=" * 62,
        f"WHW Ca sites: {len(pts)}",
        f"DNA-bound seeds: {len(seeds)} (from {seed_path.name})",
        f"Reference COM cell: a={COM_A}  c={COM_C}  beta={COM_BETA}",
        "Strict COM net: map tol "
        f"{TOL:.2f} Å; |v|-a < {LEN_A:.2f}; |v|-c < {LEN_C:.2f}; "
        f"|v|-3.84 < {LEN_384:.2f}; β within {BETA_TOL:.0f}° of {com_beta_acute():.1f}°.",
        "Crystalline: strongest COM-net pocket (a + 3.84 Å, score ≥ "
        f"{CRYST_SCORE:.2f}). Intermediate: nucleation site / partial COM order "
        f"(score ≥ {INT_SCORE:.2f} or weak axis ≥ {NUCLEATION_AXIS_MIN:.2f}).",
        "",
    ]

    domains_ix = cluster_domains(pts, cutoff=18.0)
    dom_lbl = domain_labels(pts, cutoff=18.0)
    lines.append(f"Spatial domains (Ca clusters): {len(domains_ix)}")
    for i, ix in enumerate(domains_ix):
        sub = pts[ix]
        sym = analyze_patch_symmetry(sub)
        lines.append(
            f"  domain {i+1}: {len(ix)} Ca  "
            f"cryst_idx={crystallinity_index(sym):.2f}  "
            f"a={sym['a']:.2f} c={sym['c']:.2f} 384={sym['frac_384']:.2f}"
        )
    lines.append("")

    metrics = per_ca_metrics(pts, pxyz, patch_r=10.0)
    metrics["phase"] = np.asarray(metrics["phase"], dtype=object)
    metrics, n_seed_up, seed_phase = upgrade_crystal_seed_phases(metrics, pts, bfac)
    n_radial = upgrade_radial_shell_phases(metrics, bfac, metrics["d_p"])
    com_reg = local_com_registry(pts, radius=10.0)
    pair_score, has_384, has_629, n_com = local_com_pair_correlation(pts)
    metrics["com_registry"] = com_reg
    metrics["pair_corr"] = pair_score
    dna_heavy = load_dna_heavy(pdb_path)
    r_axis = helix_axis_radius(pts, dna_heavy)
    r_phosphate = phosphate_surface_radius(pxyz, dna_heavy)
    metrics["r_axis"] = r_axis
    hotspot, hotspot_clusters = nucleation_hotspot_mask(
        pts,
        pair_score,
        has_384,
        has_629,
        n_com,
        metrics["d_p"],
        metrics["score"],
        r_axis=r_axis,
        r_phosphate=r_phosphate,
    )
    metrics["hotspot"] = hotspot
    n_hot = int(hotspot.sum())
    n_hot_dna = int((hotspot & (metrics["d_p"] < 12.0)).sum())
    if n_seed_up:
        lines.append(
            f"Crystal seed patch (bfac≈{SEED_CRYSTAL_BFAC:.0f} Å): "
            f"{n_seed_up} Ca upgraded via Ca–Ca distance fingerprint "
            f"(cluster phase={seed_phase or 'mixed'})."
        )
        lines.append("")
    if n_radial:
        lines.append(
            f"Radial shell zones (bfac {BFAC_GEL_COAT:.0f}/{BFAC_INTERMEDIATE:.0f}/"
            f"{BFAC_PRECRYSTAL:.0f}): {n_radial} Ca tagged as nucleation sites "
            "(partial COM-net + DNA distance)."
        )
        lines.append("")
    if n_hot:
        lines.append(
            f"Nucleation hotspots (COM pair-corr clusters, r_axis ≥ "
            f"{r_phosphate - 0.35:.1f} Å, interior core only): {n_hot} Ca in "
            f"{len(hotspot_clusters)} clusters ({n_hot_dna} within 12 Å of P)."
        )
        for ci, cl in enumerate(hotspot_clusters[:12]):
            lines.append(
                f"  cluster {ci + 1}: n={cl['n']}  d_P≈{cl['mean_d_p']:.1f} Å  "
                f"pair-corr={cl['mean_pair_corr']:.2f}"
            )
        if len(hotspot_clusters) > 12:
            lines.append(f"  ... {len(hotspot_clusters) - 12} more clusters")
        lines.append("")
    for ph in ("crystalline", "intermediate", "amorphous"):
        m = metrics["phase"] == ph
        n = int(m.sum())
        if n == 0:
            continue
        lines.append(
            f"  {ph:14s}: n={n:4d}  d_P median={np.median(metrics['d_p'][m]):.1f} Å  "
            f"score median={np.median(metrics['score'][m]):.3f}"
        )
        lines.append("")

    axial = axial_coordinate(pts, seeds)
    tag = (tag or pdb_path.stem).replace(".", "_")
    make_crystallinity_heatmaps(pts, metrics, bfac, dom_lbl, axial, heatmap_dir, tag)
    lines.append(f"Heatmaps written to {heatmap_dir}/")

    csv_path = heatmap_dir / f"{tag}_ca_metrics.csv"
    with csv_path.open("w") as f:
        f.write(
            "ca_index,d_p_A,crystallinity,frac_a,frac_c,frac_384,com_registry,pair_corr,hotspot,phase,domain,axial_A,shell_bfac\n"
        )
        for i in range(len(pts)):
            f.write(
                f"{i},{metrics['d_p'][i]:.3f},{metrics['score'][i]:.4f},"
                f"{metrics['frac_a'][i]:.4f},{metrics['frac_c'][i]:.4f},"
                f"{metrics['frac_384'][i]:.4f},{metrics['com_registry'][i]:.4f},"
                f"{metrics['pair_corr'][i]:.4f},{int(metrics['hotspot'][i])},"
                f"{metrics['phase'][i]},"
                f"{dom_lbl[i]},{axial[i]:.3f},{bfac[i]:.1f}\n"
            )
    lines.append(f"Per-Ca metrics: {csv_path}")
    lines.append("")

  # radius sweep with 4 growth seeds
    seed_positions = seeds
    max_r = max(SWEEP_RADII)
    xtl_atoms, cryst = parse_atoms(XTL)
    nmax = max(2, int(math.ceil(max_r / min(cryst["a"], cryst["c"]))) + 1)
    expanded, (av, bv, cv) = expand_crystal(xtl_atoms, cryst, nmax=nmax)
    ref = next(a for a in xtl_atoms if a["element"].upper() == "CA")

    lattice_by_seed = {}
    growth_atoms, _ = parse_atoms(seed_path)
    strand_cas = [
        a
        for a in growth_atoms
        if a["resname"] == "COM" and a["chain"] == "X" and a["name"].strip().upper() == "CA"
    ]
    strand_cas.sort(key=lambda a: a["resseq"])
    helix_o = np.mean([a["xyz"] for a in strand_cas], axis=0)
    for i, ca in enumerate(strand_cas):
        t, out_vec = local_frame(strand_cas, i, helix_o)
        R1 = rotation_from_to(av, t)
        best_ang, best_sc = 0.0, -1e9
        for ang in np.linspace(0, 2 * math.pi, 72, endpoint=False):
            R = rotation_around(t, ang) @ R1
            sc = float(np.dot(cv @ R.T, out_vec))
            if sc > best_sc:
                best_sc, best_ang = sc, ang
        R = rotation_around(t, best_ang) @ R1
        ca_list = []
        for a in expanded:
            if a["element"].upper() != "CA":
                continue
            xyz = (a["xyz"] - ref["xyz"]) @ R.T + ca["xyz"]
            if float(np.linalg.norm(xyz - ca["xyz"])) <= max_r + 0.05:
                ca_list.append(xyz)
        lattice_by_seed[ca["resseq"]] = np.array(ca_list)

    observed_by_seed = observed_ca_by_seed(seed_positions, pts, max_r)
    dna_heavy = load_dna_heavy(pdb_path)
    sweep_rows = sweep_growth_radius(
        seed_positions, lattice_by_seed, observed_by_seed, dna_heavy, SWEEP_RADII
    )
    sweep_text = format_sweep_report(sweep_rows)
    sweep_report.write_text(sweep_text)
    lines.append(sweep_text)

    text = "\n".join(lines) + "\n"
    report_path.write_text(text)
    print(text)


def run_legacy_no_dna():
    pts, chain, bfac = load_ca(PDB)
    lines = []
    lines.append("Symmetry search — CaOx_whewellite_noDNA.pdb")
    lines.append("=" * 62)
    lines.append(f"Ca sites: {len(pts)}   (chain X seeds + chain Z grown)")
    lines.append(
        f"Reference COM cell: a={COM_A}  b={COM_B}  c={COM_C}  beta={COM_BETA}"
    )
    lines.append("Whewellite space group (literature): P2_1/n  (or P2_1/c setting)")
    lines.append("")

    domains = cluster_domains(pts, cutoff=22.0)
    lines.append(f"Spatial domains (separate crystallites): {len(domains)}")
    for i, ix in enumerate(domains):
        lines.append(f"  domain {i+1}: {len(ix)} Ca")
    lines.append("")

    all_hits = []
    for di, ix in enumerate(domains):
        dpts = pts[ix]
        lines.append(f"--- Domain {di+1}  ({len(dpts)} Ca) ---")
        raw = candidate_vectors(dpts)
        clustered = cluster_vectors(raw, rad=0.60)[:40]
        scored = []
        for vec, pop in clustered:
            L = np.linalg.norm(vec)
            if L < 5.4 or L > 15.8:
                continue
            frac = translation_score(dpts, vec)
            axis, tlen, dlen = nearest_com_axis(vec)
            scored.append((frac, L, axis, dlen, vec, pop))
        scored.sort(key=lambda t: -t[0])

        lines.append(
            f"Best translation vectors (fraction of Ca reproduced within {TOL:.2f} Å):"
        )
        lines.append("   frac    |v|     COM-axis   |v|-axis    vector")
        kept = []
        for frac, L, axis, dlen, vec, pop in scored[:12]:
            ideal = {"a": COM_A, "b": COM_B, "c": COM_C}[axis]
            lines.append(
                f"  {frac:5.2f}   {L:6.2f} Å   {axis:>2s} {ideal:6.2f}   "
                f"Δ={dlen:5.2f}   [{vec[0]:7.2f} {vec[1]:7.2f} {vec[2]:7.2f}]"
            )
            if frac >= 0.18:
                kept.append((frac, L, axis, vec))
                all_hits.append((di + 1, frac, L, axis, vec))

        picked = {}
        for frac, L, axis, vec in kept:
            if axis not in picked and dlen_ok(L, axis):
                picked[axis] = vec
        if "a" in picked and "c" in picked:
            ang = angle_between(picked["a"], picked["c"])
            ang = min(ang, 180 - ang)
            lines.append(
                f"  Angle between recovered a and c: {ang:.1f}°  "
                f"(COM beta = {COM_BETA}° or 180-beta = {180-COM_BETA:.1f}°)"
            )
        if "a" in picked and "b" in picked:
            ang = angle_between(picked["a"], picked["b"])
            lines.append(f"  Angle between recovered a and b: {ang:.1f}°  (should be ~90°)")
        if "b" in picked and "c" in picked:
            ang = angle_between(picked["b"], picked["c"])
            lines.append(f"  Angle between recovered b and c: {ang:.1f}°  (should be ~90°)")

        origin = dpts.mean(axis=0)
        lines.append("Point / screw tests (fraction of Ca mapped onto another Ca):")
        inv = inversion_score(dpts, origin)
        lines.append(f"  inversion through domain centroid: {inv:.2f}")

        for label, vec in picked.items():
            ax = vec / np.linalg.norm(vec)
            f2 = twofold_score(dpts, origin, ax)
            f21 = screw21_score(dpts, origin, ax, np.linalg.norm(vec))
            lines.append(
                f"  2-fold about {label}: {f2:.2f}    "
                f"2_1 screw along {label} (t=|v|/2): {f21:.2f}"
            )
        lines.append("")

    seeds_arr = pts[chain == "X"]
    if len(seeds_arr) == 0:
        seeds_arr = pts[np.abs(bfac - 20.0) < 0.1]
    lines.append(f"--- Local 10 Å patches around {len(seeds_arr)} seed Ca ---")
    loc_a, loc_c, loc_384, loc_beta = [], [], [], []
    for s in seeds_arr:
        d = np.linalg.norm(pts - s, axis=1)
        nb = pts[d <= 10.2]
        sym = analyze_patch_symmetry(nb)
        if sym["a"] > 0:
            loc_a.append(sym["a"])
        if sym["c"] > 0:
            loc_c.append(sym["c"])
        if sym["frac_384"] > 0:
            loc_384.append(sym["frac_384"])
        if sym["beta"] is not None:
            loc_beta.append(sym["beta"])

    summarize_symmetry_list(loc_a, "local a-translation (~6.29 Å)", lines)
    summarize_symmetry_list(loc_c, "local c-translation (~10.12 Å)", lines)
    summarize_symmetry_list(loc_384, "local 3.84 Å Ca–Ca (intra-cell)", lines)
    if loc_beta:
        arr = np.array(loc_beta)
        lines.append(
            f"  local a∧c angle: median {np.median(arr):.1f}°  "
            f"(COM β = {COM_BETA}° or {180-COM_BETA:.1f}°)"
        )
    lines.append("")

    lines.append("Interpretation")
    lines.append("-" * 62)
    if all_hits:
        by_axis = defaultdict(list)
        for _d, frac, L, axis, _v in all_hits:
            by_axis[axis].append((frac, L))
        for axis in ("a", "b", "c"):
            if axis not in by_axis:
                continue
            fracs = [f for f, _ in by_axis[axis]]
            lens = [L for _, L in by_axis[axis]]
            lines.append(
                f"  Recovers COM {axis}-repeat: best frac={max(fracs):.2f}, "
                f"|v| = {np.median(lens):.2f} Å "
                f"(ideal { {'a':COM_A,'b':COM_B,'c':COM_C}[axis] } Å)"
            )
    lines.append("")
    lines.append("This is not a single periodic crystal. DNA is helical, so each")
    lines.append("phosphate-bound seed grew a locally monoclinic COM patch. After")
    lines.append("DNA is removed those patches still carry:")
    lines.append("  * the 6.29 Å a-chain (phosphate-templated)")
    lines.append("  * the ~10.1 Å c step (outward from the helix)")
    lines.append("  * the 3.84 Å intra-cell Ca–Ca is omitted (min Ca–Ca is 6 Å)")
    lines.append("A global P2_1/n operation will not map the whole model onto")
    lines.append("itself. Locally (one strand, ~10 Å) the symmetry is whewellite.")
    lines.append("Shared Ca sites (B=10 in the PDB) are where two patches agreed")
    lines.append("on the same lattice point — the start of crystal continuity.")
    lines.append("")
    lines.append(f"See also: {SWEEP_REPORT.name} (radius sweep)")

    text = "\n".join(lines) + "\n"
    REPORT.write_text(text)
    print(text)

    # --- radius sweep ---
    max_r = max(SWEEP_RADII)
    seed_positions, lattice_by_seed = build_oriented_lattice_ca(max_r)

    grown_ca = pts[chain == "Z"]
    if len(grown_ca) == 0:
        grown_ca = pts[chain != "X"]
    observed_by_seed = observed_ca_by_seed(seed_positions, grown_ca, max_r)
    dna_heavy = load_dna_heavy(DNA_PDB)

    sweep_rows = sweep_growth_radius(
        seed_positions,
        lattice_by_seed,
        observed_by_seed,
        dna_heavy,
        SWEEP_RADII,
    )
    sweep_text = format_sweep_report(sweep_rows)
    SWEEP_REPORT.write_text(sweep_text)
    print(sweep_text)


def relabel_metrics_csvs(out_dir: Path) -> None:
    """Apply current phase_label cuts to existing per-Ca CSVs."""
    import csv
    from collections import Counter

    paths = sorted(out_dir.glob("*_ca_metrics.csv"))
    if not paths:
        raise SystemExit(f"No *_ca_metrics.csv in {out_dir}")
    for path in paths:
        rows = list(csv.DictReader(path.open()))
        if not rows:
            continue
        old = Counter(r["phase"] for r in rows)
        for r in rows:
            d_p = float(r["d_p_A"]) if r.get("d_p_A") else None
            r["phase"] = phase_label(
                {
                    "a": float(r["frac_a"]),
                    "c": float(r["frac_c"]),
                    "frac_384": float(r["frac_384"]),
                },
                d_p=d_p,
            )
        new = Counter(r["phase"] for r in rows)
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(
            f"{path.name}: {dict(old)} -> {dict(new)}",
            flush=True,
        )


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Whewellite symmetry and crystallinity maps")
    ap.add_argument(
        "--pdb",
        type=Path,
        default=None,
        help="Growth model PDB (WHW + NUC). Default: 30A relaxed if present.",
    )
    ap.add_argument(
        "--seeds",
        type=Path,
        default=GROWTH_SEEDS,
        help="Seed COM positions (chain X)",
    )
    ap.add_argument(
        "--legacy",
        action="store_true",
        help="Analyze CaOx_whewellite_noDNA.pdb (original script)",
    )
    ap.add_argument(
        "--heatmap-dir",
        type=Path,
        default=ROOT / "figures" / "crystallinity",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Output name prefix for reports and figures (default: PDB stem)",
    )
    ap.add_argument(
        "--relabel",
        action="store_true",
        help="Rewrite phase column in existing *_ca_metrics.csv using current cuts "
        "(no per-Ca recompute).",
    )
    args = ap.parse_args()

    if args.legacy:
        run_legacy_no_dna()
        return

    if args.relabel:
        relabel_metrics_csvs(args.heatmap_dir)
        return

    pdb = args.pdb
    if pdb is None:
        if GROWTH_30A_RELAXED.exists():
            pdb = GROWTH_30A_RELAXED
        elif GROWTH_30A.exists():
            pdb = GROWTH_30A
        else:
            raise SystemExit("No growth model PDB found; pass --pdb")

    tag = args.tag or pdb.stem
    report = ROOT / f"{tag}_symmetry.txt"
    sweep = ROOT / f"{tag}_symmetry_sweep.txt"
    run_growth_analysis(pdb, args.seeds, report, args.heatmap_dir, sweep, tag=tag)


if __name__ == "__main__":
    main()
