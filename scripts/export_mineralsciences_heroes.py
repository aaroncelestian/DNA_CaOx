#!/usr/bin/env python3
"""
Export lightweight DNA–CaOx hero packs for the MineralSciences homepage.

Writes slim JSON (DNA + Ca + optional oxalate/water; no envelopes) with the
viewer setCoatView() display defaults baked in.

Usage (from DNA_CaOx repo root):
  python3 scripts/export_mineralsciences_heroes.py
  python3 scripts/export_mineralsciences_heroes.py --out /path/to/MineralSciences/hero
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DATA = ROOT / "viewer" / "model-data.js"
DEFAULT_OUT = Path.home() / "Documents" / "GitHub" / "MineralSciences" / "hero"

HERO_KEYS = ("slab", "shell15")

DISPLAY = {
    "style": "cloud",
    "color": "distance",
    "view": "side",
    "dmin": 0,
    "dmax": 75,
    "rmax": 75,
    "slab": 75,
    "phases": [True, True, True],
    "showDnaRibbons": True,
    "showDnaPairs": True,
    "showPhosphates": True,
    "showSeeds": True,
    "showOxalate": True,
    "showWater": True,
    "showHotspots": True,
    "showEnvelopes": False,
    "autoRotate": True,
    "autoRotateSpeed": 0.45,
}

CA_KEYS = ("x", "y", "z", "phase", "dP", "radial", "hotspot", "score")


def load_models(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    m = re.search(
        r"window\.DNA_CAOX_MODELS\s*=\s*(\{.*\});\s*\n?window\.DNA_CAOX_MODEL",
        text,
        re.S,
    )
    if not m:
        raise SystemExit(f"Could not parse MODELS from {path}")
    return json.loads(m.group(1))


def slim_pack(model: dict) -> dict:
    ca_src = model.get("ca") or {}
    ca = {k: ca_src[k] for k in CA_KEYS if k in ca_src}
    return {
        "geometry": model["geometry"],
        "title": model["title"],
        "cut": model.get("cut", ""),
        "source": model.get("source", ""),
        "helix": model.get("helix", {}),
        "seedRadius": model.get("seedRadius"),
        "cutKind": model.get("cutKind", "spheres"),
        "strands": model.get("strands", []),
        "pairs": model.get("pairs", []),
        "seeds": model.get("seeds", []),
        "ca": ca,
        "oxalate": model.get("oxalate", []) if DISPLAY["showOxalate"] else [],
        "water": (
            model.get("water", {"x": [], "y": [], "z": []})
            if DISPLAY["showWater"]
            else {"x": [], "y": [], "z": []}
        ),
        "display": DISPLAY,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--geom", nargs="*", default=list(HERO_KEYS))
    args = ap.parse_args()
    models = load_models(MODEL_DATA)
    args.out.mkdir(parents=True, exist_ok=True)
    for key in args.geom:
        if key not in models:
            print(f"skip unknown {key}", flush=True)
            continue
        pack = slim_pack(models[key])
        path = args.out / f"{key}.json"
        path.write_text(json.dumps(pack, separators=(",", ":")), encoding="utf-8")
        n_ca = len(pack["ca"]["x"])
        print(f"wrote {path} ({path.stat().st_size / 1024:.1f} KB, {n_ca} Ca)", flush=True)


if __name__ == "__main__":
    main()
