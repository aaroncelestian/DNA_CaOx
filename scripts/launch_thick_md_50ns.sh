#!/bin/bash
# 50 ns NVT MD at 350 K — thick (5-row) templating gel.
# Monitor: tail -f logs/templating_thick_md_50ns.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
exec >> logs/templating_thick_md_50ns.log 2>&1
echo "=== thick gel MD 50 ns $(date -u +%Y-%m-%dT%H:%MZ) ==="
.venv/bin/python scripts/run_templating_thick.py \
  --md-only \
  --md-ns 50 \
  --md-temperature 350
echo "=== finished $(date -u +%Y-%m-%dT%H:%MZ) ==="
