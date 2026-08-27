#!/bin/bash
# Detached 2 µs NVT MD at 350 K for 15-row templating gel.
# Monitor: tail -f logs/templating_15shell_md_2us_350K.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
exec >> logs/templating_15shell_md_2us_350K.log 2>&1
echo "=== MD launch $(date -u +%Y-%m-%dT%H:%MZ) ==="
.venv/bin/python scripts/run_templating_15shell.py \
  --md-only \
  --md-us 2 \
  --md-temperature 350 \
  --skip-score \
  --skip-export
echo "=== MD finished $(date -u +%Y-%m-%dT%H:%MZ) ==="
