#!/bin/bash
# Final detection config (thresholds tuned on night_test_01, evaluated on night_test_02/03).
# Usage: bash detection/run_detection.sh [nights_folder] [output_folder] [extra flags...]
#   e.g. bash detection/run_detection.sh data/nights detection/output
#        bash detection/run_detection.sh data/nights detection/output --recompute
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
NIGHTS="${1:-data/nights}"
OUT="${2:-detection/output}"
shift 2 2>/dev/null || true

python "$HERE/detect.py" --nights "$NIGHTS" --out "$OUT" \
  --refine --exclusive --nms \
  --class-gap snoring=3 voices=2 dog_bark=2 \
  --thresholds "$HERE/thresholds.json" \
  --extended "$@"
