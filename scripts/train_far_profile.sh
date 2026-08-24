#!/usr/bin/env bash
# MRNF far-scene profile: best measured reconstruction for large scenes with distant
# background (verified on Giant Bicycle @3M/@5M and torg @3M; see MRNF_PLATEAU_ANALYSIS_OX.md).
# Compact/indoor scenes should use plain defaults instead - this profile costs them 0.001-0.005 SSIM.
#
# Usage: scripts/train_far_profile.sh <dataset dir> <output dir> [max_cap] [iters] [extra args...]
set -euo pipefail
DATA=${1:?dataset dir}; OUT=${2:?output dir}; CAP=${3:-3000000}; ITERS=${4:-30000}; shift $(( $# > 4 ? 4 : $# ))
BIN=${LFS_BIN:-$(dirname "$0")/../build/LichtFeld-Studio}
REPO=$(cd "$(dirname "$0")/.." && pwd)
FILL=$((ITERS / 2))
DOSE=2000; [ "$CAP" -le 1500000 ] && DOSE=700
LFS_EXP_GROWTH_RATIO=1 \
LFS_EXP_RATIO_POW=0.75 \
LFS_EXP_FILL_ITER=$FILL \
LFS_EXP_SEED_FAR=1 \
LFS_EXP_SEED_DOSE=$DOSE \
"$BIN" \
  -d "$DATA" --output-path "$OUT" \
  --strategy mrnf --max-cap "$CAP" \
  --config "$REPO/eval/mrnf_far_profile_params.json" \
  -i "$ITERS" --train --headless --eval "$@"
