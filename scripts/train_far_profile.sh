#!/usr/bin/env bash
# Convenience wrapper; the profile lives entirely in eval/mrnf_far_profile_params.json.
# Equivalent plain CLI:
#   LichtFeld-Studio -d <data> --strategy mrnf --max-cap 3000000 \
#     --config eval/mrnf_far_profile_params.json -i 30000 --train --headless --eval
# For iteration counts other than 30k set fill_pacing_iter to half the run in the config;
# for caps <=1.5M set far_seed_dose to ~700. Compact/indoor scenes: use plain defaults.
set -euo pipefail
DATA=${1:?dataset dir}; OUT=${2:?output dir}; CAP=${3:-3000000}; ITERS=${4:-30000}; shift $(( $# > 4 ? 4 : $# ))
REPO=$(cd "$(dirname "$0")/.." && pwd)
BIN=${LFS_BIN:-$REPO/build/LichtFeld-Studio}
"$BIN" -d "$DATA" --output-path "$OUT" --strategy mrnf --max-cap "$CAP" \
  --config "$REPO/eval/mrnf_far_profile_params.json" -i "$ITERS" --train --headless --eval "$@"
