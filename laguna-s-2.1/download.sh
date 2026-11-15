#!/usr/bin/env bash
# Download checkpoints for laguna-s-2.1. Edit/run only the lines you need (multi-hundred-GB pulls). Login node.
REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"; source "$REPO/common/env.sh"
D="$DATA/store/laguna-s-2.1"; mkdir -p "$MODELS/laguna-s-2.1"
"$REPO/common/tools/hf_download.sh" poolside/Laguna-S-2.1 "$D/bf16"                      # 219 GB  (the recommended checkpoint on PVC)
ln -sfn "$D/bf16" "$MODELS/laguna-s-2.1/bf16-hf"
# FP8 (131 GB) loads but is ~5x slower and unreliable on PVC -- not recommended:
# common/tools/hf_download.sh poolside/Laguna-S-2.1-FP8 "$D/fp8"; ln -sfn "$D/fp8" "$MODELS/laguna-s-2.1/fp8-hf"
