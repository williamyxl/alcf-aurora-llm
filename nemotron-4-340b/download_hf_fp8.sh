#!/usr/bin/env bash
# OPTIONAL and NOT recommended: vLLM FP8 checkpoint (350 GB) serves but produces garbage output on PVC. Kept for reproducing that finding.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; source "$REPO_ROOT/common/env.sh"
"$REPO_ROOT/common/tools/hf_download.sh" mgoin/Nemotron-4-340B-Instruct-FP8-Dynamic "$DATA/store/nemotron-4-340b/fp8"
mkdir -p "$MODELS/nemotron-4-340b"; ln -sfn "$DATA/store/nemotron-4-340b/fp8" "$MODELS/nemotron-4-340b/fp8-hf"
