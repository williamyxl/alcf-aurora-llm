#!/usr/bin/env bash
# Run ONE vLLM engine test inside a PBS job (or interactive node):
#   single-stream (TTFT/prefill/decode) + N-way concurrency + quality check, via common/vllm/serve_conc.py
#
#   common/vllm/run_bench.sh <tag> <model_dir> <tiles> [serve_conc.py args...]
#
# Env switches:  FW=2026|2025   XPUGRAPH=1   OVERLAY_TF=1   OVERLAY_KERNELS=1
# Output: $LOGS/<tag>.out  (look for SERVE_JSON=...)
set -o pipefail
TAG=$1; MODEL=$2; TILES=$3; shift 3
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO/common/env.sh"
vllm_env; vllm_overlays; vllm_tiles 0 $((TILES-1)); vllm_cache "$TAG"
echo "host=$(hostname) $(date -Is) tag=$TAG tiles=$TILES FW=$FW XPUGRAPH=${XPUGRAPH:-0} OVERLAY_TF=${OVERLAY_TF:-0} OVERLAY_KERNELS=${OVERLAY_KERNELS:-0}"
"$FWPY" "$REPO/common/vllm/serve_conc.py" --model "$MODEL" --tag "$TAG" "$@" > "$LOGS/$TAG.out" 2>&1
rc=$?; echo "rc=$rc $(date -Is)"
grep -hE "QUALITY_TEXT=|^SINGLE |SERVE_JSON=" "$LOGS/$TAG.out" | cut -c1-900
exit $rc
