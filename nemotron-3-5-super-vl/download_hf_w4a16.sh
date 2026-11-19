#!/usr/bin/env bash
# vLLM checkpoint: RedHatAI INT4 (w4a16 compressed-tensors, 293 GB). Needs OVERLAY_KERNELS=1 (top-k=22 _moe_C). Login node.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; source "$REPO_ROOT/common/env.sh"
"$REPO_ROOT/common/tools/hf_download.sh" RedHatAI/NVIDIA-Nemotron-3-5-Super-VL-8B-quantized.w4a16 "$DATA/store/nemotron-3-5-super-vl/w4a16"
mkdir -p "$MODELS/nemotron-3-5-super-vl"; ln -sfn "$DATA/store/nemotron-3-5-super-vl/w4a16" "$MODELS/nemotron-3-5-super-vl/w4a16-hf"
