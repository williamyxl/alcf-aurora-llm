#!/usr/bin/env bash
# Central site configuration for every recipe in this repo. Source it; never edit recipes for paths.
#
#   source "$(git rev-parse --show-toplevel)/common/env.sh"      # from a login node
#   source "$REPO/common/env.sh"                                    # from a PBS script (REPO exported by qsub -v)
#
# Override anything in common/env.local.sh (gitignored) -- e.g. a different project, data dir, or account.

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO="${REPO:-$(cd "$_here/.." && pwd)}"
[ -f "$_here/env.local.sh" ] && source "$_here/env.local.sh"
[ -f "$_here/lib.sh" ] && source "$_here/lib.sh"

# ---- accounting / queues -----------------------------------------------------------------
export PBS_ACCOUNT="${PBS_ACCOUNT:-MatSciAI}"
export PBS_FS="${PBS_FS:-flare}"

# ---- data (large, not in git). One tree, per-model links: $DATA/models/<model>/<variant> -------
export DATA="${DATA:-$REPO/data}"
export MODELS="${MODELS:-$DATA/models}"
export OVERLAY="${OVERLAY:-$DATA/overlay}"        # transformers 5.23 + rope shim + patched vllm_xpu_kernels
export LLAMACPP_BUILDS="${LLAMACPP_BUILDS:-$DATA/llamacpp}"
export LOGS="${LOGS:-$REPO/logs}"                  # gitignored
mkdir -p "$LOGS" 2>/dev/null || true

# ---- vLLM engine: Aurora frameworks module (see docs/ENGINES.md) -----------------------------
#   2026 : vllm 0.26.1, torch 2.13 (default; XPU graph, Laguna, Nemotron-3-Ultra with patched kernel)
#   2025 : vllm 0.15,  torch 2.10 (legacy gpt-oss API recipe; needs VLLM_XPU_FUSED_MOE_USE_REF=1)
export FW="${FW:-2026}"
vllm_env() {
  module use /soft/modulefiles
  if [ "$FW" = "2025" ]; then
    module use /opt/aurora/26.26.0/frameworks/modulefiles
    module load frameworks
    FWROOT=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1
  else
    module load frameworks/2026.1.0
    FWROOT=/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0
  fi
  export FWROOT FWPY="$FWROOT/bin/python" VLLM_BIN="$FWROOT/bin/vllm"
  export HF_HUB_OFFLINE=1 TMPDIR=/tmp ZE_FLAT_DEVICE_HIERARCHY=FLAT
  # Fix 1: framework triton has intel backend; user's triton (if installed) shadows it and lacks intel backend
  export PYTHONPATH="$FW/lib/python3.12/site-packages:${PYTHONPATH:-}"
  # Fix 2: dummy torchcodec module to avoid broken native library in user's site-packages
  export PYTHONPATH="$REPO/dummy_torchcodec:$PYTHONPATH"
  # Fix 3: framework triton needs TRITON_BACKENDS_IN_TREE=1 to discover intel backend from directory
  export TRITON_BACKENDS_IN_TREE=1
  export TRITON_INTEL_GPU_ARCH=xpu
}
# Compose PYTHONPATH from optional overlays:  OVERLAY_TF=1 (transformers 5.23 + Beam/Muse rope shim)
#                                              OVERLAY_KERNELS=1 (vllm_xpu_kernels with top-k=22 _moe_C)
vllm_overlays() {
  local pp=""
  [ "${OVERLAY_KERNELS:-0}" = "1" ] && pp="$OVERLAY/kernels:$pp"
  [ "${OVERLAY_TF:-0}" = "1" ] && pp="$pp$OVERLAY/shim:$OVERLAY/py:"
  [ -n "$pp" ] && export PYTHONPATH="${pp}${PYTHONPATH:-}"
  [ "${XPUGRAPH:-0}" = "1" ] && export VLLM_XPU_ENABLE_XPU_GRAPH=1
  return 0
}
# Pin vLLM to a tile set: vllm_tiles 0 7  -> tiles 0..7
vllm_tiles() { export ZE_AFFINITY_MASK="$(seq -s, "$1" "$2")"; }
vllm_cache() { export TRITON_CACHE_DIR="/tmp/tc_${PBS_JOBID:-$$}_$1" SYCL_CACHE_DIR="/tmp/sc_${PBS_JOBID:-$$}_$1"; mkdir -p "$TRITON_CACHE_DIR" "$SYCL_CACHE_DIR"; }

# ---- llama.cpp SYCL: two prebuilt trees (see docs/ENGINES.md) --------------------------------
#   gptoss : master 9d57ce4 (2026-08-15)         inkling : PR 25731 @ 368b24c35 (2026-08-12): inkling + nemotron + laguna
export LLAMA="${LLAMA:-inkling}"
llamacpp_env() {
  module load oneapi/release/2025.3.1
  LLAMA_BIN="$LLAMACPP_BUILDS/$LLAMA/build/bin"
  # binaries were built in another directory; their RUNPATH is stale, so resolve libs explicitly
  export LLAMA_BIN LD_LIBRARY_PATH="$LLAMA_BIN:${LD_LIBRARY_PATH:-}"
  export ZE_FLAT_DEVICE_HIERARCHY=FLAT ONEAPI_DEVICE_SELECTOR=level_zero:gpu GGML_SYCL_ENABLE_VMM=0 ZES_ENABLE_SYSMAN=1
  export FWPY="${FWPY:-/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python}"   # stdlib-only helpers
}

# ---- proxy (compute nodes need it for any download) ------------------------------------------
export_proxy() { export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128 ftp_proxy=http://proxy.alcf.anl.gov:3128; }

# PBS header values, for scripts that generate jobs
pbs_common() { echo "-A $PBS_ACCOUNT -l filesystems=$PBS_FS -l place=scatter"; }
