#!/bin/bash
# Download unsloth Nemotron-3-Ultra MXFP4_MOE GGUF (native MoE quant, ~352 GB, 9 shards).
# Resumable. Run on a login/UAN node (proxy internet). hf skips complete shards on rerun.
#
# MXFP4_MOE mirrors the gpt-oss F4_hbm recipe family: MoE experts stay MXFP4 and are offloaded
# to CPU (-ncmoe 99); dense/attention/Mamba run on one GPU tile.
#
#   bash download_nemotron_gguf.sh
#   (or: QUANT=UD-IQ2_M bash download_nemotron_gguf.sh   # smaller ~194 GB, lower quality)

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; source "$REPO_ROOT/common/env.sh"; export_proxy
export WORKDIR="$DATA/store/nemotron-3-ultra"
export HF_REPO=${HF_REPO:-unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF}
export QUANT=${QUANT:-UD-IQ2_M}   # tested recipe; MXFP4_MOE (~352 GB) also available
export OUT_DIR=${OUT_DIR:-$WORKDIR/gguf}
export HF_HOME=${HF_HOME:-$DATA/.hf_cache}
export http_proxy=${http_proxy:-http://proxy.alcf.anl.gov:3128}
export https_proxy=${https_proxy:-http://proxy.alcf.anl.gov:3128}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}

# floor bytes per quant (rough)
case "$QUANT" in
  MXFP4_MOE)  MIN_OK_BYTES=${MIN_OK_BYTES:-330000000000};;
  UD-IQ2_M)   MIN_OK_BYTES=${MIN_OK_BYTES:-180000000000};;
  UD-IQ1_M)   MIN_OK_BYTES=${MIN_OK_BYTES:-175000000000};;
  *)          MIN_OK_BYTES=${MIN_OK_BYTES:-100000000000};;
esac

mkdir -p "$OUT_DIR" "$HF_HOME"
vllm_env >/dev/null 2>&1; export PATH="$FWROOT/bin:$PATH"   # hf CLI + huggingface_hub from the frameworks module

echo "host=$(hostname) date=$(date -Is) HF_REPO=$HF_REPO QUANT=$QUANT OUT_DIR=$OUT_DIR"
df -h "$WORKDIR" | tail -1

echo "Starting hf download (resumable), pattern ${QUANT}/* ..."
set +e
hf download "$HF_REPO" --include "${QUANT}/*" --local-dir "$OUT_DIR"
RC=$?
set -e
echo "hf_exit=$RC"

QDIR="$OUT_DIR/$QUANT"
echo "=== ${QUANT} contents ==="
ls -la "$QDIR" 2>/dev/null | head -20
N_GGUF=$(ls "$QDIR"/*.gguf 2>/dev/null | wc -l)
BYTES=$(du -sb "$QDIR" 2>/dev/null | awk '{print $1}')
echo "N_GGUF=$N_GGUF TOTAL_GB=$(awk -v b="${BYTES:-0}" 'BEGIN{printf "%.1f", b/1e9}')"

# first shard path (llama.cpp auto-loads the rest of a split set from the -00001 file)
FIRST=$(ls "$QDIR"/*-00001-of-*.gguf 2>/dev/null | head -1)
[ -z "$FIRST" ] && FIRST=$(ls "$QDIR"/*.gguf 2>/dev/null | head -1)
echo "FIRST_SHARD=$FIRST"

if [ "$N_GGUF" -ge 1 ] && [ "${BYTES:-0}" -ge "$MIN_OK_BYTES" ]; then
  # convenience symlink to the first shard
  mkdir -p "$MODELS/nemotron-3-ultra"; ln -sfn "$QDIR" "$MODELS/nemotron-3-ultra/${QUANT}-gguf"
  echo "DOWNLOAD_OK  -> $MODELS/nemotron-3-ultra/${QUANT}-gguf"
  exit 0
else
  echo "DOWNLOAD_INCOMPLETE (rerun to resume)"; exit 3
fi
