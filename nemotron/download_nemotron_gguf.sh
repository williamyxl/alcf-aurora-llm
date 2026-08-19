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
export WORKDIR=/lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/nemotron
export REPO=${REPO:-unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF}
export QUANT=${QUANT:-MXFP4_MOE}
export OUT_DIR=${OUT_DIR:-$WORKDIR/models/gguf}
export HF_HOME=${HF_HOME:-$WORKDIR/.cache/huggingface}
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
if [ -f /lus/flare/projects/MatSciAI/xiaoliyan/software/conda/envs/aurora-llm/bin/hf ]; then
  source /lus/flare/projects/MatSciAI/xiaoliyan/miniforge3/etc/profile.d/conda.sh
  conda activate /lus/flare/projects/MatSciAI/xiaoliyan/software/conda/envs/aurora-llm
fi

echo "host=$(hostname) date=$(date -Is) REPO=$REPO QUANT=$QUANT OUT_DIR=$OUT_DIR"
df -h "$WORKDIR" | tail -1

echo "Starting hf download (resumable), pattern ${QUANT}/* ..."
set +e
hf download "$REPO" --include "${QUANT}/*" --local-dir "$OUT_DIR"
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
  ln -sf "$FIRST" "$OUT_DIR/nemotron-ultra-${QUANT}.gguf"
  echo "DOWNLOAD_OK  -> $OUT_DIR/nemotron-ultra-${QUANT}.gguf"
  exit 0
else
  echo "DOWNLOAD_INCOMPLETE (rerun to resume)"; exit 3
fi
