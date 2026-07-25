#!/bin/bash
# Download Thinking Machines Inkling HF checkpoint (trainable). Resumable.
# Default: thinkingmachines/Inkling (BF16, ~2TB, 108 shards).
# Alt:    REPO=thinkingmachines/Inkling-NVFP4 bash download_inkling_hf.sh
#
# Usage:
#   bash download_inkling_hf.sh
#   qsub download_inkling_hf.pbs
#
# Gate: config.json + tokenizer files + safetensors present; TOTAL_BYTES threshold.

set -euo pipefail

export WORKDIR=/lus/flare/projects/MOFA/xiaoliyan/workdir/llm/inkling
export REPO=${REPO:-thinkingmachines/Inkling}
export OUT_DIR=${OUT_DIR:-$WORKDIR/models/inkling-hf}
export HF_HOME=${HF_HOME:-$WORKDIR/.cache/huggingface}
export http_proxy=${http_proxy:-http://proxy.alcf.anl.gov:3128}
export https_proxy=${https_proxy:-http://proxy.alcf.anl.gov:3128}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}

# BF16 full Inkling is ~2TB; NVFP4 is smaller. Override MIN_OK_BYTES if needed.
MIN_OK_BYTES=${MIN_OK_BYTES:-1800000000000}  # ~1.8TB for BF16

mkdir -p "$OUT_DIR" "$HF_HOME"

if [ -f /lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b/build-vllm-xpu/env/bin/hf ]; then
  # shellcheck source=/dev/null
  source /lus/flare/projects/MOFA/xiaoliyan/software/miniforge3/etc/profile.d/conda.sh
  conda activate /lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b/build-vllm-xpu/env
fi

echo "host=$(hostname) date=$(date -Is)"
echo "REPO=$REPO OUT_DIR=$OUT_DIR MIN_OK_BYTES=$MIN_OK_BYTES"
df -h "$WORKDIR" | tail -1

# Snapshot size before
BYTES_BEFORE=$(du -sb "$OUT_DIR" 2>/dev/null | awk '{print $1}')
echo "BYTES_BEFORE=${BYTES_BEFORE:-0}"

echo "Starting hf download (resumable)…"
# Full repo download; hf skips existing complete files
set +e
hf download "$REPO" --local-dir "$OUT_DIR"
RC=$?
set -e
echo "hf_exit=$RC"

echo "=== top-level ==="
ls -la "$OUT_DIR" | head -30

BYTES=$(du -sb "$OUT_DIR" 2>/dev/null | awk '{print $1}')
echo "TOTAL_BYTES=$BYTES TOTAL_GB=$(awk -v b="$BYTES" 'BEGIN{printf "%.1f", b/1e9}') TOTAL_TB=$(awk -v b="$BYTES" 'BEGIN{printf "%.2f", b/1e12}')"

HAS_CONFIG=0
HAS_TOK=0
N_ST=0
[ -f "$OUT_DIR/config.json" ] && HAS_CONFIG=1
[ -f "$OUT_DIR/tokenizer.json" ] || [ -f "$OUT_DIR/tokenizer.model" ] || [ -f "$OUT_DIR/tokenizer_config.json" ] && HAS_TOK=1
N_ST=$(find "$OUT_DIR" -maxdepth 1 -name 'model-*.safetensors' 2>/dev/null | wc -l)
echo "HAS_CONFIG=$HAS_CONFIG HAS_TOK=$HAS_TOK N_SAFETENSORS=$N_ST"

if [ "$HAS_CONFIG" = 1 ] && [ "$BYTES" -ge "$MIN_OK_BYTES" ]; then
  echo "DOWNLOAD_OK=1"
  # Quick tokenizer smoke (CPU)
  python3 - <<PY
from transformers import AutoTokenizer
p = "$OUT_DIR"
tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
print("TOKENIZER_OK", type(tok).__name__, "vocab", getattr(tok, "vocab_size", "?"))
PY
  exit 0
fi

echo "DOWNLOAD_PARTIAL=1 (resume with another run / qsub)"
echo "Need TOTAL_BYTES>=$MIN_OK_BYTES and config.json"
exit 2
