#!/bin/bash
# Prefer Hugging Face MXFP4 GGUF over local convert.
# Primary: ggml-org/gpt-oss-120b-GGUF → gpt-oss-120b-MXFP4.gguf
#
# Usage (login or compute, with proxy):
#   bash download_gptoss_mxfp4_gguf.sh
# Or: qsub download_gptoss_mxfp4_gguf.pbs

set -euo pipefail

export WORKDIR=/lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b
export OUT_DIR=$WORKDIR/models/ggml-org-gpt-oss-120b-GGUF
export OUT_LINK=$WORKDIR/models/openai-gpt-oss-120b-mxfp4.gguf
export http_proxy=${http_proxy:-http://proxy.alcf.anl.gov:3128}
export https_proxy=${https_proxy:-http://proxy.alcf.anl.gov:3128}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}

REPO=${REPO:-ggml-org/gpt-oss-120b-GGUF}
# Current ggml-org main file name (single-file MXFP4)
INCLUDE=${INCLUDE:-gpt-oss-120b-MXFP4.gguf}

mkdir -p "$OUT_DIR"

if [ -f "$WORKDIR/build-vllm-xpu/env/bin/huggingface-cli" ] || [ -f "$WORKDIR/build-vllm-xpu/env/bin/hf" ]; then
  # shellcheck source=/dev/null
  source /lus/flare/projects/MOFA/xiaoliyan/software/miniforge3/etc/profile.d/conda.sh
  conda activate "$WORKDIR/build-vllm-xpu/env"
fi

echo "Downloading $REPO ($INCLUDE) → $OUT_DIR"
if command -v hf >/dev/null 2>&1; then
  hf download "$REPO" --include "$INCLUDE" --local-dir "$OUT_DIR"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$REPO" --include "$INCLUDE" --local-dir "$OUT_DIR"
else
  python -m huggingface_hub.commands.huggingface_cli download "$REPO" \
    --include "$INCLUDE" --local-dir "$OUT_DIR"
fi

ls -lh "$OUT_DIR"/*.gguf
# Point smoke/perf default path at HF file (keep local convert as backup if present)
if [ -f "$OUT_DIR/$INCLUDE" ]; then
  ln -sfn "$OUT_DIR/$INCLUDE" "$OUT_LINK.hf"
  echo "HF GGUF ready: $OUT_DIR/$INCLUDE"
  echo "Symlink (non-destructive): $OUT_LINK.hf"
  echo "To switch smoke/perf: export GGUF=$OUT_LINK.hf  (or replace $OUT_LINK after backup)"
fi
echo "DOWNLOAD_OK=1"
