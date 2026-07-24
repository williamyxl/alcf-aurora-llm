#!/usr/bin/env bash
# Download community BF16 / FP16 gpt-oss-120b checkpoints (no MXFP4).
# Official openai/gpt-oss-120b is MXFP4-only; these are dequantized HF mirrors.
#
# Usage:
#   bash download_gptoss_bf16_fp16.sh          # both
#   bash download_gptoss_bf16_fp16.sh bf16     # BF16 only
#   bash download_gptoss_bf16_fp16.sh fp16     # FP16 only
#
# Repos:
#   BF16: unsloth/gpt-oss-120b-BF16  (~popular; no quantization_config)
#   FP16: twhitworth/gpt-oss-120b-fp16

set -euo pipefail
cd /lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b
mkdir -p models .cache/huggingface

source /lus/flare/projects/MOFA/xiaoliyan/software/miniforge3/etc/profile.d/conda.sh
conda activate /lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b/build-vllm-xpu/env

export HF_HOME=/lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b/.cache/huggingface
export TMPDIR=/tmp
export HF_XET_HIGH_PERFORMANCE=1

MODE="${1:-both}"

download_one() {
  local repo="$1"
  local dest="$2"
  echo "==== $(date -u) downloading $repo -> $dest ===="
  mkdir -p "$dest"
  hf download "$repo" --local-dir "$dest"
  # Sanity: must not be MXFP4
  python - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("$dest/config.json").read_text())
qc = cfg.get("quantization_config")
print("torch_dtype=", cfg.get("torch_dtype"), "quantization_config=", qc)
if qc is not None:
    raise SystemExit(f"FAIL: expected no quantization_config, got {qc}")
print("OK: unquantized config")
PY
  echo "==== $(date -u) done $repo ===="
  du -sh "$dest"
}

case "$MODE" in
  bf16)
    download_one "unsloth/gpt-oss-120b-BF16" "models/openai-gpt-oss-120b-bf16"
    ;;
  fp16)
    download_one "twhitworth/gpt-oss-120b-fp16" "models/openai-gpt-oss-120b-fp16"
    ;;
  both)
    download_one "unsloth/gpt-oss-120b-BF16" "models/openai-gpt-oss-120b-bf16"
    download_one "twhitworth/gpt-oss-120b-fp16" "models/openai-gpt-oss-120b-fp16"
    ;;
  *)
    echo "usage: $0 [both|bf16|fp16]" >&2
    exit 2
    ;;
esac

echo "All requested downloads complete."
