#!/bin/bash
# Download Unsloth Inkling UD-IQ1_S GGUF shards (~270GB). Resumable.
# Usage: bash download_inkling_ud_iq1_s.sh
#    or: qsub download_inkling_ud_iq1_s.pbs
#
# NOTE: do NOT name the pattern var INCLUDE — modules set $INCLUDE to C include paths.

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; source "$REPO_ROOT/common/env.sh"; export_proxy

export WORKDIR="$DATA/store/inkling"
export OUT_DIR=$WORKDIR/unsloth-Inkling-GGUF
export http_proxy=${http_proxy:-http://proxy.alcf.anl.gov:3128}
export https_proxy=${https_proxy:-http://proxy.alcf.anl.gov:3128}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}

HF_REPO=${HF_REPO:-unsloth/Inkling-GGUF}
# Quote default so bash does not pathname-expand *
GGUF_INCLUDE=${GGUF_INCLUDE:-'UD-IQ1_S/*'}

mkdir -p "$OUT_DIR"

vllm_env >/dev/null 2>&1; export PATH="$FWROOT/bin:$PATH"   # hf CLI + huggingface_hub from the frameworks module

echo "host=$(hostname) date=$(date -Is)"
echo "Downloading $HF_REPO pattern=$GGUF_INCLUDE → $OUT_DIR"
echo "NOTE: env INCLUDE (compiler) is intentionally ignored: ${INCLUDE:-(unset)}"
df -h "$WORKDIR" | tail -1

# Prefer explicit shard list (avoids glob / env pitfalls)
mapfile -t SHARDS < <(python3 - <<'PY'
from huggingface_hub import list_repo_files
for f in list_repo_files("unsloth/Inkling-GGUF"):
    if f.startswith("UD-IQ1_S/") and f.endswith(".gguf"):
        print(f)
PY
)

if [ "${#SHARDS[@]}" -eq 0 ]; then
  echo "FALLBACK: using --include $GGUF_INCLUDE"
  hf download "$HF_REPO" --include "$GGUF_INCLUDE" --local-dir "$OUT_DIR"
else
  echo "Downloading ${#SHARDS[@]} explicit shards"
  for f in "${SHARDS[@]}"; do
    echo "→ $f"
    hf download "$HF_REPO" "$f" --local-dir "$OUT_DIR"
  done
fi

echo "=== shards ==="
find "$OUT_DIR" -name '*UD-IQ1_S*.gguf' -printf '%p %s\n' | sort
BYTES=$(find "$OUT_DIR" -name '*UD-IQ1_S*.gguf' -printf '%s\n' | awk '{s+=$1} END{print s+0}')
echo "TOTAL_BYTES=$BYTES TOTAL_GB=$(awk -v b="$BYTES" 'BEGIN{printf "%.1f", b/1e9}')"

FIRST=$(find "$OUT_DIR" -name 'inkling-UD-IQ1_S-00001*.gguf' | head -1)
if [ -z "$FIRST" ]; then
  FIRST=$(find "$OUT_DIR" -path '*/UD-IQ1_S/*.gguf' | sort | head -1)
fi
if [ -n "$FIRST" ]; then
  # keep split filename; harness points at 00001-of-00007
echo "GGUF_FIRST=$FIRST"
  mkdir -p "$MODELS/inkling"; ln -sfn "$OUT_DIR/UD-IQ1_S" "$MODELS/inkling/UD-IQ1_S-gguf"; echo "LINK=$MODELS/inkling/UD-IQ1_S-gguf"
fi

if [ "$BYTES" -ge 200000000000 ]; then
  echo "DOWNLOAD_OK=1"
else
  echo "DOWNLOAD_PARTIAL=1 (resume with another qsub)"
  exit 2
fi
