#!/usr/bin/env bash
# Resumable Hugging Face download into the model store. Run on a LOGIN node (needs the proxy). Never downloads without being asked.
#
#   common/tools/hf_download.sh <repo_id> <dest_dir> [--include 'pattern' ...]
#   e.g. common/tools/hf_download.sh poolside/Laguna-S-2.1 $DATA/store/laguna-s-2.1/bf16
#
# Judged complete by shard-count vs model.safetensors.index.json (the hf CLI exits non-zero on success in this env, so the exit
# code is ignored). Retries on network drops (observed on multi-TB pulls). After it finishes, link the dir under $MODELS/<model>/<variant>.
set -o pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; source "$REPO/common/env.sh"; export_proxy
ID=$1; DEST=$2; shift 2
vllm_env >/dev/null 2>&1; HF="$FWROOT/bin/hf"; [ -x "$HF" ] || HF=hf
export HF_HOME="${HF_HOME:-$DATA/.hf_cache}" HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=0
unset HF_HUB_OFFLINE
complete() {
  [ -n "$(ls "$DEST"/*.gguf 2>/dev/null)" ] && [ ! -f "$DEST/model.safetensors.index.json" ] && [ -z "$(find "$DEST" -name '*.incomplete' 2>/dev/null)" ] && return 0
  local idx="$DEST/model.safetensors.index.json"; [ -f "$idx" ] || return 1
  [ -z "$(find "$DEST" -name '*.incomplete' 2>/dev/null)" ] || return 1
  local want have; want=$(python3 -c "import json;print(len(set(json.load(open('$idx'))['weight_map'].values())))")
  have=$(ls "$DEST"/*.safetensors 2>/dev/null | wc -l); [ "$have" -ge "$want" ]
}
mkdir -p "$DEST"
for i in $(seq 1 ${RETRIES:-200}); do
  echo "=== $(date -Is) $ID attempt $i"; "$HF" download "$ID" --local-dir "$DEST" --max-workers "${WORKERS:-6}" "$@" || true
  complete && { echo "=== $(date -Is) COMPLETE $DEST"; exit 0; }
  sleep 30
done
echo "gave up after retries"; exit 1
