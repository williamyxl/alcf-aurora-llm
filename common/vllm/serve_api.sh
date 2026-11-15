#!/usr/bin/env bash
# OpenAI-compatible endpoint on ONE Aurora node: NREP vLLM engines (TPE tiles each) behind common/serve/lb.py.
# Writes $LOGS/ENDPOINT.txt (lb_url). Source model-specific defaults from <model>/vllm/serve.env first.
#
#   NREP=6 TPE=2 MODEL=... SERVED_NAME=... common/vllm/serve_api.sh        (run from a PBS job)
set -o pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO/common/env.sh"
: "${MODEL:?set MODEL}"; : "${SERVED_NAME:?set SERVED_NAME}"
NREP=${NREP:-3}; TPE=${TPE:-4}; MML=${MML:-8192}; MNS=${MNS:-128}; MEMUTIL=${MEMUTIL:-0.85}; KVGIB=${KVGIB:-20}
LB_PORT=${LB_PORT:-8000}; BASE_PORT=${BASE_PORT:-8001}; EXTRA_ARGS=${EXTRA_ARGS:-}
EAGER_FLAG=${EAGER_FLAG---enforce-eager}               # set EAGER_FLAG= (empty) together with XPUGRAPH=1 for XPU graph
[ -f "$REPO/STOP" ] && { echo "STOP file present"; exit 0; }
vllm_env; vllm_overlays
export TIKTOKEN_RS_CACHE_DIR=${TIKTOKEN_RS_CACHE_DIR:-$DATA/tiktoken/.tiktoken_cache}
require_file "$MODEL/config.json" "model"
echo "$(date -Is) host=$(hostname) job=${PBS_JOBID:-none} NREP=$NREP TPE=$TPE MML=$MML MNS=$MNS KVGIB=$KVGIB FW=$FW graph=${XPUGRAPH:-0}"
pids=(); backends=()
for r in $(seq 0 $((NREP-1))); do
  port=$((BASE_PORT + r)); backends+=("http://127.0.0.1:$port")
  ( export ZE_AFFINITY_MASK=$(seq -s, $((r*TPE)) $((r*TPE+TPE-1))); vllm_cache "api_r$r"
    "$VLLM_BIN" serve "$MODEL" --tensor-parallel-size "$TPE" --max-model-len "$MML" --max-num-seqs "$MNS" \
      --gpu-memory-utilization "$MEMUTIL" --kv-cache-memory-bytes $((KVGIB*1073741824)) $EAGER_FLAG \
      --served-model-name "$SERVED_NAME" --host 127.0.0.1 --port "$port" $EXTRA_ARGS > "$LOGS/api_engine_r$r.out" 2>&1 ) &
  pids+=($!); sleep 15
done
for b in "${backends[@]}"; do wait_health "$b/health" 360 || echo "WARN: $b not ready"; done
"$FWPY" "$REPO/common/serve/lb.py" --host 0.0.0.0 --port "$LB_PORT" --backends "${backends[@]}" > "$LOGS/api_lb.out" 2>&1 &
LB=$!; sleep 3; write_endpoint "$LOGS/ENDPOINT.txt" "$LB_PORT" "$SERVED_NAME"; cat "$LOGS/ENDPOINT.txt"
trap 'kill $LB "${pids[@]}" 2>/dev/null' EXIT
wait "${pids[@]}"
