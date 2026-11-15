#!/usr/bin/env bash
# llama.cpp SYCL: launch NREP llama-server instances (TPE tiles each), then drive them with common/serve/bench_client.py
# (single-stream TTFT/prefill/decode + concurrency sweep). One recipe covers MoE->CPU single tile and full-XPU layer split.
#
#   GGUF=<first shard or file> NREP=1 TPE=1 MODE=offload common/llamacpp/serve_bench.sh
# Env: LLAMA=inkling|gptoss  GGUF  NREP  TPE  MODE=offload|fullxpu  CTX  NPAR  NTHREADS  LEVELS  MAXTOK  SERVED  SPLIT=layer|none
set -o pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO/common/env.sh"; source "$REPO/common/lib.sh"
: "${GGUF:?set GGUF}"; : "${SERVED:?set SERVED}"
NREP=${NREP:-1}; TPE=${TPE:-1}; MODE=${MODE:-fullxpu}; CTX=${CTX:-8192}; NPAR=${NPAR:-16}; NTHREADS=${NTHREADS:-16}
MAXTOK=${MAXTOK:-128}; LEVELS=${LEVELS:-1 4 8}; BASE_PORT=${BASE_PORT:-8001}; SPLIT=${SPLIT:-layer}; TAG=${TAG:-$SERVED}
llamacpp_env; require_file "$GGUF" "GGUF"; require_file "$LLAMA_BIN/llama-server" "llama.cpp build ($LLAMA)"
MOE=""; [ "$MODE" = "offload" ] && MOE="-ncmoe 99"
[ "$TPE" = 1 ] && SPLIT=none
echo "START $(date -Is) host=$(hostname) $TAG NREP=$NREP TPE=$TPE MODE=$MODE CTX=$CTX NPAR=$NPAR GGUF=$GGUF"
pids=(); ports=()
for r in $(seq 0 $((NREP-1))); do
  port=$((BASE_PORT+r)); ports+=($port)
  ( export ZE_AFFINITY_MASK=$(seq -s, $((r*TPE)) $((r*TPE+TPE-1)))
    numactl --interleave=all "$LLAMA_BIN/llama-server" -m "$GGUF" -c "$CTX" -ngl 99 --split-mode "$SPLIT" \
      -fa on -t "$NTHREADS" --no-mmap $MOE --parallel "$NPAR" --cont-batching --jinja --alias "$SERVED" \
      --host 127.0.0.1 --port "$port" > "$LOGS/${TAG}_server_r$r.out" 2>&1 ) &
  pids+=($!)
done
for p in "${ports[@]}"; do wait_health "http://127.0.0.1:$p/health" 200 || { echo "SERVER NOT READY (port $p)"; tail -40 "$LOGS/${TAG}_server_r0.out"; kill "${pids[@]}" 2>/dev/null; exit 4; }; done
echo "servers ready $(date -Is)"
if [ "$NREP" = 1 ]; then
  "$FWPY" "$REPO/common/serve/bench_client.py" --url "http://127.0.0.1:${ports[0]}/v1" --model "$SERVED" --max-tokens "$MAXTOK" --warmup --concurrency $LEVELS
else
  "$FWPY" "$REPO/common/serve/lb.py" --host 127.0.0.1 --port 8000 --backends $(for p in "${ports[@]}"; do printf 'http://127.0.0.1:%s ' $p; done) > "$LOGS/${TAG}_lb.out" 2>&1 &
  LB=$!; sleep 3
  "$FWPY" "$REPO/common/serve/bench_client.py" --url "http://127.0.0.1:8000/v1" --model "$SERVED" --max-tokens "$MAXTOK" --warmup --concurrency $LEVELS
  kill $LB 2>/dev/null
fi
kill "${pids[@]}" 2>/dev/null; echo "DONE $(date -Is)"
