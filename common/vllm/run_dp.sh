#!/usr/bin/env bash
# Data-parallel vLLM on ONE node: NREP independent engines (TPE tiles each) -> aggregate concurrency.
# GRAPH_FLAG=--graph with XPUGRAPH=1 enables the XPU graph.  Used for gpt-oss-120b MXFP4: 6 x TP2 (best, 11860 tok/s) or 3 x TP4 (6849 tok/s).
#
#   common/vllm/run_dp.sh <tag> <model_dir> [serve_conc.py args...]     # env: NREP=6 TPE=2 KVGIB=20 XPUGRAPH=0|1 (see model README)
set -o pipefail
TAG=$1; MODEL=$2; shift 2
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO/common/env.sh"
NREP=${NREP:-6}; TPE=${TPE:-2}; KVGIB=${KVGIB:-20}; STAGGER=${STAGGER:-20}
vllm_env; vllm_overlays
echo "host=$(hostname) $(date -Is) DP tag=$TAG NREP=$NREP TPE=$TPE FW=$FW XPUGRAPH=${XPUGRAPH:-0}"
pids=()
for r in $(seq 0 $((NREP-1))); do
  ( export ZE_AFFINITY_MASK=$(seq -s, $((r*TPE)) $((r*TPE+TPE-1))); vllm_cache "${TAG}_r$r"
    "$FWPY" "$REPO/common/vllm/bench_conc.py" --model "$MODEL" --tp "$TPE" --kv-cache-memory-gib "$KVGIB" ${GRAPH_FLAG:-} --tag "${TAG}_r$r" "$@" \
      > "$LOGS/${TAG}_r$r.out" 2>&1 ) &
  pids+=($!); sleep "$STAGGER"
done
for p in "${pids[@]}"; do wait "$p"; done
grep -h CONC_JSON= "$LOGS/${TAG}"_r*.out
"$FWPY" - "$LOGS" "$TAG" <<'PY'
import sys,glob,json
t=q=n=0
for f in glob.glob(f"{sys.argv[1]}/{sys.argv[2]}_r*.out"):
    for l in open(f,errors="ignore"):
        if l.startswith("CONC_JSON="):
            d=json.loads(l.split("=",1)[1]); t+=d["agg_output_tok_s"]; q+=d["req_per_s"]; n+=1
print(f"NODE_AGGREGATE replicas={n} sum_output_tok_s={t:.1f} sum_req_s={q:.2f}")
PY
