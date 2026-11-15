#!/usr/bin/env bash
# llama.cpp SYCL one-shot decode (llama-completion) or batched-bench. Used for single-stream and aggregate numbers.
#   GGUF=... TILES=0 MODE=moecpu|gpu BENCH=completion|batched common/llamacpp/oneshot.sh
# MODE=moecpu : 1 tile, experts on CPU, HBM NUMA bind (gpt-oss "F4_hbm": 42.4 tok/s). MODE=gpu : TILES tiles, layer split.
set -o pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; source "$REPO/common/env.sh"; source "$REPO/common/lib.sh"
: "${GGUF:?set GGUF}"; MODE=${MODE:-gpu}; BENCH=${BENCH:-completion}; CTX=${CTX:-4096}; NTILES=${NTILES:-12}; NP=${NP:-16}; PP=${PP:-64}; TG=${TG:-128}
llamacpp_env; require_file "$GGUF"
PROMPT=${PROMPT:-"What is a metal-organic framework (MOF)? Answer in one short paragraph."}
if [ "$MODE" = moecpu ]; then
  export ZE_AFFINITY_MASK=0; PRE="numactl --physcpubind=${CPUS:-1-51,105-155} --membind=${HBM_NODE:-2}"; ARGS="-sm none -ncmoe 99 -t ${NTHREADS:-32} --numa numactl"
else
  export ZE_AFFINITY_MASK=$(seq -s, 0 $((NTILES-1))); PRE=""
  TS=$(python3 -c "print(','.join(['%.10f'%(1/$NTILES)]*$NTILES))"); ARGS="-sm layer -ts $TS"
fi
if [ "$BENCH" = batched ]; then
  $PRE "$LLAMA_BIN/llama-batched-bench" -m "$GGUF" -c "$CTX" -ngl 99 -fa on $ARGS --no-mmap -npp "$PP" -ntg "$TG" -npl "$NP"
else
  $PRE "$LLAMA_BIN/llama-completion" -m "$GGUF" -p "$PROMPT" -n "$TG" -c "$CTX" -ngl 99 $ARGS -fa on --seed 42 -no-cnv --single-turn --jinja \
    --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.01 --no-mmap --no-display-prompt
fi
echo "EXIT=$?"
