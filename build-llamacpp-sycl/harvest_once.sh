#!/bin/bash
# One-shot harvest / optional submit — safe on Aurora login nodes (no sleep loop).
# Usage: bash harvest_once.sh
#        bash harvest_once.sh --submit   # submit next missing PRIORITY cycle if slots free

set -uo pipefail
ROOT=/lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b
LLAMA=$ROOT/build-llamacpp-sycl
PBS=$ROOT/bench_llamacpp_sycl_perf.pbs
PBS_G=$ROOT/bench_llamacpp_sycl_phaseG.pbs
LOGS=$LLAMA/logs
PRIORITY=(G0 G1 G0s G1s G0t G1t)
SUBMIT=0
[[ "${1:-}" == "--submit" ]] && SUBMIT=1

job_count() {
  qstat -u "$USER" 2>/dev/null | awk 'NR>5 && ($10=="Q"||$10=="R"||$10=="H"){c++} END{print c+0}'
}

is_done() {
  local c=$1
  [ -f "$LOGS/perf_${c}.out" ] && grep -q 'PERF_DONE=1' "$LOGS/perf_${c}.out" 2>/dev/null
}

is_inflight() {
  local c=$1
  qstat -u "$USER" 2>/dev/null | grep -E "ll-${c}([^0-9]|$)" | grep -Ev ' E | C ' >/dev/null 2>&1
}

echo "=== qstat ==="
qstat -u "$USER" 2>/dev/null || echo '(empty)'
echo "jobs=$(job_count)"
echo
echo "=== cycles ==="
for c in "${PRIORITY[@]}"; do
  if is_done "$c"; then
    sum=$(bash "$LLAMA/extract_metrics.sh" "$LOGS/perf_${c}_cli.log" "$LOGS/perf_${c}_bench.log" 2>/dev/null | grep METRICS_SUMMARY | tail -1)
    echo "$c DONE $sum"
  elif is_inflight "$c"; then
    echo "$c INFLIGHT"
  elif [ -f "$LOGS/perf_${c}.out" ]; then
    echo "$c PARTIAL"
  else
    echo "$c —"
  fi
done

if [[ "$SUBMIT" != 1 ]]; then
  echo
  echo "(pass --submit to queue next missing cycle on debug/debug-scaling)"
  exit 0
fi

n=$(job_count)
if [[ "$n" -gt 0 ]]; then
  echo "defer submit: jobs=$n already queued/running"
  exit 0
fi

for c in "${PRIORITY[@]}"; do
  is_done "$c" && continue
  is_inflight "$c" && continue
  envf="$LLAMA/cycles/${c}.env"
  [ -f "$envf" ] || continue
  # G* → phaseG PBS; else perf PBS
  if [[ "$c" == G* ]]; then
    script=$PBS_G
    q=debug
    [[ "$c" == G0* ]] && q=debug-scaling
  else
    script=$PBS
    q=debug
  fi
  jid=$(qsub -q "$q" -v "CYCLE=$c" -N "ll-$c" -o "$LOGS/perf_${c}.out" "$script")
  echo "submitted $c → $jid on $q"
  exit 0
done
echo "nothing to submit (all PRIORITY done or missing env)"
