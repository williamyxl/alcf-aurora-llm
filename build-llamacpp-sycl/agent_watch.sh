#!/bin/bash
# Drip-feed remaining cycles as PBS Q slots free. Safe to re-run.
set -uo pipefail
ROOT=/lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b
LLAMA=$ROOT/build-llamacpp-sycl
LOG=$LLAMA/logs/agent_watch.log
mkdir -p "$LLAMA/logs"
PENDING=(P24 P12 P16 P18 P19 P15 P23)
REBUILD_BIN=$LLAMA/build-mxfp4-reorder/bin/llama-completion

is_done() {
  local c=$1
  [ -f "$LLAMA/logs/perf_${c}.out" ] && grep -q 'PERF_DONE=1' "$LLAMA/logs/perf_${c}.out" 2>/dev/null
}

is_inflight() {
  local c=$1
  qstat -u "$USER" 2>/dev/null | grep -E "ll-${c}([^0-9]|$)|llamacpp-mxfp4" | grep -Ev ' E | C ' >/dev/null 2>&1
}

job_count() {
  # PBS qstat: state is the column with single letter Q/R/H/E (10th field in default format)
  qstat -u "$USER" 2>/dev/null | awk 'NR>5 && ($10=="Q" || $10=="R" || $10=="H") {c++} END{print c+0}'
}

echo "$(date -Is) watch start" | tee -a "$LOG"

for round in $(seq 1 180); do
  bash "$LLAMA/agent_cycle.sh" harvest P11 P12 P13 P15 P16 P17 P18 P19 P23 P24 >>"$LOG" 2>&1 || true
  n=$(job_count)
  echo "$(date -Is) round=$round jobs=$n" >>"$LOG"

  if [ "$n" -lt 3 ]; then
    submitted=0
    for c in "${PENDING[@]}"; do
      is_done "$c" && continue
      is_inflight "$c" && continue
      # alternate queues
      q=debug-scaling
      [ $((round % 2)) -eq 0 ] && q=debug
      if QUEUE=$q bash "$LLAMA/agent_cycle.sh" submit "$c" >>"$LOG" 2>&1; then
        if grep -q "SUBMITTED cycle=$c" "$LOG"; then
          submitted=1
          break
        fi
      fi
      # try other queue once
      oq=debug
      [ "$q" = debug ] && oq=debug-scaling
      if QUEUE=$oq bash "$LLAMA/agent_cycle.sh" submit "$c" >>"$LOG" 2>&1; then
        submitted=1
        break
      fi
    done
    if [ "$submitted" -eq 0 ] && [ -x "$REBUILD_BIN" ] && ! is_done P24 && ! is_inflight P24; then
      QUEUE=debug-scaling bash "$LLAMA/agent_cycle.sh" submit P24 >>"$LOG" 2>&1 || \
        QUEUE=debug bash "$LLAMA/agent_cycle.sh" submit P24 >>"$LOG" 2>&1 || true
      submitted=1
    fi
    all=1
    for c in "${PENDING[@]}"; do is_done "$c" || all=0; done
    if [ "$all" -eq 1 ] && { [ ! -x "$REBUILD_BIN" ] || is_done P24; }; then
      echo "$(date -Is) ALL_PENDING_DONE" | tee -a "$LOG"
      bash "$LLAMA/agent_cycle.sh" harvest P11 P12 P13 P15 P16 P17 P18 P19 P23 P24 | tee -a "$LOG"
      exit 0
    fi
  fi
  sleep 90
done
echo "$(date -Is) watch timeout" | tee -a "$LOG"
