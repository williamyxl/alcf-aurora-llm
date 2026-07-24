#!/bin/bash
# Agentic smoke/perf loop helpers for llama.cpp SYCL gpt-oss-120b.
# Usage:
#   bash agent_cycle.sh status
#   bash agent_cycle.sh submit P12 P13 P15   # debug-scaling by default
#   bash agent_cycle.sh submit-debug P15     # debug queue
#   bash agent_cycle.sh harvest
#   bash agent_cycle.sh next                 # submit next pending from QUEUE

set -euo pipefail
ROOT=/lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b
LLAMA=$ROOT/build-llamacpp-sycl
PBS=$ROOT/bench_llamacpp_sycl_perf.pbs
LOGS=$LLAMA/logs
LEDGER=$LLAMA/CYCLE_LOG.md
QUEUE_FILE=$LLAMA/AGENT_QUEUE.txt
mkdir -p "$LOGS"

# Default experiment order (new solutions first after in-flight P11)
DEFAULT_QUEUE=(P12 P13 P15 P16 P17 P18 P19 P23 P14_tp2)

cmd=${1:-status}
shift || true

submit_one() {
  local cycle=$1
  local q=${2:-debug-scaling}
  local envf=$LLAMA/cycles/${cycle}.env
  if [ ! -f "$envf" ]; then
    echo "MISSING_ENV $envf"
    return 1
  fi
  local out=$LOGS/perf_${cycle}.out
  local jid errf
  errf=$(mktemp)
  if ! jid=$(qsub -q "$q" -v CYCLE="$cycle" -N "ll-${cycle}" -o "$out" "$PBS" 2>"$errf"); then
    echo "SUBMIT_FAIL cycle=$cycle queue=$q err=$(tr '\n' ' ' <"$errf")"
    rm -f "$errf"
    return 1
  fi
  rm -f "$errf"
  if [ -z "$jid" ]; then
    echo "SUBMIT_FAIL cycle=$cycle empty_jid"
    return 1
  fi
  echo "SUBMITTED cycle=$cycle queue=$q job=$jid out=$out"
  echo "$(date -Is) SUBMIT $cycle $jid $q" >> "$LOGS/agent_actions.log"
}

harvest_one() {
  local cycle=$1
  local out=$LOGS/perf_${cycle}.out
  local cli=$LOGS/perf_${cycle}_cli.log
  local bench=$LOGS/perf_${cycle}_bench.log
  if [ ! -f "$out" ]; then
    echo "NO_OUT $cycle"
    return 0
  fi
  if ! grep -q 'PERF_DONE=1\|BUILD_OK=1\|COMPLETION_FAIL\|error\|REFUSING' "$out" 2>/dev/null; then
    # may still be running
    if grep -q 'PERF_DONE=1' "$out" 2>/dev/null; then
      :
    else
      echo "RUNNING_OR_EMPTY $cycle"
      return 0
    fi
  fi
  echo "=== $cycle ==="
  bash "$LLAMA/extract_metrics.sh" "$cli" "$bench" 2>/dev/null || true
  grep -E 'PERF_DONE|COMPLETION_EXIT|TILE_PIN|METRICS_SUMMARY|error|FAIL|REFUSING' "$out" 2>/dev/null | tail -8 || true
}

case "$cmd" in
  status)
    echo "date=$(date -Is)"
    qstat -u "$USER" 2>/dev/null | head -20 || true
    echo "--- best ---"
    grep -E 'Best:|P7c|P7b|P10|P11|P12' "$LEDGER" | tail -8 || true
    ;;
  submit)
    q=${QUEUE:-debug-scaling}
    if [ "$#" -eq 0 ]; then
      set -- "${DEFAULT_QUEUE[@]}"
    fi
    rc=0
    for c in "$@"; do
      submit_one "$c" "$q" || rc=1
    done
    exit "$rc"
    ;;
  submit-debug)
    if [ "$#" -eq 0 ]; then
      echo "usage: submit-debug CYCLE [CYCLE...]"
      exit 2
    fi
    rc=0
    for c in "$@"; do
      submit_one "$c" debug || rc=1
    done
    exit "$rc"
    ;;
  harvest)
    if [ "$#" -eq 0 ]; then
      set -- P11 P12 P13 P15 P16 P17 P18 P19 P23 P24 P14_tp2
    fi
    for c in "$@"; do
      harvest_one "$c"
    done
    ;;
  next)
    if [ ! -f "$QUEUE_FILE" ]; then
      printf '%s\n' "${DEFAULT_QUEUE[@]}" > "$QUEUE_FILE"
    fi
    while read -r c; do
      [ -z "$c" ] && continue
      if [ -f "$LOGS/perf_${c}.out" ] && grep -q 'PERF_DONE=1' "$LOGS/perf_${c}.out" 2>/dev/null; then
        continue
      fi
      # skip if already queued/running with matching name
      if qstat -u "$USER" 2>/dev/null | grep -q "ll-${c}\|llamacpp"; then
        echo "WAIT queue busy; not submitting $c"
        exit 0
      fi
      submit_one "$c" "${QUEUE:-debug-scaling}"
      exit 0
    done < "$QUEUE_FILE"
    echo "QUEUE_EMPTY"
    ;;
  *)
    echo "usage: $0 status|submit|submit-debug|harvest|next [cycles...]"
    exit 2
    ;;
esac
