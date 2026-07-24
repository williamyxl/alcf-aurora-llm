#!/bin/bash
# One-shot harvest / optional submit — safe on Aurora login nodes (no sleep loop).
# Usage: bash harvest_once.sh
#        bash harvest_once.sh --submit   # fill free slots on BOTH debug and debug-scaling
#
# Queue policy: use debug AND debug-scaling in parallel (each queue ≤1 job/user).
#   Prefer: pure-GPU PG*/C_PG* → debug-scaling ; MoE MO*/C_MO*/MO_HBM → debug
#   If preferred busy, fall back to the other queue when select fits (debug max 2 nodes).

set -uo pipefail
ROOT=/lus/flare/projects/MOFA/xiaoliyan/workdir/llm/inkling
LLAMA=$ROOT/build-llamacpp-sycl
PBS=$ROOT/bench_llamacpp_sycl_perf.pbs
PBS_RPC=$ROOT/bench_llamacpp_sycl_rpc_multinode.pbs
LOGS=$LLAMA/logs
PRIORITY=(PG1 PG2 PG4 PG8 PG10 PG12 PG14 PG16 PG18 PG20 PG22 PG24 MO1 MO2 C_PG8 C_PG10 C_PG12 C_PG14 C_PG16 C_PG18 C_PG20 C_PG22 C_PG24 C_MO1 C_MO2 MO_HBM C_MO_HBM)
# TP>12 (PBS_SELECT>1) → debug-scaling + RPC multinode PBS (separate SYCL+RPC build)

SUBMIT=0
[[ "${1:-}" == "--submit" ]] && SUBMIT=1

queue_busy() {
  local q=$1
  qstat -u "$USER" 2>/dev/null | awk -v q="$q" '
    NR>5 && ($10=="Q"||$10=="R"||$10=="H") {
      qq=$3
      # queue column may be truncated (debug-s*)
      if (index(qq, "debug-s")==1 || qq ~ /^debug-scaling/) qq="debug-scaling"
      else if (index(qq, "debug")==1) qq="debug"
      if (qq==q) c++
    }
    END { print c+0 }'
}

job_count() {
  qstat -u "$USER" 2>/dev/null | awk 'NR>5 && ($10=="Q"||$10=="R"||$10=="H"){c++} END{print c+0}'
}

is_done() {
  local c=$1
  local f="$LOGS/perf_${c}.out"
  [ -f "$f" ] || return 1
  grep -q 'PERF_DONE=1' "$f" 2>/dev/null || return 1
  # Treat crashed RPC/completion (nonzero exit + no gen tps) as not done → allow retry after harness fix.
  if grep -qE 'COMPLETION_EXIT=0' "$f" 2>/dev/null; then
    return 0
  fi
  if grep -qE 'METRICS_GEN_TPS=[0-9]' "$f" 2>/dev/null; then
    return 0
  fi
  # Explicit terminal fails we do not auto-retry (OOM / SKIPPED / escalated).
  if grep -qE 'FAIL_OOM|SKIPPED|HARVEST_DONE|COMPLETION_EXIT=137|Killed' "$f" 2>/dev/null; then
    return 0
  fi
  if grep -qE 'COMPLETION_EXIT=[1-9]|COMPLETION_EXIT=[1-9][0-9]+' "$f" 2>/dev/null; then
    return 1
  fi
  return 0
}

is_inflight() {
  local c=$1
  qstat -u "$USER" 2>/dev/null | grep -E "ll-${c}([^0-9]|$)" | grep -Ev ' E | C ' >/dev/null 2>&1
}

cycle_select() {
  local envf=$1
  local sel
  sel=$(grep -E '^export PBS_SELECT=' "$envf" 2>/dev/null | head -1 | cut -d= -f2 | tr -d \"\' || true)
  echo "${sel:-1}"
}

# Preferred queue for a cycle
preferred_queue() {
  local c=$1 envf sel
  envf="$LLAMA/cycles/${c}.env"
  sel=1
  [ -f "$envf" ] && sel=$(cycle_select "$envf")
  # Multi-node / TP>12 always on debug-scaling (RPC backend tests)
  if [[ "$sel" -gt 1 ]]; then
    echo debug-scaling
    return
  fi
  case "$c" in
    MO*|C_MO*|MO_HBM|C_MO_HBM) echo debug ;;
    *) echo debug-scaling ;;
  esac
}

# Can this cycle run on queue q given select=sel?
queue_accepts() {
  local q=$1 sel=$2
  case "$q" in
    debug)
      # debug: max 2 nodes
      [[ "$sel" -le 2 ]]
      ;;
    debug-scaling)
      [[ "$sel" -le 256 ]]
      ;;
    *) return 1 ;;
  esac
}

pick_queue() {
  # stdout: queue name if a free slot exists that can take this cycle; else empty
  local c=$1 sel=$2
  local pref alt
  pref=$(preferred_queue "$c")
  if [[ "$pref" == debug ]]; then alt=debug-scaling; else alt=debug; fi

  if [[ "$(queue_busy "$pref")" -eq 0 ]] && queue_accepts "$pref" "$sel"; then
    echo "$pref"
    return 0
  fi
  if [[ "$(queue_busy "$alt")" -eq 0 ]] && queue_accepts "$alt" "$sel"; then
    echo "$alt"
    return 0
  fi
  return 1
}

echo "=== qstat ==="
qstat -u "$USER" 2>/dev/null || echo '(empty)'
echo "jobs=$(job_count) debug_busy=$(queue_busy debug) debug-scaling_busy=$(queue_busy debug-scaling)"
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
  echo "(pass --submit to fill free slots on debug AND debug-scaling)"
  exit 0
fi

submitted=0
# Fill each free queue with the next cycle that prefers it.
# Fallback to the other queue only for select=1 jobs (avoid parking 2-node TP>12 on debug as overflow).
for q in debug-scaling debug; do
  [[ "$(queue_busy "$q")" -eq 0 ]] || continue
  picked=
  # Pass 1: preferred queue match
  for c in "${PRIORITY[@]}"; do
    is_done "$c" && continue
    is_inflight "$c" && continue
    envf="$LLAMA/cycles/${c}.env"
    [ -f "$envf" ] || continue
    sel=$(cycle_select "$envf")
    queue_accepts "$q" "$sel" || continue
    [[ "$(preferred_queue "$c")" == "$q" ]] || continue
    picked=$c
    break
  done
  # Pass 2: fallback select=1 only if preferred queue is busy
  if [[ -z "$picked" ]]; then
    for c in "${PRIORITY[@]}"; do
      is_done "$c" && continue
      is_inflight "$c" && continue
      envf="$LLAMA/cycles/${c}.env"
      [ -f "$envf" ] || continue
      sel=$(cycle_select "$envf")
      [[ "$sel" -eq 1 ]] || continue
      queue_accepts "$q" "$sel" || continue
      pref=$(preferred_queue "$c")
      [[ "$pref" != "$q" ]] || continue
      [[ "$(queue_busy "$pref")" -gt 0 ]] || continue
      picked=$c
      break
    done
  fi
  [[ -n "$picked" ]] || continue
  envf="$LLAMA/cycles/${picked}.env"
  sel=$(cycle_select "$envf")
  pref=$(preferred_queue "$picked")
  script=$PBS
  # TP>12 / multi-node → RPC harness on debug-scaling only
  if [[ "$sel" -gt 1 ]]; then
    script=$PBS_RPC
    q=debug-scaling
    if [[ "$(queue_busy debug-scaling)" -gt 0 ]]; then
      echo "defer $picked: debug-scaling busy (RPC multinode)"
      continue
    fi
  fi
  jid=$(qsub -q "$q" -l "select=${sel}" -v "CYCLE=$picked" -N "ll-$picked" -o "$LOGS/perf_${picked}.out" "$script")
  echo "submitted $picked → $jid on $q select=$sel script=$(basename "$script") (preferred=$pref)"
  submitted=$((submitted + 1))
done

if [[ "$submitted" -eq 0 ]]; then
  echo "nothing to submit (queues full or all PRIORITY done/missing env)"
fi
