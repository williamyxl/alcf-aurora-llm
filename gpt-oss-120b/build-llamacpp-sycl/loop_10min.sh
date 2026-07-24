#!/bin/bash
# DISABLED — Aurora login nodes forbid long-lived processes.
# Do NOT nohup / background this script. Use one-shot `harvest_once.sh` or ask
# the Cursor agent to qstat/harvest when you say "continue".
#
# Historical: 10-minute harvest → submit loop (kept below for reference only).

echo "REFUSED: loop_10min.sh must not run on login nodes (no long processes)." >&2
echo "Use: bash harvest_once.sh   OR ask the agent to check qstat." >&2
exit 1

set -uo pipefail
ROOT=/lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b
LLAMA=$ROOT/build-llamacpp-sycl
PBS=$ROOT/bench_llamacpp_sycl_perf.pbs
PBS_G=$ROOT/bench_llamacpp_sycl_phaseG.pbs
LOGS=$LLAMA/logs
STATUS=$LLAMA/LOOP_STATUS.md
LEDGER=$LLAMA/CYCLE_LOG.md
mkdir -p "$LOGS"

# Priority: Phase G remaining (F done); step-downs after G0/G1
PRIORITY=(G0 G1 G0s G1s G0t G1t)

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

harvest_line() {
  local c=$1
  local out=$LOGS/perf_${c}.out
  local cli=$LOGS/perf_${c}_cli.log
  local bench=$LOGS/perf_${c}_bench.log
  if [ ! -f "$out" ]; then
    echo "| $c | — | no out |"
    return
  fi
  local sum
  sum=$(bash "$LLAMA/extract_metrics.sh" "$cli" "$bench" 2>/dev/null | grep METRICS_SUMMARY | tail -1 || true)
  local done=no
  grep -q 'PERF_DONE=1' "$out" 2>/dev/null && done=yes
  echo "| $c | done=$done | ${sum:-partial} |"
}

# Append one CYCLE_LOG.md table row if PERF_DONE and not already present
append_ledger_row() {
  local c=$1
  local out=$LOGS/perf_${c}.out
  local cli=$LOGS/perf_${c}_cli.log
  local bench=$LOGS/perf_${c}_bench.log
  [ -f "$out" ] || return 0
  grep -q 'PERF_DONE=1' "$out" 2>/dev/null || return 0
  grep -qE "^\\| \\*?\\*?${c}\\*?\\*? " "$LEDGER" 2>/dev/null && return 0
  local ttft="-" pref="-" gen="-" bttft="-" bpref="-" bgen="-" notes="auto-harvest"
  local sum
  sum=$(bash "$LLAMA/extract_metrics.sh" "$cli" "$bench" 2>/dev/null | grep METRICS_SUMMARY | tail -1 || true)
  if [ -n "$sum" ]; then
    ttft=$(echo "$sum" | sed -n 's/.*ttft_ms=\([0-9.NA]*\).*/\1/p')
    pref=$(echo "$sum" | sed -n 's/.*prefill_tps=\([0-9.NA]*\).*/\1/p')
    gen=$(echo "$sum" | sed -n 's/.* gen_tps=\([0-9.NA]*\).*/\1/p')
    bttft=$(echo "$sum" | sed -n 's/.*bench_ttft_ms=\([0-9.NA]*\).*/\1/p')
    bpref=$(echo "$sum" | sed -n 's/.*bench_prefill_tps=\([0-9.NA]*\).*/\1/p')
    bgen=$(echo "$sum" | sed -n 's/.*bench_gen_tps=\([0-9.NA]*\).*/\1/p')
  fi
  [ -n "$ttft" ] || ttft="-"
  [ -n "$pref" ] || pref="-"
  [ -n "$gen" ] || gen="-"
  [ -n "$bttft" ] || bttft="-"
  [ -n "$bpref" ] || bpref="-"
  [ -n "$bgen" ] || bgen="-"
  grep -q 'COMPLETION_FAIL=1' "$out" 2>/dev/null && notes="COMPLETION_FAIL"
  grep -q 'BENCH_FAIL=1' "$out" 2>/dev/null && notes="${notes}; BENCH_FAIL"
  grep -qiE 'OUT_OF_HOST_MEMORY|out of memory|OOM' "$cli" 2>/dev/null && notes="OOM"
  printf '| %s | - | (auto) | **%s** | **%s** | **%s** | %s | %s | %s | %s |\n' \
    "$c" "$ttft" "$pref" "$gen" "$bttft" "$bpref" "$bgen" "$notes" >> "$LEDGER"
  echo "$(date -Is) LEDGER_APPEND $c gen=$gen" >> "$LOGS/agent_actions.log"
}

submit_one() {
  local c=$1
  local q=${2:-debug-scaling}
  local out=$LOGS/perf_${c}.out
  local envf=$LLAMA/cycles/${c}.env
  local pbs=$PBS
  # Phase G: same 59m PBS as perf (debug queues only — no capacity)
  case "$c" in G*) pbs=$PBS_G ;; esac
  [ -f "$envf" ] || { echo "MISSING_ENV $c"; return 1; }
  local jid
  # Only debug / debug-scaling
  case "$q" in debug|debug-scaling) ;; *) q=debug-scaling ;; esac
  if ! jid=$(qsub -q "$q" -l walltime=00:59:00 -v CYCLE="$c" -N "ll-${c}" -o "$out" "$pbs" 2>/tmp/qsub_loop_err); then
    echo "SUBMIT_FAIL $c $(tr '\n' ' ' </tmp/qsub_loop_err)"
    return 1
  fi
  echo "SUBMITTED $c $jid queue=$q pbs=$(basename "$pbs")"
  echo "$(date -Is) SUBMIT $c $jid $q $(basename "$pbs")" >> "$LOGS/agent_actions.log"
}

tick() {
  local now
  now=$(date -Is)
  local n
  n=$(job_count)
  {
    echo "# Loop status"
    echo
    echo "Updated: $now"
    echo
    echo "## Queue (jobs≈$n)"
    echo '```'
    qstat -u "$USER" 2>/dev/null | head -20 || true
    echo '```'
    echo
    echo "## Harvest"
    echo "| Cycle | State | Metrics |"
    echo "|-------|-------|---------|"
    for c in P14_tp2 P14_tp2b P14_tp4 P14_tp2_igc P14_tp8 P11 P24 P25 F0 F1 F2 F3 F4 F4h G0 G1 G0s G1s G0t G1t; do
      harvest_line "$c"
      append_ledger_row "$c" || true
    done
    echo
    echo "## Best"
    echo "- Real gen: **30.01** (P14_tp2) — target met"
    echo "- Phase F: 1 tile + MoE→CPU + local-DDR NUMA (F1+); F0=unbound baseline"
    echo "- Phase G FINAL: max ctx 131072 — G0 vs G1 (G1 NUMA required)"
    echo "- Next submit priority: ${PRIORITY[*]}"
  } > "$STATUS"

  # Append compact tick to loop log
  echo "$(date -Is) jobs=$n" >> "$LOGS/loop_10min.log"
  for c in P14_tp2 P14_tp2b P14_tp4 P14_tp2_igc P14_tp8; do
    if is_done "$c"; then
      bash "$LLAMA/extract_metrics.sh" "$LOGS/perf_${c}_cli.log" "$LOGS/perf_${c}_bench.log" 2>/dev/null \
        | grep METRICS_SUMMARY | sed "s/^/$c /" >> "$LOGS/loop_10min.log" || true
    fi
  done

  # Submit up to fill ~2 jobs max (leave headroom)
  if [ "$n" -lt 2 ]; then
    for c in "${PRIORITY[@]}"; do
      is_done "$c" && continue
      is_inflight "$c" && continue
      q=debug-scaling
      [ $((RANDOM % 2)) -eq 0 ] && q=debug
      if submit_one "$c" "$q"; then
        echo "submitted $c" >> "$LOGS/loop_10min.log"
        break
      fi
      oq=debug
      [ "$q" = debug ] && oq=debug-scaling
      if submit_one "$c" "$oq"; then
        echo "submitted $c on $oq" >> "$LOGS/loop_10min.log"
        break
      fi
    done
  else
    echo "defer submit jobs=$n" >> "$LOGS/loop_10min.log"
  fi
}

echo "$(date -Is) loop_10min START" | tee -a "$LOGS/loop_10min.log"
# ~48h coverage: Phase G jobs are 6h; queue wait can be long
for round in $(seq 1 288); do
  echo "$(date -Is) tick round=$round" >> "$LOGS/loop_10min.log"
  tick || true

  # Exit early when every priority cycle has PERF_DONE
  all_done=1
  for c in "${PRIORITY[@]}"; do
    if ! is_done "$c"; then
      all_done=0
      break
    fi
  done
  if [ "$all_done" -eq 1 ]; then
    echo "$(date -Is) ALL_PRIORITY_DONE — exiting loop" | tee -a "$LOGS/loop_10min.log"
    break
  fi

  if is_done P14_tp2; then
    g=$(bash "$LLAMA/extract_metrics.sh" "$LOGS/perf_P14_tp2_cli.log" "$LOGS/perf_P14_tp2_bench.log" 2>/dev/null | sed -n 's/.*gen_tps=\([0-9.]*\).*/\1/p' | head -1)
    echo "P14_tp2 gen_tps=$g (target met)" >> "$LOGS/loop_10min.log"
  fi
  sleep 600
done
echo "$(date -Is) loop_10min END" | tee -a "$LOGS/loop_10min.log"
