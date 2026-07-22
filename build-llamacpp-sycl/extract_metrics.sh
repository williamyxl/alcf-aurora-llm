#!/bin/bash
# Extract TTFT / prefill TPS / generation TPS from llama.cpp logs.
# Usage: extract_metrics.sh <cli_or_smoke_log> [bench_log]
# Portable: grep/sed/awk only (no rg — missing on Aurora compute nodes).

set -uo pipefail

CLI_LOG=${1:-}
BENCH_LOG=${2:-}

TTFT_MS=""
PREFILL_TPS=""
GEN_TPS=""
N_PROMPT_TOK=""
N_GEN_TOK=""
BENCH_PREFILL_TPS=""
BENCH_GEN_TPS=""
BENCH_TTFT_MS=""

if [ -n "$CLI_LOG" ] && [ -f "$CLI_LOG" ]; then
  line=$(grep -a "prompt eval time" "$CLI_LOG" 2>/dev/null | tail -1 || true)
  if [ -n "$line" ]; then
    TTFT_MS=$(echo "$line" | sed -n 's/.*prompt eval time =\s*\([0-9.]*\) ms.*/\1/p')
    N_PROMPT_TOK=$(echo "$line" | sed -n 's/.*ms \/ *\([0-9]*\) tokens.*/\1/p')
    PREFILL_TPS=$(echo "$line" | sed -n 's/.*, *\([0-9.]*\) tokens per second.*/\1/p')
  fi
  line=$(grep -a "eval time =" "$CLI_LOG" 2>/dev/null | grep -v "prompt eval" | tail -1 || true)
  if [ -n "$line" ]; then
    N_GEN_TOK=$(echo "$line" | sed -n 's/.*ms \/ *\([0-9]*\) runs.*/\1/p')
    GEN_TPS=$(echo "$line" | sed -n 's/.*, *\([0-9.]*\) tokens per second.*/\1/p')
  fi
fi

if [ -n "$BENCH_LOG" ] && [ -f "$BENCH_LOG" ]; then
  BENCH_PREFILL_TPS=$(grep -a "pp512" "$BENCH_LOG" 2>/dev/null | tail -1 | sed -n 's/.*pp512[^0-9]*\([0-9.]*\).*/\1/p' || true)
  BENCH_GEN_TPS=$(grep -a "tg128" "$BENCH_LOG" 2>/dev/null | tail -1 | sed -n 's/.*tg128[^0-9]*\([0-9.]*\).*/\1/p' || true)
  if [ -n "${BENCH_PREFILL_TPS:-}" ]; then
    BENCH_TTFT_MS=$(awk -v pp="$BENCH_PREFILL_TPS" 'BEGIN{ if (pp+0>0) printf "%.2f", 1000.0*512.0/pp; }')
  fi
fi

echo "METRICS_TTFT_MS=${TTFT_MS:-NA}"
echo "METRICS_PREFILL_TPS=${PREFILL_TPS:-NA}"
echo "METRICS_GEN_TPS=${GEN_TPS:-NA}"
echo "METRICS_N_PROMPT_TOK=${N_PROMPT_TOK:-NA}"
echo "METRICS_N_GEN_TOK=${N_GEN_TOK:-NA}"
echo "METRICS_BENCH_PREFILL_TPS=${BENCH_PREFILL_TPS:-NA}"
echo "METRICS_BENCH_GEN_TPS=${BENCH_GEN_TPS:-NA}"
echo "METRICS_BENCH_TTFT_MS=${BENCH_TTFT_MS:-NA}"
echo "METRICS_SUMMARY ttft_ms=${TTFT_MS:-NA} prefill_tps=${PREFILL_TPS:-NA} gen_tps=${GEN_TPS:-NA} | bench_ttft_ms=${BENCH_TTFT_MS:-NA} bench_prefill_tps=${BENCH_PREFILL_TPS:-NA} bench_gen_tps=${BENCH_GEN_TPS:-NA}"
