# Inkling campaign status

Updated: 2026-07-24T02:08:00+00:00

## Verdict
**Q** — 8693776 C_PG24 still queued (debug-scaling, 2nds); no perf_C_PG24.out / metrics yet.

## Queue
| Job | Cycle | State | Queue | NDS | Wall |
|-----|-------|-------|-------|-----|------|
| **8693776** | **C_PG24** max-ctx + FILL | **Q** | debug-scaling | 2 | 00:59 |

## Logs (`perf_C_PG24.out` / rpc)
- **Not present** (job has not started). Only stale `perf_C_PG24.out.skipped_wrongly`.
- qstat -f: `job_state = Q`; `comment = Not Running: Insufficient amount of resource: at_queue`
- PERF_DONE / COMPLETION_EXIT / METRICS_SUMMARY / N_PROMPT_TOK / FILL_CTX / OUT_OF_RESOURCES: **N/A**

## Next
When 8693776 finishes: harvest metrics; if PASS with real fill → log CYCLE_LOG + resubmit **C_PG22** fill for true max-ctx metrics. If FAIL_OOM → try C_PG22 fill. If PASS but 20-tok → fix PBS + resubmit C_PG24.

## Prior
- C_PG22 PASS (short 20-tok prompt; FILL_CTX missing in RPC PBS — **fixed**)
- C_PG14–20 max-ctx FAIL_OOM
- Short PG8–24 complete
