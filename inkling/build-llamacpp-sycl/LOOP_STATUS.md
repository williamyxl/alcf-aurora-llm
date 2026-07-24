# Inkling campaign status

Updated: 2026-07-24T03:20:00+0000

## Campaign
**COMPLETE** — Phase **S + M + HBM** done. No further jobs. **Wake can stop.**

## Verdict
**FAIL_OOM** — job **8693927** **C_PG22** FILL done. FILL wrote (~8220 tok est); load OK; `UR_RESULT_ERROR_OUT_OF_RESOURCES` at generate; COMPLETION_EXIT=1; all metrics NA.

## Queue
| Job | Cycle | State | Queue | NDS | Notes |
|-----|-------|-------|-------|-----|-------|
| 8693776 | C_PG24 max-ctx + FILL | **done FAIL_OOM** | debug-scaling | 2 | OUT_OF_RESOURCES @ generate |
| **8693927** | **C_PG22** FILL re-run | **done FAIL_OOM** | debug-scaling | 2 | same OOM @ generate; N_PROMPT=NA |

*(no jobs queued or running)*

## Metrics (C_PG22 FILL)
- COMPLETION_EXIT=1
- FILL_CTX=8192 / approx_tok_est≈8220
- METRICS_N_PROMPT_TOK=NA | TTFT/prefill/gen=NA
- Error: `UR_RESULT_ERROR_OUT_OF_RESOURCES` (ggml-sycl.cpp:3665) at generate start

## Phase wrap-up
| Phase | Status |
|-------|--------|
| **S** short ladder PG8–24 | **complete** |
| **M** max-ctx single + multi-node | **complete** (multi-node fill not viable) |
| **HBM** MoE membind stress | **complete** (C_MO_HBM FAIL_OOM) |

BEST_RECIPE frozen. No further C_PG* FILL re-runs unless recipe changes (FA/KV).

## Prior
- C_PG24 FAIL_OOM with true 8k fill (8693776)
- C_PG22 PASS* short 20-tok only (archived) — superseded by FILL FAIL
- C_PG14–20 max-ctx FAIL_OOM
- Short PG8–24 complete; single-node C_PG8–12 + C_MO1 PASS
