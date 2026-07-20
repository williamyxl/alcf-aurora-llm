# RESUME — performance work (paused 2026-07-20)

Paused for several days. Stack bring-up (Phases 0–6) remains **CLOSED**. Resume from this checklist.

## On resume

1. `cd /lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b`
2. Confirm no stale PBS jobs: `qstat -u xiaoliyan`
3. **P7 first (GATE):** `qsub bench_perf.pbs`  
   - Code already has `disable_log_stats=False` in `bench_perf.py` / `one_chat.py`.  
   - Pass when log shows `ttft_source=engine` and numeric `prefill_tok_s` / `decode_tok_s` on warm2.  
   - Ingest into `build-vllm-xpu/PERF.md` + update `SUCCESS_PERF.md` if metrics change narrative.
4. **TP=2/4/8 scaling (standing rule):**  
   - `qsub bench_perf_tp4.pbs` (`debug`)  
   - `qsub bench_perf_tp2.pbs` (`debug-scaling`)  
   - TP=8 = P7 job above  
   - TP2/4 use `--kv-cache-memory-gib 8` (already in PBS) to avoid util-planner OOM.  
   - Ingest dated section in `build-vllm-xpu/perf-team/SCALING_TP248.md`.
5. **Then optimization** (each campaign must include TP=2/4/8 + P7 fields):  
   - Fused MoE quality (largest remaining tok/s lever; fused ~1.47 but quality FAIL)  
   - P4 serve / concurrent  
   - P6 `max_model_len=131072` (`bench_perf_ctx131k.pbs`)

## Current KPI (pre-P7 engine metrics)

| TP | Status | warm2 e2e tok/s | Notes |
|----|--------|-----------------|-------|
| 8 | PASS quality | ≈0.372 | job 8680399; TTFT was `fallback_wall` |
| 4 | OOM (util KV) | — | fixed in PBS via 8 GiB KV pin; not re-validated |
| 2 | OOM (util KV) | — | same |

## Key paths

| Item | Path |
|------|------|
| Plan | `build-vllm-xpu/PERF_PLAN.md` |
| Perf log | `build-vllm-xpu/PERF.md` |
| Closure (S2–S5) | `build-vllm-xpu/SUCCESS_PERF.md` |
| Scaling | `build-vllm-xpu/perf-team/SCALING_TP248.md` |
| Bench | `bench_perf.py`, `bench_perf*.pbs` |
| Pre-P7 TP8 log backup | `build-vllm-xpu/logs/bench_perf.out.pre_p7_8680399` |

## Pause actions taken

- Documented standing rule: every future metric campaign runs TP=2/4/8.
- P7 code landed; validation job **8680906** was still Q (debug starved) — **qdel on pause**; resubmit on resume.
- Agent 30-min progress loop stopped on pause.
