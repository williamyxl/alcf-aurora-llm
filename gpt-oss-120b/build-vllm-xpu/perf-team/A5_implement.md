# A5 — Implementation: gpt-oss-120b XPU perf (Aurora)

**Agent:** A5 (Implementer)  
**Date:** 2026-07-18  
**Input:** `A4_review.md` (authoritative); `A3_solutions.md`  
**Scope:** S1 → S2 → S3(harness) → S4(harness) → S5(harness). Deferred S6/S7 untouched.

---

## Summary

Shipped metric hygiene (null TTFT + `ttft_source`), durable-cache PBS (default TMPDIR unchanged), and MoE / eager / TP12 experiment harnesses. Defaults remain PASS: REF MoE, `enforce_eager=true`, TP=8, util 0.82. **No experimental jobs submitted** (8680399 Phase 0 baseline left alone).

---

## Files touched

| Path | Change |
|------|--------|
| `one_chat.py` | S1: `ttft_s=null` + `ttft_source` (`engine` / `fallback_wall`); always `wall_s` / `e2e_tok_s` |
| `bench_perf.py` | S1: same metrics; S4: `--enforce-eager` (default true); S5: `--gpu-memory-utilization` (default 0.82); PERF_JSON enriched |
| `bench_perf.pbs` | **unchanged** (per-job TMPDIR caches for 8680399 continuity) |
| `bench_perf_persist.pbs` | **new** S2: durable `$WORKDIR/.cache/{triton,sycl}_xpu_gptoss`; JIT warn helper |
| `infer_chat.pbs` | S2: same durable cache paths; still `unset SYCL_CACHE_PERSISTENT` |
| `bench_perf_moe_ref.pbs` | **new** S3: REF=1, `--moe-mode ref` |
| `bench_perf_moe_fused.pbs` | **new** S3: unset REF; `--moe-mode fused` |
| `bench_perf_moe_fp8.pbs` | **new** S3: unset REF; `MXFP4_FP8=1`; `--moe-mode mxfp4_fp8` |
| `bench_perf_graphs.pbs` | **new** S4: `--enforce-eager false`; Dynamo/compile still disabled |
| `bench_perf_tp12.pbs` | **new** S5: `--tp 12 --moe-mode ref` |
| `.gitignore` / `FILES.md` | Allowlist all new `bench_perf_*.pbs` |
| `build-vllm-xpu/PERF.md` | One-line: Phase 5 “343 s TTFT” was e2e wall |
| `build-vllm-xpu/SUCCESS_INFER.md` | Same TTFT note; durable-cache recipe row |
| `build-vllm-xpu/perf-team/A5_implement.md` | This report |

**Not touched:** S6 (`infer_serve` / `bench_serve`), S7 (FLASH_ATTN), stack rebuild, OpenCL selector, `SYCL_CACHE_PERSISTENT`.

---

## Jobs submitted / not submitted

| Job / PBS | Status |
|-----------|--------|
| `bench_perf.pbs` (8680399) | Left as-is (already queued Phase 0); **not** re-submitted |
| `bench_perf_persist.pbs` | **Not submitted** (gated: prefer ingest P0 first) |
| `bench_perf_moe_{ref,fused,fp8}.pbs` | **Not submitted** |
| `bench_perf_graphs.pbs` | **Not submitted** |
| `bench_perf_tp12.pbs` | **Not submitted** |
| `infer_chat.pbs` | **Not re-submitted** (cache path change only) |

---

## S1 — Metric hygiene

- Missing / ≤0 `first_token_latency` → `ttft_s: null` (JSON null), `ttft_source: "fallback_wall"`.
- Engine TTFT present → `ttft_source: "engine"`.
- Per-run `wall_s` / `e2e_tok_s` always present; top-level PERF_JSON has `*_ttft_source` fields.

## S2 — Persist caches + JIT helper

- Default `bench_perf.pbs`: still `$TMPDIR/..._${JOBTAG}`.
- Persist / chat: `TRITON_CACHE_DIR=$WORKDIR/.cache/triton_xpu_gptoss`, `SYCL_CACHE_DIR=$WORKDIR/.cache/sycl_xpu_gptoss`, `unset SYCL_CACHE_PERSISTENT`.
- After persist run, PBS echoes counts:
  ```bash
  grep -c 'reduce_segments' $LOGS/bench_perf_persist.out
  grep -c 'jit_monitor' $LOGS/bench_perf_persist.out
  ```
  Manual cross-job: compare those counts + `cold_*` vs prior `warm_*` when dirs persist.

## S3–S5 harness defaults

| Lever | Default (PASS) | Experiment PBS |
|-------|----------------|----------------|
| MoE | `VLLM_XPU_FUSED_MOE_USE_REF=1` | fused / fp8 clones |
| Eager | `--enforce-eager true` | `bench_perf_graphs.pbs` → false |
| TP / util | `--tp 8`, `--gpu-memory-utilization 0.82` | `bench_perf_tp12.pbs` → tp 12 |

`one_chat.py` still hardcodes `enforce_eager=True` (A4: do not flip).

---

## How A6 should verify

1. **S1:** Parse a METRICS_JSON / PERF_JSON where engine TTFT is absent → `ttft_s` is null, `ttft_source=fallback_wall`, `wall_s`/`e2e_tok_s` present; never treat null/fallback as first-token.
2. **S2:** Confirm `bench_perf.pbs` still uses TMPDIR; persist PBS + `infer_chat.pbs` use durable paths and never set `SYCL_CACHE_PERSISTENT`; JIT helper greps documented above.
3. **S3–S5:** CLI/PBS exist; defaults REF + eager true + TP8; `quality_ok` exit-2 gate intact; no OpenCL in `ONEAPI_DEVICE_SELECTOR`.
4. **Jobs:** No fused/graphs/TP12 submit in this pass; 8680399 remains the P0 baseline.
5. **Deferred:** S6/S7 absent (no serve bench, no FLASH).

**A6 fail if:** default recipe moved to fused/graphs/TP12; TTFT still equals wall without `ttft_source`; OpenCL / `SYCL_CACHE_PERSISTENT` reintroduced; quality-fail ranked as win.
