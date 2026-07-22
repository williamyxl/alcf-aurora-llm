# SUCCESS_PERF — S2–S5 performance closure (quality-gated)

**Date:** 2026-07-18 (closure) · **Updated:** 2026-07-21 (P7 + TP scaling)  
**Baseline job (historical TP=8):** `8680399`  
**Current best practice:** [`BEST_PRACTICE.md`](BEST_PRACTICE.md) — **TP=2**, warm2 e2e ≈ **1.15** tok/s  
**Living log:** [`PERF.md`](PERF.md) · plan: [`PERF_PLAN.md`](PERF_PLAN.md) · scaling: [`perf-team/SCALING_TP248.md`](perf-team/SCALING_TP248.md)

## Verdict

**Current best quality-passing single-stream recipe uses 2 GPU tiles (TP=2):** warm2 e2e ≈ **1.15** tok/s, decode ≈ **1.22** tok/s, engine TTFT ≈ **7.5 s** (job **8681063**).  

Historical Phase 5 / S2–S5 default was TP=8 ≈ **0.37** warm e2e tok/s. Faster MoE paths (~1.47 tok/s) still **fail** the quality gate and are discarded. P7 fixed fake TTFT (`fallback_wall`); true engine metrics are required for all claims.

## Best quality-passing recipe (2026-07-21)

| Setting | Value |
|---------|--------|
| TP | **2** (not 8) |
| MoE | **REF** (`VLLM_XPU_FUSED_MOE_USE_REF=1`) |
| Attention | `TRITON_ATTN` |
| `enforce_eager` | `True` |
| `max_model_len` | `4096` |
| dtype | `bfloat16` |
| KV | `--kv-cache-memory-gib 8`, `max_num_seqs=2` |
| Selector | `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` |
| Triton | Intel device extensions + OpenCL-optional `driver.c` patch |

See [`BEST_PRACTICE.md`](BEST_PRACTICE.md) for full checklist and run commands.

## TP scaling (2026-07-21, P7 metrics)

| TP | Job | warm2 e2e | warm2 decode | warm2 TTFT | quality |
|----|-----|-----------|--------------|------------|---------|
| **2** | **8681063** | **1.147** | **1.220** | **7.50 s** | true |
| 4 | 8681062 | 0.658 | 0.711 | 15.77 s | true |
| 8 | 8681016 | 0.366 | 0.400 | 32.08 s | true |

**Recommendation:** prefer **TP=2** for single-stream. Inverse scaling under REF MoE BS=1.

## Phase 0 baseline (historical KPI, TP=8)

| Run | wall_s | e2e_tok_s | ttft_s | ttft_source | quality_ok |
|-----|--------|-----------|--------|-------------|------------|
| cold | 435.01 | 0.294 | null | fallback_wall | true |
| warm | 343.93 | 0.372 | null | fallback_wall | true |
| warm2 | 343.71 | 0.372 | null | fallback_wall | true |

### PERF_JSON excerpt (job 8680399 warm2)

```json
{
  "n_tiles": 8,
  "moe_mode": "ref",
  "attn": "TRITON_ATTN",
  "dtype": "bfloat16",
  "max_tokens": 128,
  "enforce_eager": true,
  "gpu_memory_utilization": 0.82,
  "cold_ttft_s": null,
  "warm_ttft_s": null,
  "warm2_ttft_s": null,
  "cold_ttft_source": "fallback_wall",
  "warm_ttft_source": "fallback_wall",
  "warm2_ttft_source": "fallback_wall",
  "cold_e2e_tok_s": 0.2942460696768506,
  "warm_e2e_tok_s": 0.372168935232958,
  "warm2_e2e_tok_s": 0.37240353868616627,
  "n_prompt_tokens": 172,
  "n_output_tokens": 128,
  "quality_ok": true
}
```

Raw log: `build-vllm-xpu/logs/bench_perf.out`.

## S2–S5 outcomes

| Stage | Job | Result | Keep? |
|-------|-----|--------|-------|
| **S2** persist caches | 8680469 | warm2 ≈0.37; no within-job steady-state win vs TMPDIR baseline | Yes (ops hygiene); not a speed win |
| **S3** fused MoE | 8680525 | warm2 ≈1.47 tok/s, **quality FAIL** (all-`!` / token-id-0) | **Discard** |
| **S3** mxfp4_fp8 | 8680546 | warm2 ≈1.47 tok/s, **quality FAIL** (same) | **Discard** |
| **S4** `enforce_eager=false` | 8680603 | warm2 ≈0.37, quality OK; compile/cudagraph still NONE | No win vs eager |
| **S5** TP=12 | 8680623 | **FAIL** at LLM init — `64 attn heads % 12 != 0` | Invalid for gpt-oss |

Valid TP for 64 heads: 1, 2, 4, 8, 16, … — **not 12**.

## Pending (updated 2026-07-21)

**Session recovery:** [`RESUME.md`](RESUME.md).

| Item | Status |
|------|--------|
| **P7** true TTFT / prefill / decode | **PASS** (8681016) |
| **TP=2/4/8 scaling** with P7 + KV pin | **COMPLETE** — TP=2 best; see `BEST_PRACTICE.md` |
| Fused MoE quality (path to ≫1.15 tok/s quality-ok) | **FAILED** TP=2/4/8 — see `FUSED_MOE_QUALITY.md` |
| BF16/FP16 unquant MoE | **FAILED** quality (~3 tok/s max) — `HALFPREC_TP248.md` / `FAILED_ATTEMPTS.md` |
| Cast/upcast-as-bottleneck hypothesis | **Killed** |
| Clamp-skip / newer kernels | next — `BETTER_SOLUTIONS.md` (no frameworks module) |
| **P4** serve / concurrent | after a quality-OK fast path |
| **P6** `max_model_len=131072` | include TP=2/4/8 |

Default context stays `4096`.

## Related

- [`BEST_PRACTICE.md`](BEST_PRACTICE.md) — **start here for how to run**
- [`SUCCESS_INFER.md`](SUCCESS_INFER.md) — Phase 5 quality PASS (historical TP=8)
- [`RESUME.md`](RESUME.md) — cold session recovery
