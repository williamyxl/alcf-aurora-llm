# SUCCESS_PERF — S2–S5 performance closure (quality-gated)

**Date:** 2026-07-18  
**Baseline job:** `8680399` (`bench_perf.pbs`, host `x4303c1s3b0n0`)  
**Living log:** [`PERF.md`](PERF.md) · plan: [`PERF_PLAN.md`](PERF_PLAN.md)

## Verdict

**Not a speed breakthrough.** After S2–S5, the best **quality-passing** recipe remains the Phase 5 PASS stack at ≈ **0.37 warm / warm2 e2e tok/s** (cold ≈ **0.29**). Faster MoE paths (~1.47 tok/s) fail the quality gate and are discarded. Engine TTFT unavailable (`ttft_s=null`, `ttft_source=fallback_wall`).

## Best quality-passing recipe (unchanged)

Same as Phase 5 PASS / `SUCCESS_INFER.md` / `infer_chat.pbs`:

| Setting | Value |
|---------|--------|
| TP | **8** |
| MoE | **REF** (`VLLM_XPU_FUSED_MOE_USE_REF=1`) |
| Attention | `TRITON_ATTN` |
| `enforce_eager` | `True` |
| `max_model_len` | `4096` |
| dtype | `bfloat16` |
| Selector | `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` |
| Triton | Intel device extensions + OpenCL-optional `driver.c` patch |

## Phase 0 baseline (KPI)

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
  "text_preview": "analysisWe need to answer three points, under 200 words, numbered. Provide IUPAC isotherm type: Type I (a) typical for microporous materials with strong adsorption at low pressure and plateau. Reason:",
  "quality_ok": true,
  "runs": {
    "warm2": {
      "ttft_s": null,
      "ttft_source": "fallback_wall",
      "e2e_tok_s": 0.37240353868616627,
      "wall_s": 343.71316784900046,
      "n_output_tokens": 128,
      "quality_ok": true
    }
  }
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

Valid TP for 64 heads: 1, 2, 4, 8, 16, … — **not 12**. Full-node tile use needs EP/other sharding, not attn TP=12.

## Pending (paused 2026-07-20)

**Full session recovery:** [`RESUME.md`](RESUME.md) (ordered steps, recipes, OOM/P7 root causes, job ledger, ingest template).

| Item | Status |
|------|--------|
| **P7** true TTFT / prefill / decode tok/s (`disable_log_stats=False`) | **code landed**; validate on resume |
| **TP=2/4/8 scaling** with P7 metrics + KV pin | **standing rule** for all future campaigns |
| Fused MoE quality (path to ≫0.37 tok/s) | next opt after scaling |
| **P4** serve / concurrent | after P7 + scaling |
| **P6** `max_model_len=131072` | after P7; include TP=2/4/8 |

Default context stays `4096`. Engine TTFT was unavailable on job 8680399 (`fallback_wall`); P7 is meant to fix that.

## Related

- Inference gate: [`SUCCESS_INFER.md`](SUCCESS_INFER.md)
- Experiment log: [`PERF.md`](PERF.md)
- Resume checklist: [`RESUME.md`](RESUME.md)
- Team notes: [`perf-team/`](perf-team/)
