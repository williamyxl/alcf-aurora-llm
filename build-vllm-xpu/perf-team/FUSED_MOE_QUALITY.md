# Fused MoE quality campaign — TP=2/4/8

**Goal:** Find a quality-correct fused / faster MoE path (historically ~1.47 tok/s @ TP=8 but all-`!` / token-id-0).  
**Rule:** `quality_ok=false` → discard from ranking; keep REF + TP=2 as best practice until a quality win.

## Code path (vllm_xpu_kernels)

- Env `VLLM_XPU_FUSED_MOE_USE_REF=1` → `ref_fused_moe` in `moe_utils.py` (dequant + naive GEMM).
- REF unset + MXFP4 weights → fused recipe `mxfp4` (or `mxfp4_fp8` if `VLLM_XPU_FUSED_MOE_USE_MXFP4_FP8=1`).
- Source: `site-packages/vllm_xpu_kernels/fused_moe_interface.py` (`_use_ref`, `_get_recipe`).

## Jobs (2026-07-21)

| TP | Script | Job | Queue | Status |
|----|--------|-----|-------|--------|
| 2 | `bench_perf_moe_fused_tp2.pbs` | **8681118** | debug-scaling | queued |
| 4 | `bench_perf_moe_fused_tp4.pbs` | **8681117** | debug | queued |
| 8 | `bench_perf_moe_fused.pbs` | (submit after 2/4 finish or slot free) | debug | pending |

## Results (fill after PERF_JSON)

| TP | Job | warm2_e2e | warm2_decode | warm2_ttft | ttft_src | quality_ok | notes |
|----|-----|-----------|--------------|------------|----------|------------|-------|
| 2 | 8681118 | | | | | | |
| 4 | 8681117 | | | | | | |
| 8 | | | | | | | Historical 8680525: ~1.47 e2e, FAIL |

## Next if all quality FAIL

1. Compare fused vs REF intermediate tensors (single layer / first MoE block) offline if feasible.
2. Try `mxfp4_fp8` at TP=2 only (already failed @ TP=8).
3. Check upstream vllm-xpu-kernels issues for MXFP4 accuracy on PVC.
4. Do **not** change default recipe away from REF+TP=2 without quality_ok.
