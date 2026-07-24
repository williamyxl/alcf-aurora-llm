# Fused MoE quality campaign — TP=2/4/8 (**FAILED quality**)

**Goal:** Quality-correct fused MXFP4 MoE (historically ~1.47 tok/s @ TP=8 but all-`!`).  
**Status:** **CLOSED — FAIL** at every TP. Keep REF + TP=2.

## Code path (vllm_xpu_kernels)

- `VLLM_XPU_FUSED_MOE_USE_REF=1` → `ref_fused_moe` (slow, quality OK).
- REF unset → cutlass grouped GEMM `mxfp4` (faster, quality FAIL).
- Suspect: half-split `gemm1_clamp_limit` on interleaved gpt-oss SwiGLU (`fused_moe_interface.py` L376–381).

## Jobs (2026-07-21)

| TP | Script | Job | Host | Status |
|----|--------|-----|------|--------|
| 2 | `bench_perf_moe_fused_tp2.pbs` | **8681118** | `x4519c0s5b0n0` | **DONE FAIL** |
| 4 | `bench_perf_moe_fused_tp4.pbs` | **8681117** | `x4400c7s0b0n0` | **DONE FAIL** |
| 8 | `bench_perf_moe_fused.pbs` | **8681141** | (see log) | **DONE FAIL** |
| 8 | (hist) | **8680525** | — | FAIL (pre-P7 metrics) |

## Results (warm2)

| TP | Job | warm2_e2e | warm2_decode | warm2_ttft | ttft_src | quality_ok | text |
|----|-----|-----------|--------------|------------|----------|------------|------|
| 2 | 8681118 | **5.10** | **5.17** | 0.53 s | engine | **false** | all-`!` / id0 |
| 4 | 8681117 | **2.99** | **3.03** | 0.89 s | engine | **false** | all-`!` / id0 |
| 8 | 8681141 | **1.41** | **1.46** | 3.75 s | engine | **false** | all-`!` / id0 |
| 8 | 8680525 | ~1.47 | — | — | fallback | **false** | all-`!` / id0 |

Inverse scaling (TP=2 fastest) like REF; quality fails at every TP.

## Verdict

**Discard fused MXFP4 for production.** Related half-prec campaign also FAIL — see [`HALFPREC_TP248.md`](HALFPREC_TP248.md). Next fix attempts: clamp-skip / newer kernels (`BETTER_SOLUTIONS.md`).
