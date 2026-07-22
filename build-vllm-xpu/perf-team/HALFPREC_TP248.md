# BF16 / FP16 unquantized MoE — TP=2/4/8 (**FAILED quality**)

**Campaign:** 2026-07-21  
**Status:** **CLOSED — FAIL** (all successful runs `quality_ok=false`, all-`!` / token-id-0)  
**Hypothesis tested:** Resident BF16/FP16 MoE weights bypass broken MXFP4 fused/REF and reach ≫1.2 (ideally ≫13) decode tok/s with quality.  
**Outcome:** Speed only ~**3 tok/s** best case (TP=4); **quality FAIL** same as fused MXFP4. **Precision casting is not the bottleneck.**

## Setup

| Item | Value |
|------|--------|
| BF16 ckpt | `models/openai-gpt-oss-120b-bf16` (`unsloth/gpt-oss-120b-BF16`, ~218 GiB, `quantization_config=None`) |
| FP16 ckpt | `models/openai-gpt-oss-120b-fp16` (`twhitworth/gpt-oss-120b-fp16`, ~218 GiB, `quantization_config=None`) |
| Harness | `bench_perf_halfprec.pbs` + `bench_perf.py --model/--dtype` |
| Env | `VLLM_XPU_FUSED_MOE_USE_REF` **unset**; expect unquantized `XPUExperts` |
| Attn | `TRITON_ATTN`, eager, `max_model_len=4096` |

### KV plan (tile ≈ 64 GiB)

| TP | Weights / tile | KV pin | Result |
|----|----------------|--------|--------|
| 2 | ~109 GiB | 1 GiB | FP16 **OOM**; BF16 still running / expect OOM |
| 4 | ~54.5 GiB | 4 GiB | Ran; quality FAIL |
| 8 | ~27 GiB | 8 GiB | Ran; quality FAIL |

## Jobs

| Dtype | TP | KV | Job | Queue | Status |
|-------|----|----|-----|-------|--------|
| bf16 | 8 | 8 | **8681162** | debug | **DONE FAIL** |
| bf16 | 4 | 4 | **8681163** | debug-scaling | **DONE FAIL** |
| bf16 | 2 | 1 | **8681210** | debug | R / expect OOM |
| fp16 | 8 | 8 | **8681177** | debug | **DONE FAIL** |
| fp16 | 4 | 4 | **8681178** | debug-scaling | **DONE FAIL** |
| fp16 | 2 | 1 | **8681207** | debug-scaling | **DONE OOM** |

Logs: `build-vllm-xpu/logs/bench_perf_{bf16,fp16}_tp{2,4,8}.out`

## Results (warm2, engine TTFT)

| Dtype | TP | warm2_e2e | warm2_decode | warm2_ttft | warm2_prefill | quality_ok | text |
|-------|----|-----------|--------------|------------|---------------|------------|------|
| bf16 | 8 | 1.41 | 1.45 | 3.37 s | 51 | **false** | all-`!` / id0 |
| bf16 | 4 | **2.96** | **2.98** | 0.74 s | 232 | **false** | all-`!` / id0 |
| fp16 | 8 | 1.51 | 1.58 | 4.35 s | 40 | **false** | all-`!` / id0 |
| fp16 | 4 | 2.77 | 2.80 | 0.72 s | 237 | **false** | all-`!` / id0 |
| fp16 | 2 | — | — | — | — | — | **XPU OOM** (~1014 MiB alloc fail) |
| bf16 | 2 | — | — | — | — | — | pending / expect OOM |

### Comparison baselines

| Recipe | TP | warm2 decode | quality |
|--------|----|--------------|---------|
| REF MXFP4 (best practice) | 2 | **1.22** | OK |
| Fused MXFP4 | 2 | **5.17** | FAIL |
| Unquant BF16 (this campaign) | 4 | **2.98** | FAIL |

## Verdict

1. **Discard BF16/FP16 unquantized MoE for production** on this stack — same garbage as fused MXFP4.
2. **MXFP4→BF16 upcast is not why Aurora is ~1 tok/s.** Resident half-prec still ≪13 and quality-broken.
3. Bug is in the **non-REF MoE (or shared) XPU path**, not weight storage format.
4. Keep **REF + TP=2** (`BEST_PRACTICE.md`) until a quality-OK fast path exists.
5. Next: fused clamp-skip / newer kernels (see `BETTER_SOLUTIONS.md`) — **not** more dtype A/B.

## Do not retry without a code/kernel change

- Re-download other BF16 mirrors expecting a different outcome  
- TP=2 half-prec without multi-node / more HBM per shard  
- Claiming half-prec “wins” on e2e while `quality_ok=false`
