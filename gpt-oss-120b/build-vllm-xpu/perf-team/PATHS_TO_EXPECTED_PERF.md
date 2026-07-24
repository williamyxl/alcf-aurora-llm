# Paths to expected Intel GPU LLM performance (Aurora / PVC)

**As of:** 2026-07-21  
**Problem:** Quality-OK gpt-oss-120b on Aurora is ~**1.2 decode tok/s** (REF MoE). User baseline: Titan V + CPU-offload MoE/KV ≈ **13 decode / 40+ prefill**. Aurora has ample HBM — **software** is the bottleneck.  
**Agents:** kernel path map + upstream research (this session).

**Bar for “normal” here:** single-stream, quality-OK, warm decode **≥10 tok/s** (Titan-V-class), preferably with prefill ≫40 when resident on-GPU.

---

## Root cause (confirmed by measurement + code)

| Path | Decode @ TP=2 | Quality | What runs |
|------|---------------|---------|-----------|
| `VLLM_XPU_FUSED_MOE_USE_REF=1` | ~1.22 | OK | `ref_fused_moe`: dequant + `torch.matmul` |
| REF unset (fused) | ~5.17 | **FAIL** all-`!` | cutlass grouped GEMM MXFP4 |

Same class always: `XPUExpertsMxFp4` → `XpuFusedMoe`. REF only flips the implementation inside that class.

**Strong local bug suspect:** fused path applies **contiguous half-split** `gemm1_clamp_limit` on gpt-oss **interleaved** gate/up (`fused_moe_interface.py` L376–381). REF never does that half-split; `swigluoai_and_mul` expects interleaved layout. That asymmetry alone can destroy logits → token-id-0.

Upstream ([vLLM #33679](https://github.com/vllm-project/vllm/pull/33679)): gpt-oss MXFP4 XPU had **accuracy / `!!!!` issues** after IPEX→`vllm-xpu-kernels` migration; Intel’s workarounds were (1) rollback **v0.15.1 + IPEX**, (2) Triton attn; later **newer kernel wheels** fixed Arc Pro cases. Our pins (`g4002cea90` / `g109b736b8`, 2026-07-17) may predate fixes — **unverified on PVC**.

---

## Ranked ways to reach normal performance

### 1b. Download BF16 / FP16 checkpoints — **TRIED; FAILED quality (2026-07-21)**

| | |
|--|--|
| **Result** | BF16 TP4 warm2 decode ~**3.0**; FP16 TP4 ~**2.8**; all `quality_ok=false` (all-`!`). TP2 OOM. |
| **Jobs** | 8681162/63, 8681177/78, 8681207 (OOM); see [`HALFPREC_TP248.md`](HALFPREC_TP248.md) |
| **Conclusion** | **Casting / MXFP4 storage is not the bottleneck.** Discard for production. |

### 1. Fix fused MXFP4 MoE (or upgrade kernels) — **P0 / still open**

| | |
|--|--|
| **Fused campaign** | TP2/4/8 all quality FAIL; best ~5.2 decode @ TP=2 — [`FUSED_MOE_QUALITY.md`](FUSED_MOE_QUALITY.md) |
| **Ledger** | [`FAILED_ATTEMPTS.md`](FAILED_ATTEMPTS.md) |

| | |
|--|--|
| **Why** | Fused already ~**5×** REF on this node; quality is the only blocker. Upstream has been fixing related garbage-output bugs via kernel bumps. |
| **How** | (a) Patch: skip half-split clamp for `swigluoai` / interleaved gpt-oss; (b) single-layer fused-vs-REF tensor diff; (c) rebuild or pull newer `vllm-xpu-kernels` + matching vLLM; (d) re-bench TP=2 fused with quality gate. |
| **Expected** | Quality @ ~5 tok/s immediately if clamp/layout is the bug; further headroom toward ≥10 with less host overhead / better GEMM. **Not guaranteed ≥10** until measured. |
| **Effort** | Med (patch) → High (full upgrade/bisect) |
| **Risk** | Med — PVC may differ from Arc validation hardware |

### 2. Side stack: vLLM **0.15.1 + IPEX** MoE (pre–kernel migration)

| | |
|--|--|
| **Why** | Intel explicitly recommended this when kernel-path gpt-oss went gibberish. |
| **How** | Second conda env; do **not** displace PASS stack until quality+speed proven. |
| **Expected** | Unknown on Aurora PVC; possible quality-OK faster MoE than REF. |
| **Effort** | High | **Risk** | Med–High (dual stack; may still be slow on PVC) |

### 3. After fused is correct: EP / serve / speculative

| Lever | Role |
|-------|------|
| **Expert parallel** | Experimental on XPU; helps multi-tile MoE **after** fused works. Under REF, more TP already **hurts**. |
| **Continuous batching (P4)** | Raises **aggregate** tok/s; does not fix BS=1 chat lag alone. |
| **Speculative decoding** | Intel lists experimental; ~1.3–2× only on a healthy base (≥5 tok/s). Useless on REF@1.2. |

### 4. llama.cpp SYCL — **in progress (MXFP4, same-GPU 2 tiles)**

| | |
|--|--|
| **Plan** | [`../../build-llamacpp-sycl/PLAN.md`](../../build-llamacpp-sycl/PLAN.md) — ≤100 cycles to **>30 tok/s** |
| **Pin** | `ZE_AFFINITY_MASK=0.0,0.1` (GPU0 both tiles) |
| **Status** | BUILD OK; convert `8681247` Q; smoke/perf after GGUF |

### 5. Long shots (parked)

- IPEX-LLM FlashMoE, OpenVINO — wrong quant/model story or immature for this checkpoint.

---

## Not worth retrying (already failed)

- Fused / `mxfp4_fp8` **without** a code/kernel fix  
- Higher TP under REF (inverse scaling)  
- `enforce_eager=False` alone  
- FLASH_ATTN for gpt-oss XPU  
- OpenCL in device selector  
- TP=12  
- Expecting native FP4 tensor-core magic on PVC  

---

## Recommended execution sequence

1. **llama.cpp SYCL** per [`PLAN.md`](../../build-llamacpp-sycl/PLAN.md): GGUF → smoke (meaningful text) → perf on **same-GPU 2 tiles** → iterate ≤100 until **>30 tok/s**.  
2. If FAIL at 100 → **clamp-skip fused MoE** on vLLM TP=2.  
3. If still FAIL → **newer kernels** / layer-wise REF vs fused dump.  
4. Only then: EP / serve / specdec / P6 131k.

Until a quality-OK fast path exists, **REF+TP=2 remains a correctness floor, not a performance target.**

---

## References

- Local: `BEST_PRACTICE.md`, `FUSED_MOE_QUALITY.md`, `fused_moe_interface.py` L376–381  
- [vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels)  
- [vLLM #33679 MXFP4 MoE + accuracy thread](https://github.com/vllm-project/vllm/pull/33679)  
- [RFC #33214 XPU kernel migration](https://github.com/vllm-project/vllm/issues/33214)  
- [ALCF Aurora vLLM](https://docs.alcf.anl.gov/aurora/data-science/inference/vllm/) (Llama/TP ops; little MoE MXFP4 perf guidance)  
- [Intel vLLM XPU model matrix](https://docs.vllm.ai/en/stable/models/hardware_supported_models/xpu/) (lists gpt-oss MXFP4; validated primarily on Arc Pro B-series)
