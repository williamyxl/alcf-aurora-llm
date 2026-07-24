# Better solutions toward ≫13 tok/s quality-OK (agent review 2026-07-21)

**Expectation:** Max 1550 with resident weights should **beat** Titan V+CPU-offload (~13 decode), not lose by 10×.  
**Ruled out by measurement:** MXFP4→BF16 “upcast tax” as primary bottleneck (BF16/FP16 halfprec ~3 tok/s + quality FAIL).

Agents: [alt stacks](df5a43c8-e804-4968-95c2-802b516cafd5) · [local escapes](a0ff9893-8d88-489c-a242-2ca0726b4312)

---

**Checkpoint policy (2026-07-21):** future work stays on **OpenAI MXFP4** (`models/openai-gpt-oss-120b`). No more BF16/FP16 unquant campaigns.

## Ranked better solutions

### 1. llama.cpp SYCL (full HBM, MXFP4) — **in progress**

| | |
|--|--|
| **Plan** | [`../../build-llamacpp-sycl/PLAN.md`](../../build-llamacpp-sycl/PLAN.md) — smoke → perf → ≤100 cycles until **>30 tok/s** |
| **Constraints** | MXFP4 only; **2 tiles same GPU** (`ZE_AFFINITY_MASK=0.0,0.1`) |
| **Status** | BUILD OK; convert queued; smoke/perf pending |

### 2. Patch fused MoE on current kernels (clamp / interleaved SwiGLU)

| | |
|--|--|
| **Why** | Fused already **~5 decode @ TP=2**; quality FAIL. Suspect: half-split clamp (`fused_moe_interface.py` L376–381). |
| **If quality lands** | ~4× vs REF; may still need more for ≫13. |

### 3. Newer `vllm-xpu-kernels` / vLLM 0.17+ **side env** (not frameworks)

| | |
|--|--|
| **Why** | Arc Pro `!!!!` fixed by newer kernel wheels ([#33679](https://github.com/vllm-project/vllm/pull/33679)). |
| **How** | Self-built conda or Apptainer — **not** `module load frameworks`. |

### 4. Demote / park

| Path | Why |
|------|-----|
| **`module load frameworks`** | **Excluded** |
| BF16/FP16 unquant | **Failed** — casting not the issue; stay on MXFP4 |
| Higher TP under REF | Inverse scaling |
| EP / speculative | Only after quality-OK base ≥~5 |

---

## Recommended next experiment

1. **Finish llama.cpp SYCL build + MXFP4 GGUF convert + quality/perf smoke.**  
2. If FAIL → clamp-skip fused MoE on vLLM.  
3. If still FAIL → newer kernels side env.
