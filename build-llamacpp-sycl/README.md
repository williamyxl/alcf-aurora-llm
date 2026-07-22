# llama.cpp SYCL on Aurora — gpt-oss-120b MXFP4

**Status:** **BUILD OK** → convert **8681247** → then smoke/perf per [`PLAN.md`](PLAN.md)  
**Checkpoint policy:** **MXFP4 only** — `models/openai-gpt-oss-120b`.  
**Hardware policy:** Phase A–E = **2 tiles** of one Max 1550 (`ZE_AFFINITY_MASK=0,1`). **Phase F** = **1 tile** + MoE→CPU. **Phase G (final)** = max ctx **131072** · G0 2-tile vs G1 1-tile MoE→CPU.  
**Success bar:** meaningful text AND eval ≥ **30 tok/s** (max **100** cycles).

## Campaign plan

→ **[`PLAN.md`](PLAN.md)** · ledger [`CYCLE_LOG.md`](CYCLE_LOG.md) · pin [`same_gpu_2tiles.env.sh`](same_gpu_2tiles.env.sh)

```bash
# After GGUF exists:
qsub smoke_llamacpp_sycl.pbs          # Phase C — quality
qsub bench_llamacpp_sycl_perf.pbs     # Phase D/E — tok/s
```


## Goal

Escape hatch from broken/slow vLLM-XPU MoE: different engine, keep MoE on Max 1550 HBM (no CPU MoE offload like Titan V). Quality gate + tok/s vs REF vLLM TP=2 (~1.22 decode).

## Layout

| Path | Role |
|------|------|
| `build-llamacpp-sycl/llama.cpp/` | Source (ggml-org master) |
| `build-llamacpp-sycl/build/` | CMake/Ninja SYCL build tree |
| `build-llamacpp-sycl/logs/` | Build + convert + bench logs |
| `models/openai-gpt-oss-120b/` | **Source MXFP4** HF safetensors |
| `models/openai-gpt-oss-120b-mxfp4.gguf` | Converted GGUF (after convert job) |

## Build (from scratch)

```bash
qsub build_llamacpp_sycl.pbs
# log: build-llamacpp-sycl/logs/build.out
```

CMake (oneAPI 2025.3.1):

```text
GGML_SYCL=ON
GGML_SYCL_F16=ON
GGML_SYCL_DEVICE_ARCH=pvc
CMAKE_C_COMPILER=icx
CMAKE_CXX_COMPILER=icpx
```

Requires: `module load oneapi/release/2025.3.1`, oneMKL (`MKLROOT`), Level Zero at runtime.

## Source pin

- Repo: `ggml-org/llama.cpp` (shallow clone)
- Commit: `76f46ad` (`hexagon: add CLAMP op #25934`)
- SYCL gpt-oss ops present: `add-id`, `mxfp4` cpy/dequant, `swiglu_oai`
- **Build:** login node, ~18 min; `GGML_SYCL=ON` + `F16` + `DEVICE_ARCH=pvc` (AOT). Binaries: `build/bin/llama-cli`, `llama-server`, `llama-bench`. `libggml-sycl.so` ≈ 284 MiB. Log: `logs/build_login.out`.
- **Note:** `llama-cli` needs Level Zero GPU (compute node); login will throw `sycl::exception` at device select — expected.

## Convert / obtain MXFP4 GGUF

**Policy: check Hugging Face before local convert.**

1. Preferred: `ggml-org/gpt-oss-120b-GGUF` → `gpt-oss-120b-MXFP4.gguf`  
   `bash download_gptoss_mxfp4_gguf.sh` or `qsub download_gptoss_mxfp4_gguf.pbs`
2. Fallback only: `qsub convert_gptoss_mxfp4_gguf.pbs` (local HF safetensors → GGUF)

Local convert already produced `models/openai-gpt-oss-120b-mxfp4.gguf` (61G, job 8681247).


## Run (after GGUF) — same-GPU 2 tiles only

```bash
qsub smoke_llamacpp_sycl.pbs
qsub bench_llamacpp_sycl_perf.pbs
```

Tile pin (mandatory): `COMPOSITE` + `ZE_AFFINITY_MASK=0.0,0.1` via `same_gpu_2tiles.env.sh`.  
Iterate per `PLAN.md` until eval ≥30 tok/s or 100 cycles.


## Related

- vLLM best practice (slow, quality OK): `build-vllm-xpu/BEST_PRACTICE.md`
- Failed BF16/FP16 / fused: `build-vllm-xpu/perf-team/FAILED_ATTEMPTS.md`
- Why this path: `build-vllm-xpu/perf-team/BETTER_SOLUTIONS.md`
