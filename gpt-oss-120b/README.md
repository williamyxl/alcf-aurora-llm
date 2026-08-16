# gpt-oss-120b on Aurora (PVC / Intel Data Center GPU Max 1550)

Inference + perf work for gpt-oss-120b MXFP4 on ALCF Aurora. Two engines characterized: **llama.cpp
SYCL** and **vLLM-XPU**. Decode target **>30 tok/s achieved**.

| | |
|--|--|
| Workdir | `/lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/gpt-oss-120b` |
| Model (vLLM) | `models/openai-gpt-oss-120b` (HF MXFP4) |
| Model (llama.cpp) | `models/openai-gpt-oss-120b-mxfp4.gguf` |
| Status | **WORKING** — vLLM 31.9 tok/s (TP=4), llama.cpp F4_hbm 41.6 tok/s (1 tile) |

## ⭐ Current best results (2026-08-16) — gpt-oss-120b MXFP4, single stream, decode tok/s, quality-OK

| Engine / recipe | 4096 ctx | 131072 ctx | tiles | prefill tok/s |
|-----------------|---------:|-----------:|:-----:|--------------:|
| **llama.cpp F4_hbm** (1-tile MoE→CPU, HBM NUMA) | **41.6** | **38.1** | 1 | ~42–46 |
| llama.cpp P14_tp2 (2-tile pure GPU) | 34.0 | — | 2 | 495 (pp512) |
| **vLLM TP=4** (frameworks 0.15, IPEX Marlin) | **31.9** | 31.4 | 4 | ~1670 |
| vLLM TP=2 | 29.6 | 28.9 | 2 | ~1290 |
| vLLM old self-built REF-MoE (2026-07, superseded) | ~1.2 | — | 2 | — |

**Production recipes:** [`BEST_RECIPES.md`](build-vllm-xpu/opus4.8-new_campaign/BEST_RECIPES.md) —
vLLM full-node (concurrency) + llama.cpp F4_hbm (single user).
**Start here:** [`build-vllm-xpu/opus4.8-new_campaign/README.md`](build-vllm-xpu/opus4.8-new_campaign/README.md)
→ full results [`VLLM_RESULTS.md`](build-vllm-xpu/opus4.8-new_campaign/VLLM_RESULTS.md) ·
concurrency [`CONCURRENCY_RESULTS.md`](build-vllm-xpu/opus4.8-new_campaign/CONCURRENCY_RESULTS.md) ·
timestamped debug [`DEBUG_LOG.md`](build-vllm-xpu/opus4.8-new_campaign/DEBUG_LOG.md).

**Full-node serving:** vLLM 3× TP=4 (all 12 tiles) → **~4565 tok/s aggregate** (36 req/s), ~9–40× the
best llama.cpp full-node option. **Single user:** llama.cpp F4_hbm (1 tile) → **41.6 tok/s** decode.

## Two ways to run gpt-oss-120b on Aurora

### A. vLLM via the Aurora frameworks module (recommended for serving / high prefill)
Use the upgraded **`frameworks/2025.3.1`** env (vllm 0.15 + torch 2.10 + triton 3.6 + ipex 2.10),
which runs gpt-oss MXFP4 via the **IPEX Marlin** MoE backend. Must run under the *full* `module load
frameworks` env (sets SYCL/CCL the IPEX Marlin JIT needs).
```bash
qsub -q debug-scaling -v MODEL=$PWD/models/openai-gpt-oss-120b,TP=4,MML=4096,MNS=1,MEMUTIL=0.85,KVGIB=8 \
  -o build-vllm-xpu/opus4.8-new_campaign/logs/vllm_tp4.out \
  build-vllm-xpu/opus4.8-new_campaign/vllm_run_fw3.pbs
```

### B. llama.cpp SYCL F4_hbm (fastest decode; only single-tile option)
```bash
qsub -q debug-scaling -v CTX=4096 build-vllm-xpu/opus4.8-new_campaign/llama_f4hbm_ctx.pbs   # or CTX=131072
```

> The old self-built **`build-vllm-xpu/env`** stack (vllm 0.27.1 / torch 2.13) does **not** run on PVC
> (attention crashes: triton 3.7.2 `get_native` SIGSEGV; SYCL FA2 xe2-only). oneAPI 2026 fixed its ABI
> but not its attention. Kept for record; do not use for inference. See campaign `DEBUG_LOG.md` P11–P18.

## Historical: self-built stack gates (2026-07, superseded)

| Phase | Status | Artifact |
|-------|--------|----------|
| 0–4 Build stack | PASS | `build-vllm-xpu/VERSIONS.md` |
| 5 Inference (REF MoE, ~1.15 tok/s) | PASS | `build-vllm-xpu/SUCCESS_INFER.md` |
| 6 LoRA/SFT 1 epoch | PASS | `build-vllm-xpu/SUCCESS_TRAIN.md` |

Old perf docs (REF MoE ~1.15 e2e tok/s): [`BEST_PRACTICE.md`](build-vllm-xpu/BEST_PRACTICE.md) ·
[`PERF.md`](build-vllm-xpu/PERF.md) · [`SUCCESS_PERF.md`](build-vllm-xpu/SUCCESS_PERF.md). These are
superseded by the 2026-08 campaign above (25×+ faster).

## Quick start — inference

See **"Two ways to run"** above. Fastest paths:
- vLLM: `build-vllm-xpu/opus4.8-new_campaign/vllm_run_fw3.pbs` (TP=4, frameworks module)
- llama.cpp: `build-vllm-xpu/opus4.8-new_campaign/llama_f4hbm_ctx.pbs` (F4_hbm, 1 tile)

Full recipes + all env details: `build-vllm-xpu/opus4.8-new_campaign/VLLM_WORKING_RECIPE.md`.

## Quick start — LoRA train (1 epoch smoke, old self-built stack)

```bash
qsub train_lora_smoke.pbs
# log: build-vllm-xpu/logs/train_lora.out
# adapter: checkpoints/lora-smoke/adapter/
```

Uses Torch+IPEX+PEFT/TRL (not vLLM). Loads with `Mxfp4Config(dequantize=True)`; TRL `loss_type="nll"`.

## Layout

```
workdir/alcf-aurora-llm/gpt-oss-120b/
  README.md                 # this file
  models/openai-gpt-oss-120b/            # HF MXFP4 (vLLM)
  models/openai-gpt-oss-120b-mxfp4.gguf  # MXFP4 GGUF (llama.cpp)
  build-llamacpp-sycl/      # llama.cpp SYCL build + cycle recipes (F4_hbm, P14_tp2, ...)
    build/bin/              # llama-completion, llama-bench, ...
    cycles/*.env            # recipe env files
  build-vllm-xpu/
    opus4.8-new_campaign/   # ⭐ 2026-08 campaign: WORKING recipes + results
      README.md             # campaign index — START HERE
      VLLM_RESULTS.md       # full results (vLLM TP/context sweep + llama.cpp F4_hbm)
      VLLM_WORKING_RECIPE.md
      DEBUG_LOG.md          # timestamped P1–P21
      vllm_run_fw3.pbs      # working vLLM launcher (frameworks module)
      llama_f4hbm_ctx.pbs   # llama.cpp F4_hbm runner (CTX param)
      vllm_bench2.py        # bench (two-call metrics)
      logs/                 # job output (gitignored)
    BEST_PRACTICE.md        # OLD self-built recipe (superseded)
    env/                    # OLD self-built conda env (does not run on PVC)
    pins.env / VERSIONS.md / PERF.md / SUCCESS_*.md / perf-team/ / patches/
  build_vllm_xpu_*.pbs      # OLD phased build jobs
  infer_chat.pbs / one_chat.py / train_lora_smoke.pbs / lora_one_epoch.py
```

## Stack versions

### Working vLLM stack — Aurora `frameworks/2025.3.1` module (use this)
| Component | Version |
|-----------|---------|
| vllm | 0.15.0 |
| torch | 2.10.0a0 (XPU) |
| triton | 3.6.0 |
| IPEX | 2.10.10 |
| MoE backend | IPEX Marlin (MXFP4) |
| oneAPI | 2025.3 (frameworks module) |

### llama.cpp SYCL (built this campaign)
ggml-org main, SYCL PVC (`-DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DEVICE_ARCH=pvc`),
`build-llamacpp-sycl/build/bin`.

### Old self-built vLLM stack (superseded; does NOT run on PVC — kept for record)
See `build-vllm-xpu/VERSIONS.md`. torch 2.10 + triton 3.6 + IPEX 2.10 + vllm 0.1.dev (REF MoE ~1.15
tok/s). A later self-built vllm 0.27.1 / torch 2.13 attempt (with oneAPI 2026 kernels) crashes in
attention on PVC.

## Patches (must re-apply after reinstall)

| Patch | Why |
|-------|-----|
| `patches/triton_intel_driver_opencl_optional.txt` (+ `driver.c` copy) | AuroraBug#102: L0-only selector used to throw in Triton’s OpenCL twin-device probe |
| `patches/mem_info.cpp.aurora-ze-fallback` | ZE `mem_info` for vllm_xpu_kernels build |
| `patches/block_table_slot_mapping_torch_fallback.txt` | Torch fallback when `HAS_TRITON=False` (kept for resilience) |

## PBS conventions

All smoke/build jobs: `-q debug`, `walltime=00:59:59`, `-A MatSciAI`, `#PBS -j oe`, `filesystems=flare`. One running debug job per user.

## Known pitfalls / key findings (2026-08 campaign)

1. **Max 1550 (PVC) has no native FP4/FP8** (INT4/INT8 only). MXFP4 MoE works via **IPEX Marlin**
   (vLLM frameworks) or llama.cpp SYCL GGUF kernels — not via the self-built cutlass xe2 MoE (crashes).
2. **Use the frameworks module for vLLM** (`frameworks/2025.3.1`, vllm 0.15). The older
   `frameworks/2025.2.0` (vllm 0.10) had no gpt-oss path — source of earlier "frameworks can't run
   gpt-oss" reports. Run under the *full* `module load frameworks` env, else IPEX Marlin JIT throws
   `sycl::exception: No device`.
3. **Self-built vllm 0.27.1 / torch 2.13 does not run on PVC** — triton 3.7.2 `get_native` SIGSEGV and
   SYCL FA2 is xe2-only. oneAPI 2026 fixes the ABI but not attention.
4. **vLLM XPU has no weight/MoE CPU-offload** (`cpu_offload_gb` uses a CUDA-only UVA op). The llama.cpp
   F4_hbm single-tile+CPU-MoE recipe has no vLLM equivalent; min vLLM config is TP=2.
5. **vLLM decode sweet spot = TP=4** (31.9 tok/s); TP=8 regresses on decode (collective overhead) but
   wins prefill. Max context (131072) costs only ~1–2% decode (gpt-oss `sliding_window=128`).
6. **`--kv-cache-memory-gib`** avoids the util-planner OOM (8 at 4096; ~40 at 131072).

### Historical (self-built stack) pitfalls
7. OpenCL in `ONEAPI_DEVICE_SELECTOR` → vLLM SEGV; fused MXFP4 MoE without REF → all-`!`; FLASH_ATTN
   alone → garbled (use TRITON_ATTN); self-built Triton 3.8 JIT broken (keep 3.6); TRL `chunked_nll`
   crashes with `device_map=auto` (use `loss_type="nll"`); MXFP4 training needs `Mxfp4Config(dequantize=True)`.
