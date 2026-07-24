# gpt-oss-120b on Aurora (self-built Torch-XPU + vLLM)

Self-contained **Torch-XPU + IPEX + oneCCL + vLLM** stack for ALCF Aurora (PVC / Intel Data Center GPU Max). **No `module load frameworks`.**

| | |
|--|--|
| Workdir | `/lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b` |
| Env | `build-vllm-xpu/env` (Python 3.12) |
| Model | `models/openai-gpt-oss-120b` (MXFP4 MoE) |
| Status | Phases 0–6 **CLOSED**; perf **IN PROGRESS** — best practice: [`build-vllm-xpu/BEST_PRACTICE.md`](build-vllm-xpu/BEST_PRACTICE.md) |

## Project gates

| Phase | Status | Artifact |
|-------|--------|----------|
| 0–4 Build stack | PASS | `build-vllm-xpu/VERSIONS.md` |
| 5 Inference | PASS | `build-vllm-xpu/SUCCESS_INFER.md` |
| 6 LoRA/SFT 1 epoch | PASS | `build-vllm-xpu/SUCCESS_TRAIN.md` |

**Performance (2026-07-21):** **TP=2 is best** for single-stream REF MoE (warm2 ≈ **1.15** e2e tok/s).  
**Start here:** [`BEST_PRACTICE.md`](build-vllm-xpu/BEST_PRACTICE.md) · scaling: [`SCALING_TP248.md`](build-vllm-xpu/perf-team/SCALING_TP248.md)  
**Session recovery:** [`RESUME.md`](build-vllm-xpu/RESUME.md) · log: [`PERF.md`](build-vllm-xpu/PERF.md) · closure: [`SUCCESS_PERF.md`](build-vllm-xpu/SUCCESS_PERF.md).

Living log: `build-vllm-xpu/PHASE_STATUS.md`

## Quick start — inference

```bash
cd /lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b
qsub infer_chat.pbs
# log: build-vllm-xpu/logs/test_gptoss.out
```

Required runtime recipe (also in `SUCCESS_INFER.md` / `infer_chat.pbs`):

- `module load oneapi/release/2025.3.1` only (no frameworks)
- `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`
- `TRITON_INTEL_DEVICE_EXTENSIONS="cl_intel_subgroup_matrix_multiply_accumulate cl_intel_subgroup_matrix_multiply_accumulate_tensor_float32 cl_intel_subgroup_2d_block_io cl_intel_bfloat16_conversions"`
- `VLLM_XPU_FUSED_MOE_USE_REF=1`
- `attention_backend="TRITON_ATTN"`, **TP=2** (best practice; historical PASS used TP=8)
- `--kv-cache-memory-gib 8` / `kv_cache_memory_bytes` for TP=2/4 (avoids util-planner OOM)
- Triton `driver.c` OpenCL try/catch patch (see `build-vllm-xpu/patches/`)

Do **not** put OpenCL in the device selector (`*:gpu` / dual) — Triton smoke may pass, then vLLM multiprocess SEGVs.

## Quick start — LoRA train (1 epoch smoke)

```bash
qsub train_lora_smoke.pbs
# log: build-vllm-xpu/logs/train_lora.out
# adapter: checkpoints/lora-smoke/adapter/
```

Uses Torch+IPEX+PEFT/TRL (not vLLM). Loads with `Mxfp4Config(dequantize=True)`; TRL `loss_type="nll"`.

## Layout

```
workdir/llm/gpt-oss-120b/
  README.md                 # this file
  models/openai-gpt-oss-120b/
  checkpoints/lora-smoke/adapter/
  infer_chat.pbs / one_chat.py / triton_xpu_smoke.py
  train_lora_smoke.pbs / lora_one_epoch.py
  build_vllm_xpu_*.pbs      # phased build jobs
  build-vllm-xpu/
    BEST_PRACTICE.md        # current recommended recipe (TP=2)
    env/                    # conda env
    pins.env
    VERSIONS.md
    PHASE_STATUS.md
    SUCCESS_INFER.md
    SUCCESS_TRAIN.md
    SUCCESS_PERF.md         # S2–S5 perf closure
    PERF.md                 # living perf experiment log
    RESUME.md               # pause/resume checklist (perf)
    PERF_PLAN.md
    perf-team/              # A1–A6 + SCALING_TP248.md
    patches/                # runtime/source patches
    xiaoliyan/              # built wheels
    logs/
```

## Stack versions (summary)

See `build-vllm-xpu/VERSIONS.md` for full detail.

| Component | Runtime version |
|-----------|-----------------|
| torch | 2.10.0a0+git449b176 (XPU/PVC) |
| triton | **3.6.0+git225cdbde** (frameworks wheel; self-built 3.8 JIT broken) |
| IPEX | 2.10.10+gitd0f992f |
| oneccl-bind-pt | 2.8.0+xpu |
| vllm | 0.1.dev1+g109b736b8 (XPU) |
| vllm-xpu-kernels | 0.1.dev1+g4002cea90 |
| peft / trl | 0.19.1 / 1.8.0 |

## Patches (must re-apply after reinstall)

| Patch | Why |
|-------|-----|
| `patches/triton_intel_driver_opencl_optional.txt` (+ `driver.c` copy) | AuroraBug#102: L0-only selector used to throw in Triton’s OpenCL twin-device probe |
| `patches/mem_info.cpp.aurora-ze-fallback` | ZE `mem_info` for vllm_xpu_kernels build |
| `patches/block_table_slot_mapping_torch_fallback.txt` | Torch fallback when `HAS_TRITON=False` (kept for resilience) |

## PBS conventions

All smoke/build jobs: `-q debug`, `walltime=00:59:59`, `-A MatSciAI`, `#PBS -j oe`, `filesystems=flare`. One running debug job per user.

## Known pitfalls

1. **OpenCL in `ONEAPI_DEVICE_SELECTOR`** → vLLM SEGV after Triton smoke OK.
2. **Fused XPU MXFP4 MoE without REF** → all-`!` / token id 0; set `VLLM_XPU_FUSED_MOE_USE_REF=1`.
3. **FLASH_ATTN alone on gpt-oss XPU** → garbled / zero tokens; use `TRITON_ATTN`.
4. **Self-built Triton 3.8** → JIT broken; keep 3.6 + Aurora `driver.c` patch.
5. **TRL default `chunked_nll`** → crashes with `device_map="auto"`; use `loss_type="nll"`.
6. **MXFP4 training** → transformers requires `Mxfp4Config(dequantize=True)`.
7. **TP=8 for single-stream** → slow (~0.37 e2e); use **TP=2** (~1.15 e2e) — see `BEST_PRACTICE.md`.
8. **TP=2/4 without KV pin** → OOM (util planner reserves ~50 GiB KV); use `--kv-cache-memory-gib 8`.
