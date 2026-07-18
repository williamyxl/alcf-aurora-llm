# Co-built stack versions — CLOSED (Phases 0–6 PASS)

**Updated:** 2026-07-18T02:30:00+00:00  
**Env:** `/lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b/build-vllm-xpu/env`  
**Gates:** `SUCCESS_INFER.md` + `SUCCESS_TRAIN.md`

## Runtime packages (installed in `$ENV`)

| Package | Version | Notes |
|---------|---------|--------|
| torch | 2.10.0a0+git449b176 | Built for XPU/PVC; pin `v2.10.0` to match IPEX |
| triton | 3.6.0+git225cdbde | **Runtime** frameworks wheel (self-built 3.8 JIT broken; backup `triton-3.8.0-broken-backup.whl`) |
| intel_extension_for_pytorch | 2.10.10+gitd0f992f | |
| oneccl-bind-pt | 2.8.0+xpu | |
| vllm | 0.1.dev1+g109b736b8.d20260717.xpu | `VLLM_TARGET_DEVICE=xpu` |
| vllm-xpu-kernels | 0.1.dev1+g4002cea90.d20260717 | |
| transformers | 5.14.1 | |
| peft | 0.19.1 | Phase 6 |
| trl | 1.8.0 | Phase 6; use `loss_type=nll` |
| datasets | 5.0.0 | Phase 6 |
| accelerate | 1.14.0 | Phase 6 |

## Build wheels (`build-vllm-xpu/xiaoliyan/`)

```
intel_extension_for_pytorch-2.10.10+gitd0f992f-cp312-cp312-linux_x86_64.whl
oneccl_bind_pt-2.8.0+xpu-cp312-cp312-linux_x86_64.whl
torch-2.10.0a0+git449b176-cp312-cp312-linux_x86_64.whl
triton-3.6.0+git225cdbde-cp312-cp312-linux_x86_64.whl   # RUNTIME
triton-3.8.0+gita4fdf97a-cp312-cp312-linux_x86_64.whl   # build artifact; NOT used
triton-3.8.0-broken-backup.whl
torchvision-0.25.0+8ac84ee-cp312-cp312-linux_x86_64.whl  # frameworks; --no-deps
vllm-0.1.dev1+g109b736b8.d20260717.xpu-cp312-cp312-linux_x86_64.whl
vllm_xpu_kernels-0.1.dev1+g4002cea90.d20260717-cp312-cp312-linux_x86_64.whl
```
Install local wheels with:

```bash
pip install --force-reinstall --no-deps path/to/wheel.whl
```

## pins.env (build-time)

See `pins.env`. Torch major.minor must match IPEX (`v2.10.0` / `xpu-main`). Runtime Triton is pinned to frameworks commit `225cdbde` (3.6), not the self-built 3.8 wheel.

## Patches applied in `$ENV`

| File | Purpose |
|------|---------|
| `patches/triton_intel_driver_opencl_optional.txt` | Triton `backends/intel/driver.c`: OpenCL twin probe in try/catch so `level_zero:gpu` works with `TRITON_INTEL_DEVICE_EXTENSIONS` |
| `patches/mem_info.cpp.aurora-ze-fallback` | vllm_xpu_kernels ZE mem_info for Aurora |
| `patches/block_table_slot_mapping_torch_fallback.txt` | vLLM `block_table.py` torch fallback if `HAS_TRITON=False` |

## Runtime recipe pointers

- Inference: `../README.md`, `SUCCESS_INFER.md`, `../infer_chat.pbs`
- Training: `SUCCESS_TRAIN.md`, `../train_lora_smoke.pbs`
- Status log: `PHASE_STATUS.md`
