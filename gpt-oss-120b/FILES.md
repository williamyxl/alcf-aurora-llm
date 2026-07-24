# Files to git-track

Policy: **scripts + docs for setup / inference / training only**.  
Never: model weights, LoRA adapters, wheels, conda `env/`, `build-src/`, logs.

## Track

```
.gitignore
README.md
FILES.md                 # this list
install.sh
download_model.sh

# stack build (PBS)
build_vllm_xpu_phaseA.pbs
build_vllm_xpu_llvm_triton.pbs
build_vllm_xpu_torch.pbs
build_vllm_xpu_torch_install.pbs
build_vllm_xpu_ipex.pbs
build_vllm_xpu_ipex_install.pbs
build_vllm_xpu_ccl.pbs
build_vllm_xpu_kernels.pbs
build_vllm_xpu_vllm.pbs
build_vllm_xpu_repair_env.pbs

# inference
infer_chat.pbs
infer_serve.pbs          # if present
one_chat.py
triton_xpu_smoke.py
bench_perf.pbs
bench_perf.py
bench_perf_persist.pbs
bench_perf_moe_ref.pbs
bench_perf_moe_fused.pbs
bench_perf_moe_fused_tp2.pbs
bench_perf_moe_fused_tp4.pbs
bench_perf_moe_fp8.pbs
bench_perf_graphs.pbs
bench_perf_tp12.pbs
bench_perf_tp2.pbs
bench_perf_tp4.pbs
bench_perf_ctx131k.pbs
bench_perf_halfprec.pbs
download_gptoss_bf16_fp16.sh

# llama.cpp SYCL (root PBS / helpers)
build_llamacpp_sycl.pbs
build_llamacpp_sycl_mxfp4_reorder.pbs
convert_gptoss_mxfp4_gguf.pbs
download_gptoss_mxfp4_gguf.pbs
download_gptoss_mxfp4_gguf.sh
smoke_llamacpp_sycl.pbs
bench_llamacpp_sycl.pbs
bench_llamacpp_sycl_perf.pbs
bench_llamacpp_sycl_phaseG.pbs

# training
train_lora_smoke.pbs
lora_one_epoch.py

# pins / recipes / patches / closure docs
build-vllm-xpu/pins.env
build-vllm-xpu/VERSIONS.md
build-vllm-xpu/PHASE_STATUS.md
build-vllm-xpu/SUCCESS_INFER.md
build-vllm-xpu/SUCCESS_TRAIN.md
build-vllm-xpu/SUCCESS_PERF.md
build-vllm-xpu/BEST_PRACTICE.md
build-vllm-xpu/PERF_PLAN.md
build-vllm-xpu/PERF.md
build-vllm-xpu/RESUME.md
build-vllm-xpu/perf-team/*.md
build-vllm-xpu/patches/

# llama.cpp SYCL campaign (docs + cycle envs + harness; no clone/build/logs)
build-llamacpp-sycl/*.md          # includes CYCLE_LOG.md, PLAN.md, BEST_RECIPE.md
build-llamacpp-sycl/*.sh
build-llamacpp-sycl/cycles/*.env
```

## Do not track

| Path | Why |
|------|-----|
| `models/` | HF weights (~100GB+) |
| `checkpoints/` | LoRA/adapters |
| `build-vllm-xpu/env/` | conda env |
| `build-vllm-xpu/xiaoliyan/` | wheels |
| `build-vllm-xpu/build-src/` | source trees |
| `build-vllm-xpu/frameworks-standalone/` | recipe clone |
| `build-vllm-xpu/logs/`, `*.out` | job output |
| `build-llamacpp-sycl/llama.cpp/` | nested clone |
| `build-llamacpp-sycl/build*/` | SYCL build trees |
| `build-llamacpp-sycl/logs/` | job output |
| `.cache/` | HF/torch caches |
| `diag_*`, `triton_diag.pbs` | one-off bring-up probes |

Model download stays a **script** (`download_model.sh`); weights stay on Lustre outside git.
