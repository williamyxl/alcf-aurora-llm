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

# training
train_lora_smoke.pbs
lora_one_epoch.py

# pins / recipes / patches / closure docs
build-vllm-xpu/pins.env
build-vllm-xpu/VERSIONS.md
build-vllm-xpu/PHASE_STATUS.md
build-vllm-xpu/SUCCESS_INFER.md
build-vllm-xpu/SUCCESS_TRAIN.md
build-vllm-xpu/PERF_PLAN.md
build-vllm-xpu/patches/
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
| `.cache/` | HF/torch caches |
| `diag_*`, `triton_diag.pbs` | one-off bring-up probes |

Model download stays a **script** (`download_model.sh`); weights stay on Lustre outside git.
