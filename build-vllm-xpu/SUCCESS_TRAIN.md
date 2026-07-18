# SUCCESS_TRAIN — Phase 6 PASS

**Date:** 2026-07-18T02:11:09+00:00  
**Job:** `8680277.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`  
**Host:** `x4600c0s2b0n0`  
**Log:** `build-vllm-xpu/logs/train_lora.out` (run starting `host=x4600c0s2b0n0`)

## Verdict

One full LoRA/SFT epoch (`epochs=1`) completed on XPU for local `models/openai-gpt-oss-120b` using the self-built Torch-XPU + IPEX env (no vLLM, no frameworks). Adapter written under workdir.

## Prerequisites

- Phase 5 CLOSED: `build-vllm-xpu/SUCCESS_INFER.md` present
- Packages in `$ENV`: peft 0.19.1, trl 1.8.0, datasets 5.0.0, accelerate 1.14.0, ipex 2.10.10 (torch unchanged at 2.10.0a0+git449b176)

## Recipe

| Setting | Value |
|---------|--------|
| Script | `lora_one_epoch.py` / `train_lora_smoke.pbs` |
| Device | XPU (`xpu_count=12`), `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` |
| Load | `Mxfp4Config(dequantize=True)` → bf16 (MXFP4 not trainable as-is) |
| LoRA | `r=8`, targets `q_proj`,`v_proj` (~3.0M trainable / 116.8B) |
| Data | 8 short Q/A samples; `num_train_epochs=1` → 8 steps |
| TRL | `loss_type="nll"` (default `chunked_nll` crashes on `device_map` partial forwards) |
| Other | `device_map="auto"`, gradient checkpointing, bf16 |

## TRAIN_JSON

```json
{"ok":true,"device":"xpu","epochs":1,"n_samples":8,"n_steps":8,"train_loss":6.931554317474365,"train_s":20.3895333110122,"load_s":160.69036316301208,"xpu_count":12,"adapter_path":"/lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b/checkpoints/lora-smoke/adapter","target_modules":["q_proj","v_proj"],"lora_r":8}
```

## Artifacts

- Adapter: `checkpoints/lora-smoke/adapter/`
- Versions: `VERSIONS.md`
- Inference gate: `SUCCESS_INFER.md`
- Project README: `../README.md`
- Phase log: `PHASE_STATUS.md`
