# Inkling LoRA smoke

```json
{
  "ok": true,
  "epochs": 1,
  "n_samples": 8,
  "n_steps": 8,
  "train_loss": 12.2063627243042,
  "train_s": 29.756149969995022,
  "load_s": 16.17909306299407,
  "world_size": 48,
  "mode": "fsdp",
  "n_loaded": 4,
  "adapter_path": "/lus/flare/projects/MOFA/xiaoliyan/workdir/llm/inkling/checkpoints/lora-smoke/adapter",
  "target_modules": [
    "q_proj",
    "v_proj"
  ],
  "lora_r": 8,
  "loader": "meta+fsdp2+param_broadcast"
}
```
