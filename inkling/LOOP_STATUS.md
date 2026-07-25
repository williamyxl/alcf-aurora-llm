# Inkling FT loop status

**Updated:** 2026-07-25T09:36Z (wake tick #175) — **DONE**

| Phase | Status |
|-------|--------|
| 0–1 | **PASS** |
| 2 LoRA FSDP | **PASS** — job **`8701235`**: load 16s, train 8 steps / ~30s, `SUCCESS_TRAIN.md` written. Wake loop stopped. |
| `SUCCESS_TRAIN.md` | **present** |

## Metrics
- mode=fsdp, world=48, smoke_layers=2, cpu_offload + force_sum_reduce
- n_loaded=4 (q/v), train_loss≈12.21, load_s≈16, train_s≈30
- adapter: `checkpoints/lora-smoke/adapter/`
