# Inkling campaign + FT status

Updated: 2026-07-25T09:36Z

## Infer campaign (llama.cpp SYCL)
**COMPLETE** — BEST_RECIPE frozen. Max-ctx multi-node FILL OOM; no further C_PG* FILL re-runs unless recipe changes.

## Finetune (HF LoRA smoke)
See root [`../LOOP_STATUS.md`](../LOOP_STATUS.md) and [`../FINETUNE_PLAN.md`](../FINETUNE_PLAN.md).

| Phase | Status |
|-------|--------|
| 0 HF download | **PASS** (~1.9 TB) |
| 1 XPU LoRA probe | **PASS** |
| 2 FSDP2 LoRA smoke | **PASS** (job `8701235`) |

Wake loop stopped after `SUCCESS_TRAIN.md`.
