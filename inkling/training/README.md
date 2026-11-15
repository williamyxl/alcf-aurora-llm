# Inkling LoRA/SFT (reference, NOT currently reproducible as-is)

A 4-node FSDP2 LoRA smoke (8 steps) passed on 2026-07-24 (`RESULT_2026-07.md`, plan in `FINETUNE_PLAN.md`). It needs the full BF16
checkpoint (`thinkingmachines/Inkling`, ~1.9 TB, not downloaded) and the same self-built Torch+IPEX+peft env described in
`../../gpt-oss-120b/training/README.md`, which no longer exists. Adapter export under FSDP was still flaky when the work stopped.
