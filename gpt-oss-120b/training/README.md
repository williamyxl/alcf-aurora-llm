# gpt-oss-120b LoRA/SFT smoke (reference, NOT currently reproducible as-is)

One LoRA epoch ran on XPU on 2026-07-18 (`RESULT_2026-07.md`): MXFP4 -> bf16 dequantize, `peft` r=8 on q/v, 8 steps.

It used a **self-built Torch 2.10 + IPEX 2.10 + peft/trl** stack (`ENV_VERSIONS_2026-07.md`, `pins.env`), not an Aurora module. That env was
removed during later work, and the current `frameworks/2026.1.0` module (torch 2.13) ships `accelerate` and `datasets` but not `peft`,
`trl` or IPEX. The scripts here are kept as the record of what worked; to re-run, rebuild that env or `pip install peft trl` on top of
the 2026.1.0 module and expect to adapt `lora_one_epoch.py` (IPEX calls) -- untested.
