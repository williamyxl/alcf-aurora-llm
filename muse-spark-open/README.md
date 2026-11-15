# Muse Spark Open (Meta, 816B-A42B, 62 layers, multimodal)

**Status: not servable on Aurora today.** Checkpoint downloaded (BF16, 1.5 TB, `download.sh`, ungated unsloth mirror of `meta-models/`); nothing here produces output.

1. Stock module: unknown `model_type: muse_spark`.
2. `OVERLAY_TF=1` makes vLLM resolve it (`TransformersMultiModalMoEForCausalLM`).
3. A 4-layer dummy-weight build still runs out of XPU memory on one node (202k vocab, hidden 8192, vision and audio towers). Probe: `vllm/arch_probe_dummy.pbs`.
4. BF16 is 1.5 TB: it needs about three nodes. The NVFP4 (533 GB) and MXFP4 (509 GB) variants would fit one node but PVC has no FP4 path.
5. llama.cpp has only the unrelated 30B `muse-glimmer`.
