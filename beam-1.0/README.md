# Beam 1.0 (Reflection, 501B-A23B, 52 layers, 256 experts top-8, sliding window 512)

**Status: not servable on Aurora today.** Checkpoint downloaded (BF16, 934 GB, `download.sh`); nothing here produces output.

What works and what stops it, in order:
1. Stock `frameworks/2026.1.0` does not know `model_type: beam` (its transformers is 5.14; Beam landed in 5.22).
2. `OVERLAY_TF=1` (transformers 5.23 + rope shim) fixes that: vLLM resolves the config and the class (`TransformersMoEForCausalLM`).
3. Model construction in the worker still fails: Beam's full-attention layers have `rope_parameters: null` (no RoPE), and a fourth site in the vLLM Transformers backend indexes into it. The shim patches three sites; this one is not patched. Probe: `vllm/arch_probe_dummy.pbs` (6 layers, dummy weights).
4. Even if that is fixed, BF16 (934 GB) exceeds one node's 768 GB HBM. FP8 (503 GB) would fit but FP8 is unreliable on PVC.
5. vLLM `main` and llama.cpp `master` have added Beam (checked 2026-11-15); neither is in the installed module or the local llama.cpp builds.
