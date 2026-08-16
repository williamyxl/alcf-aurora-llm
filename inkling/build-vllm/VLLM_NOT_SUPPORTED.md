# Inkling on vLLM-XPU — NOT SUPPORTED (2026-08-16)

Inkling cannot run on vLLM on Aurora PVC. Two independent blockers:

1. **No model class in the runnable stack.** Inkling arch = `InklingForConditionalGeneration`
   (`model_type: inkling_mm_model`, multimodal; custom ops: shortconv_kernel, d_rel/rel_extent,
   log_scaling, sliding-window patterns). The Aurora `frameworks/2025.3.1` vLLM **0.15.0** has NO
   Inkling class (searched registry + whole package). Upstream vLLM `main` DOES add Inkling, but that
   needs a newer vLLM than any XPU-working build here.
2. **Newer vLLM doesn't run on PVC.** The self-built vllm 0.27.1 / torch 2.13 stack crashes in attention
   on PVC (triton 3.7.2 get_native SIGSEGV; SYCL FA2 xe2-only — see gpt-oss DEBUG_LOG P11-P18). And
   PVC has no native FP4 (NVFP4 checkpoint wouldn't help). Inkling's custom kernels also lack XPU impls.

**Conclusion:** use **llama.cpp SYCL (PR #25731 build, no RPC)** for Inkling on Aurora. See
`../build-llamacpp-sycl/`. vLLM path is blocked until a newer vLLM-XPU with Inkling support exists.
