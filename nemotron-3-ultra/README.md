# Nemotron-3-Ultra-550B-A55B (hybrid Mamba2 + MoE, 512 experts top-22)

Runs on both engines, but neither is fast. Pick by goal.

| Engine | Recipe | Result |
|---|---|---|
| vLLM, INT4 w4a16 | `vllm/serve_int4_patched.pbs` (needs `OVERLAY_KERNELS=1`) | **7.8 tok/s** single, **115 tok/s at 16-way**, quality OK |
| llama.cpp, UD-IQ2_M, 1 tile (experts on CPU) | `llamacpp/single_tile_moecpu.pbs` | 7.1 tok/s decode, TTFT 2.7 s; aggregate flat ~6-7 tok/s c=1..32 |
| llama.cpp, 3 x 4 tiles full-XPU | `llamacpp/dp3_fullxpu.pbs` | 12.9 / 13.9 / **15.7 tok/s** at c=1/4/8 |

## vLLM: why the patched kernel
Ultra routes each token to 22 of 512 experts. Stock `vllm_xpu_kernels` (0.1.11.x) only has top-k dispatch for {1,2,3,4,6,7,8,10} and raises `RuntimeError: Unsupported TopK value`. `common/vllm/kernels/topk22.patch` adds 22 in two tables; `common/vllm/setup_overlay.sh kernels` rebuilds `_moe_C` (about ten minutes, compute node) and `OVERLAY_KERNELS=1` loads it. With it the model loads (INT4, hybrid Mamba2 + MoE) and generates correctly.

Limits found: the recipe uses `--max-num-seqs 16 --max-model-len 2048`; a larger batch ran out of XPU resources (`UR_RESULT_ERROR_OUT_OF_RESOURCES`) on one node. Weight loading takes ~33 min for 293 GB, so the run needs the full 59-min debug walltime. Eager only (XPU graph untested here). The top-k=22 path is functionally verified, not tuned; the 115 tok/s figure is at 16-way and will not match gpt-oss.

## Setup
```
bash nemotron-3-ultra/download_hf_w4a16.sh          # vLLM INT4, 293 GB
bash nemotron-3-ultra/download_gguf.sh               # llama.cpp, QUANT=UD-IQ2_M default
qsub -v REPO=$PWD common/vllm/setup_overlay.pbs      # builds the top-k=22 kernel once
```
llama.cpp details and the CPU-bound explanation: `llamacpp/MEASUREMENTS.md`, `llamacpp/FINDINGS.md`.
