# Nemotron-4-340B-Instruct (dense, 96 layers)

Dense, so there is no MoE to offload: every token costs full 340B FLOPs. Only the llama.cpp route gives usable output on Aurora.

| Engine | Recipe | Result | Usable |
|---|---|---|---|
| llama.cpp i1-Q4_K_M, 1 server on 6 tiles | `llamacpp/single_6tile.pbs` | 2.4 tok/s decode, TTFT 2.4 s, prefill 11 tok/s | yes |
| llama.cpp, 2 x 6 tiles data parallel | `llamacpp/dp2_6tile.pbs` | 4.4 / 6.0 / 6.6 tok/s at c=1/4/8 | yes |
| vLLM FP8-Dynamic, TP8 eager | `vllm/fp8_not_usable.pbs` | 10.0 tok/s single, 381 tok/s at 64-way | **no: output is garbage** |

An earlier llama.cpp run of 3 x 4 tiles reached ~9.9 tok/s aggregate (`llamacpp/MEASUREMENTS.md`); that layout was not rerun.
Single server with `--parallel 1` serialises requests (TTFT 87 s at 4 concurrent), so use `NPAR=8` for concurrency.

The vLLM FP8 run proves the scheduler and the dense Nemotron path work on XPU (and batches 64 sequences), but the FP8 kernels return nonsense text on PVC. XPU graph is not usable for this model: capture over 96 layers hits the 30-minute distributed-init timeout.

**vLLM BF16 was not tested: BF16 Nemotron-4-340B is ~1 TB, which does not fit one Aurora node's 768 GB HBM.** Downloading BF16 weights (~1 TB) would not help without multi-node vLLM, which is not set up in this repo. A BF16 or INT4 vLLM checkpoint was not tried.

Setup: `bash nemotron-4-340b/download_gguf.sh` (streams and assembles a multipart GGUF, ~196 GB; needs ~one part of spare space).
