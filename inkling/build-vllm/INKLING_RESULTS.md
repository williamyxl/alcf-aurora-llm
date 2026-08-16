# Inkling (UD-IQ1_S) on Aurora PVC — one-node results & best recipes

**Date:** 2026-08-16
**Engine:** llama.cpp SYCL (PR #25731 inkling arch, no RPC). **vLLM is NOT usable** for Inkling — see
`VLLM_NOT_SUPPORTED.md` (no model class in the runnable frameworks-0.15 vLLM; PVC has no FP4).
**Model:** Thinking Machines Inkling, Unsloth `UD-IQ1_S` GGUF (7 shards, 252 GB). Quality-OK.

## Best recipes (one Aurora node = 12 PVC tiles)

| Workload | Recipe | Performance |
|----------|--------|-------------|
| **Single user (lowest latency)** | **pure-GPU 12-tile** (`-sm layer` even split, `-fa on`, `--no-mmap`) | **6.63 tok/s** decode, prefill 6.70, TTFT 2.5 s |
| **High concurrency** | **pure-GPU 12-tile batched** (`llama-batched-bench`) | **~16.7 tok/s** aggregate gen (saturates) |

## Measured

### Single-stream decode
| Recipe | tiles | decode tok/s | prefill tok/s | TTFT (ms) | load (s) | job |
|--------|:-----:|-------------:|--------------:|-----------|----------|-----|
| **PG12 pure-GPU** | 12 | **6.63** | 6.70 | 2538 | 116 | 8760348 |
| PG10 pure-GPU | 10 | 6.46 | 7.06 | 2407 | 160 | (earlier) |
| MO1 1-tile MoE→CPU (`--preferred=2`) | 1 | 6.11 | 10.47 | 1624 | 283 | (earlier) |

Pure-GPU 12-tile is the best single-stream decode. MoE→CPU offload gives *higher prefill* (10.5) but
lower decode; 1-tile `--membind=2` hard-bind OOMs on the 252 GB `--no-mmap` load (use `--preferred=2`).

### High-concurrency (batched-bench, 12-tile pure-GPU, -npp 64 -ntg 128)
| B (parallel) | prefill S_PP tok/s | **gen S_TG tok/s** | total S tok/s | job |
|-------------:|-------------------:|-------------------:|--------------:|-----|
| 64  | 43.2 | 16.64 | 20.9 | 8760356 |
| 128 | 43.8 | **16.76** | 21.1 | 8760375 |

**Batching does NOT scale for Inkling** on llama.cpp: gen throughput saturates at ~16.7 tok/s (64-way ≈
128-way). The IQ1_S + custom inkling kernels don't benefit from higher concurrency like a fused-MoE
model does. Peak node aggregate ≈ **16.7 tok/s**.

## Context
- Inkling is a large 1-bit-ish quant (IQ1_S) with a custom architecture (shortconv, rel-position,
  sliding-window). Only llama.cpp (PR #25731) supports it on Aurora.
- vs gpt-oss-120b MXFP4 on the same node: gpt-oss does 41.6 tok/s single-stream (llama.cpp F4_hbm) and
  ~4565 tok/s concurrent (vLLM). Inkling is far slower — inherent to the model size/quant/arch and the
  lack of a scalable batched engine.

## Recipes / harness
- Runner: `inkling_llama_node.pbs` (`-v MODE=single|conc SUB=pg12|mo_hbm TILES=12 NP=<n> CTX=<n>`).
- Single user: `qsub -v MODE=single,SUB=pg12,TILES=12,CTX=4096 inkling_llama_node.pbs`
- Concurrency: `qsub -v MODE=conc,SUB=pg12,TILES=12,NP=64,CTX=16384 inkling_llama_node.pbs`
- Model: `models/unsloth-Inkling-GGUF/UD-IQ1_S/inkling-UD-IQ1_S-00001-of-00007.gguf`
- Build: `build-llamacpp-sycl/build/bin` (PR #25731). Logs: `build-vllm/logs/`.

## Verdict
For Inkling on Aurora, use **llama.cpp SYCL pure-GPU 12-tile**: ~6.6 tok/s single-user, ~16.7 tok/s
max concurrent. vLLM is not an option until a newer vLLM-XPU adds the Inkling architecture.
