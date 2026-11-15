# gpt-oss-120b (117B MoE, MXFP4)

The best-supported model on Aurora: runs on both engines, and is the only one that gets multi-thousand tok/s on one node.

## Recommended recipes

| Goal | Recipe | Result |
|---|---|---|
| Max node throughput (vLLM) | `vllm/dp_6xtp2.pbs` : 6 x TP2 data parallel | **11860 tok/s, 92.7 req/s** (128-way per engine, eager) |
| Lowest latency, many tiles (vLLM) | `vllm/single_graph.pbs` : 1 x TP4, XPU graph | **60.0 tok/s** decode, 0.03 s TTFT |
| Lowest latency, 1 tile (llama.cpp) | `llamacpp/f4hbm_single_tile.pbs` : MoE on CPU, HBM NUMA bind | **42.4 tok/s** decode |
| OpenAI API for users | `vllm/serve_api.pbs` : 6 x TP2 + load balancer | endpoint in `logs/ENDPOINT.txt` |

Setup: see the repo `README.md` (`common/env.sh`), then `bash gpt-oss-120b/download.sh` (login node).

## Measured on `frameworks/2026.1.0` (vLLM 0.26.1), 2026-11
| Config | Single-stream decode | Concurrency (agg) |
|---|---|---|
| MXFP4, 1 x TP4, eager | 26.3 tok/s (cold 32.0) | |
| MXFP4, 1 x TP4, **XPU graph** | **60.0** (cold 93.3), prefill 2887 tok/s | |
| MXFP4, 3 x TP4, eager | | 6849 tok/s, 53.5 req/s |
| MXFP4, **6 x TP2**, eager | | **11860 tok/s, 92.7 req/s** |
| BF16, TP8, eager (`vllm/bf16_tp8.pbs`) | 24.5 | 2459 tok/s, 19.2 req/s |

Earlier `frameworks/2025.3.1` (vLLM 0.15) reference: 31.9 tok/s single, 4565 tok/s / 35.7 req/s on 3 x TP4. That module needs `VLLM_XPU_FUSED_MOE_USE_REF=1`; see `vllm/serve_api_legacy_fw2025.pbs` and `vllm/MEASUREMENTS_fw2025.md`.
The 6 x TP2 and XPU-graph runs were measured separately; combining the graph with data parallel is untested.

## llama.cpp (MXFP4 GGUF, `LLAMA=gptoss` build)
| Config | Result |
|---|---|
| 1 tile, MoE on CPU, HBM bind ("F4_hbm") | **42.4 tok/s** decode, 48.6 tok/s prefill (41.6 on the earlier run: same engine) |
| 2 tiles pure GPU | ~30 tok/s decode, better prefill |
Details and the NUMA/tile layout reasoning: `llamacpp/MEASUREMENTS.md`.

## Notes
- Needs the harmony tiktoken vocab cached once for the API (`common/serve/download_tiktoken_cache.sh`).
- BF16 single-stream with greedy decoding and `ignore_eos` degenerated into repetition in one warm run; the concurrency run was clean. Treat BF16 single-stream quality as unchecked.
- `training/` holds a LoRA smoke test from July 2026 that no longer reproduces (see its README).
