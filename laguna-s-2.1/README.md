# Laguna S 2.1 (Poolside, 48 layers, 256 experts top-10, sliding window 512)

Second model, after gpt-oss-120b, that serves at high concurrency on Aurora with vLLM. BF16 only; vLLM only (no llama.cpp run attempted, although the `inkling` llama.cpp build has a `laguna` architecture).

## Recommended recipe
`vllm/serve_conc_graph.pbs` : BF16, one engine on 8 tiles (TP8), **XPU graph**.

| Config | Single-stream decode | Concurrency at 128-way | Notes |
|---|---|---|---|
| BF16 TP8, **XPU graph** | **36.6 tok/s** (prefill 1069 tok/s) | **2342 tok/s, 18.3 req/s** | answers coherent, 512/512 requests OK |
| BF16 TP8, eager | 13.6 tok/s | 1352 tok/s, 10.6 req/s | `vllm/serve_conc_eager.pbs` |
| FP8 TP8, eager | 2.84 tok/s | 129.5 tok/s | works but ~5x slower than BF16; not recommended |

Model load takes ~13 min (219 GB read from Lustre). A 6k-token prompt prefilled at 5135 tok/s (eager run). OpenAI endpoint: `vllm/serve_api.pbs`.

## Eager recipe: verified working again after the triton-shadowing fix
See `vllm/RUN_LAGUNA.md` (re-verified 2026-11-20/21: 14.3 tok/s single stream, 1350 tok/s at 128-way). If a vLLM job fails with `0 active drivers` or `Cannot find a working triton installation`, a pip-installed triton in `~/.local/aurora/frameworks/*` is shadowing the framework one; `common/env.sh` now sets `PYTHONNOUSERSITE=1` to avoid that. Details: `vllm/ROOT_CAUSE_triton_shadowing.md`.

## Setup
`bash laguna-s-2.1/download.sh` (login node, 219 GB), then `qsub -v REPO=$PWD laguna-s-2.1/vllm/serve_conc_graph.pbs`. No overlays needed.

## Not done
- FP8 with the XPU graph failed to construct (`laguna-s-2.1/vllm/fp8_not_recommended.pbs` is the eager variant); the FP8 MoE path is the reason to avoid FP8 on this hardware.
- TP4 / data-parallel layouts: the 219 GB BF16 weights leave little room at TP4 (55 GB/tile of 64); untested.
