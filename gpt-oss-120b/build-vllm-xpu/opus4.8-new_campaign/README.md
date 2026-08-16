# opus4.8-new_campaign — gpt-oss-120b on Aurora PVC (vLLM + llama.cpp)

Campaign to diagnose why vLLM was ~25–35× slower than llama.cpp on Aurora Max 1550 (PVC), get a
working vLLM inference, and push decode >30 tok/s. **Outcome: achieved** — vLLM gpt-oss-120b MXFP4
now runs at **31.9 tok/s decode (TP=4)**, and llama.cpp F4_hbm hits **41.6 tok/s (1 tile)**.

## Read these first
| Doc | Contents |
|-----|----------|
| **`BEST_RECIPES.md`** | ⭐ The two production recipes: vLLM full-node (concurrency) + llama.cpp F4_hbm (single user). **Start here.** |
| **`VLLM_SERVING_RECIPE.md`** | Production hosting: vLLM high-concurrency service (capacity queue, week-long, `serve/`). |
| **`VLLM_RESULTS.md`** | Full results tables (vLLM TP sweep, context sweep, llama.cpp F4_hbm) + engine comparison. |
| **`VLLM_WORKING_RECIPE.md`** | The exact working vLLM stack + launch recipe. |
| **`DEBUG_LOG.md`** | Timestamped problem→solution log (P1–P21). Every fix and dead end. |
| `RESULTS.md` | llama.cpp best-recipe reruns (F4_hbm, P14_tp2, Inkling MO1/PG10). |
| `VLLM_INVESTIGATION.md` | Root-cause narrative for the vLLM slowness/crashes. |
| **`CONCURRENCY_RESULTS.md`** | High-concurrency (batched) throughput: vLLM vs llama.cpp on one node. |
| `PLAN.md`, `E9_parity_audit.md` | Original E1–E9 experiment design + parity audit. |
| `SESSION_RECOVERY.md` | Cross-session state/recovery notes. |

## Headline results (gpt-oss-120b MXFP4, PVC, single stream, decode tok/s, quality-OK)

| Engine / recipe | 4096 ctx | 131072 ctx | tiles | prefill tok/s |
|-----------------|---------:|-----------:|:-----:|--------------:|
| llama.cpp F4_hbm (1-tile, MoE→CPU, HBM) | **41.6** | **38.1** | 1 | ~42–46 |
| llama.cpp P14_tp2 (2-tile GPU) | 34.0 | — | 2 | 495 (pp512) |
| vLLM TP=4 (frameworks 0.15, IPEX Marlin) | **31.9** | 31.4 | 4 | ~1670 |
| vLLM TP=2 | 29.6 | 28.9 | 2 | ~1290 |
| vLLM old REF-MoE (historical) | ~1.2 | — | 2 | — |

## High-concurrency (batched serving, one node) — see `CONCURRENCY_RESULTS.md`

| Concurrency | vLLM agg gen tok/s (best TP) | llama.cpp agg gen tok/s (2-tile GPU) |
|-------------|----------------------------:|-------------------------------------:|
| 64-way  | 1031 | 194 |
| 128-way | 2049 | 277 |
| **256-way** | **3597** (TP=8) | 417 |

**vLLM continuous batching peaks at ~3597 tok/s aggregate (256-way, TP=8) — ~113× single-stream and
~8.6× llama.cpp.** For serving, use vLLM; for single-stream/single-tile, use llama.cpp.

## ⭐ Production recipes (see `BEST_RECIPES.md`)

| Workload | Recipe | Throughput | Launcher |
|----------|--------|-----------|----------|
| **High concurrency / serving** | **vLLM 3× TP=4 (12 tiles, full node)** | ~4565 tok/s agg, 36 req/s | `vllm_dp_node.pbs` |
| **Single user / low latency** | **llama.cpp F4_hbm (1 tile, MoE→CPU)** | 41.6 tok/s decode | `llama_f4hbm_ctx.pbs` |

## Working recipes (one-liners)

**vLLM (frameworks module, IPEX Marlin MXFP4):** `vllm_run_fw3.pbs`
```bash
qsub -q debug-scaling -v MODEL=<mxfp4-dir>,TP=4,MML=4096,MNS=1,MEMUTIL=0.85,KVGIB=8,TAG=fw_tp4 \
  -o logs/vllm_fw_tp4.out vllm_run_fw3.pbs
```
Requires full `module load frameworks` env (sets SYCL/CCL the IPEX Marlin JIT needs). No mpiexec.

**llama.cpp F4_hbm (1-tile MoE→CPU + HBM):** `llama_f4hbm_ctx.pbs`
```bash
qsub -q debug-scaling -v CTX=4096 llama_f4hbm_ctx.pbs      # or CTX=131072
```

## Key findings
1. **vLLM works via the upgraded frameworks module** (`frameworks/2025.3.1` = vllm 0.15 + torch 2.10 +
   triton 3.6 + ipex 2.10, IPEX Marlin MXFP4 MoE). The older `frameworks/2025.2.0` (vllm 0.10) had no
   gpt-oss path — hence earlier "frameworks can't run gpt-oss" reports.
2. **The self-built vllm 0.27.1 stack does NOT run on PVC** — attention crashes (triton 3.7.2
   `get_native` SIGSEGV; SYCL FA2 xe2-only). oneAPI 2026 fixed that stack's ABI but not its attention.
3. **PVC has no native FP4/FP8** (INT4/INT8 only); MXFP4 MoE works via IPEX Marlin (vLLM) or the
   llama.cpp SYCL GGUF kernels.
4. **vLLM XPU has no weight/MoE CPU-offload** (`cpu_offload_gb` uses a CUDA-only UVA op) — the
   llama.cpp F4_hbm single-tile recipe has no vLLM equivalent; min vLLM config is TP=2.

## Environments
- vLLM (working): Aurora `frameworks/2025.3.1` module env (do not rebuild).
- llama.cpp SYCL: `build-llamacpp-sycl/build/bin` (built this campaign).
- Downloads/tooling: `software/conda/envs/aurora-llm`.
- Self-built vLLM (non-working on PVC, kept for record): `software/conda/envs/vllm-xpu`.
