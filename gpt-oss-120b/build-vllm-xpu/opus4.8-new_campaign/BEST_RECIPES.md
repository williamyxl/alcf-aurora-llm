# Best recipes — gpt-oss-120b MXFP4 on Aurora (PVC / Max 1550)

Two production recipes, chosen by workload. Both measured this campaign (2026-08-16), quality-OK.

| Workload | Engine / recipe | Throughput |
|----------|-----------------|-----------|
| **High concurrency / serving (full node)** | **vLLM 3× TP=4 data-parallel (12 tiles)** | **~4565 tok/s** aggregate gen, ~36 req/s |
| **Single user / lowest latency (1 tile)** | **llama.cpp F4_hbm (1-tile, MoE→CPU + HBM)** | **41.6 tok/s** decode (4096) / 38.1 (128K) |

---

## Recipe A — vLLM full-node batch inference (high concurrency)

**Use for:** serving many concurrent requests, maximum node throughput.
**Why:** vLLM continuous batching + IPEX Marlin MXFP4 MoE (on-GPU) scales with tiles. gpt-oss allows TP
∈ {2,4,8} only (64 heads / 8 KV heads; TP=12 invalid), so fill all 12 tiles as **3 data-parallel
engines of TP=4** on disjoint `ZE_AFFINITY_MASK` groups {0-3},{4-7},{8-11}.

**Stack:** Aurora `frameworks/2025.3.1` (vllm 0.15.0, torch 2.10, triton 3.6, ipex 2.10.10). Run under
the *full* `module load frameworks` env (sets SYCL/CCL the IPEX Marlin JIT needs).

**Launch** (`vllm_dp_node.pbs`):
```bash
qsub -q debug-scaling \
  -v MODEL=/lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/gpt-oss-120b/models/openai-gpt-oss-120b,\
NREP=3,TPE=4,MML=4096,MNS=128,NPROMPTS=512,MAXTOK=128,KVGIB=20,TAG=prod_dp \
  -o build-vllm-xpu/opus4.8-new_campaign/logs/prod_dp.out \
  build-vllm-xpu/opus4.8-new_campaign/vllm_dp_node.pbs
```
Key knobs: `NREP=3 TPE=4` (12 tiles); `MNS` (max_num_seqs) = concurrency per engine (128 best tested);
`KVGIB` sizes the KV pool (raise for more concurrency). Each engine loads its own model copy.

**Measured (3× TP=4, 12 tiles):**
| per-engine mns | node agg gen tok/s | node req/s |
|----------------|--------------------|-----------|
| 64  | 2939 | 23.0 |
| **128** | **4565** | 35.7 |

**Single-engine alternatives** (if not filling the node): `vllm_conc_fw.pbs` — 1×TP8 mns=256 → 3597
tok/s (8 tiles); 1×TP4 mns=128 → 1561 tok/s.

**For a real server** (vs the offline-batch harness): launch `vllm serve` per engine with the same
`module load frameworks` env, `-tp 4`, distinct `ZE_AFFINITY_MASK` and `--port`, then load-balance
across the 3 ports.

---

## Recipe B — llama.cpp single-tile with MoE offload (single user)

**Use for:** one interactive stream at the lowest latency / highest single-stream decode, using only
**one GPU tile** (leaves the other 11 tiles free for other jobs).
**Why:** F4_hbm keeps attention/dense on the GPU tile and offloads the 128 MoE experts to CPU, bound to
the socket's HBM NUMA node — fastest single-stream decode measured (41.6 tok/s), beating any pure-GPU
llama.cpp or vLLM single-stream config.

**Stack:** self-built llama.cpp SYCL (`build-llamacpp-sycl/build/bin`), `oneapi/release/2025.3.1`.

**Launch** (`llama_f4hbm_ctx.pbs`):
```bash
qsub -q debug-scaling -v CTX=4096 \
  build-vllm-xpu/opus4.8-new_campaign/llama_f4hbm_ctx.pbs      # or CTX=131072 for max context
```
Recipe internals: 1 tile (`ZE_AFFINITY_MASK=0`), `-ngl 99 -sm none`, `-ncmoe 99` (MoE experts→CPU),
`numactl --physcpubind=1-51,105-155 --membind=2` (socket-0 cores + HBM NUMA node 2), `-t 32`,
`-fa on`, `--no-mmap`.

**Measured:**
| context | TTFT (ms) | prefill tok/s | decode tok/s |
|---------|-----------|---------------|--------------|
| 4096 | 411 | 41.4 | **41.56** |
| 131072 (max) | 372 | 45.8 | **38.07** |

**Do NOT replicate this across all 12 tiles for throughput** — the speed comes from CPU-MoE offload;
6–12 instances saturate shared CPU/HBM bandwidth (per-instance drops to ~15–23 tok/s; node agg ~110
tok/s) and 12× resident copies exceed host RAM (mmap-sharing crashes the GPU path). For node throughput
use Recipe A. (See `CONCURRENCY_RESULTS.md`.)

---

## Why these two (not the alternatives)

- vLLM **cannot** do single-tile (no weight/MoE CPU offload on XPU; `cpu_offload_gb` uses a CUDA-only
  op — see DEBUG_LOG P19). Minimum vLLM config = TP=2.
- llama.cpp **does not** scale to high concurrency: its batched-bench (2-tile GPU) peaks ~417 tok/s at
  256-way, and MoE-offload instances don't replicate. vLLM full-node is ~9–40× higher.
- The self-built vLLM 0.27.1 stack does not run on PVC (attention crashes; DEBUG_LOG P11–P18). Use the
  frameworks module.

## Quick reference

| Metric | Recipe A (vLLM 3×TP4) | Recipe B (llama.cpp F4_hbm) |
|--------|-----------------------|-----------------------------|
| Tiles used | 12 (full node) | 1 |
| Best for | concurrency / serving | single-stream latency |
| Peak gen tok/s | ~4565 aggregate | 41.6 single-stream |
| Max context | 131072 (≈no penalty) | 131072 (~8% penalty) |
| Prefill tok/s | ~1670/engine | ~42–46 |
| Launcher | `vllm_dp_node.pbs` | `llama_f4hbm_ctx.pbs` |

Full data: `CONCURRENCY_RESULTS.md`, `VLLM_RESULTS.md`; timestamped log: `DEBUG_LOG.md`.
