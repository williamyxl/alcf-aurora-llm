# High-concurrency (batched) inference — gpt-oss-120b MXFP4 on one Aurora node (PVC)

**Date:** 2026-08-16
**Question:** can we do high-concurrency inference on a single node? Tested vLLM (continuous batching)
and llama.cpp (batched-bench). **Yes — vLLM scales to ~3600 tok/s aggregate; llama.cpp to ~417.**
All outputs quality-OK.

## vLLM — continuous batching (frameworks 0.15, IPEX Marlin MXFP4)

Aggregate throughput = total output tokens / wall, sending `num_prompts` requests with `max_num_seqs`
concurrent slots. Harness: `vllm_concurrency.py` / `vllm_conc_fw.pbs`. max_tokens=128, prompt ~93 tok.

| TP | max_num_seqs | num_prompts | agg output tok/s | agg total tok/s | req/s | KV cache (tok) | quality | job |
|----|-------------:|------------:|-----------------:|----------------:|------:|----------------|:-------:|-----|
| 4  | 64  | 256  | 1031 | 1781 | 8.06 | 1.83M | 256/256 | 8759507 |
| 4  | 128 | 512  | 1561 | 2694 | 12.19 | 2.00M | 512/512 | 8759526 |
| 8  | 128 | 512  | 2049 | 3538 | 16.01 | 4.76M | 512/512 | 8759545 |
| **8** | **256** | **1024** | **3597** | **6211** | **28.10** | 4.58M | 1024/1024 | 8759557 |

- **Peak: TP=8, mns=256 → 3597 tok/s output (6211 total incl prefill), 28 req/s.** ~113× the
  single-stream decode (31.9 tok/s). Continuous batching + paged attention scale strongly on PVC.
- **For throughput, TP=8 > TP=4** (more compute + larger KV pool) — opposite of single-stream, where
  TP=4 was the decode sweet spot.
- Larger `max_num_seqs` keeps scaling; bounded by KV cache pool (raise `--kv-cache-memory-gib`).

## llama.cpp — batched-bench (2-tile pure GPU, `llama-batched-bench`)

`-npp 64 -ntg 128 -npl <B>`, `-sm layer -ts 0.5,0.5 -ngl 99 -fa on`. Harness: `llama_batched.pbs MODE=gpu2`.
S_TG = aggregate generation tok/s across B parallel sequences.

| B (parallel) | prefill S_PP tok/s | **gen S_TG tok/s** | total S tok/s | N_KV | job |
|-------------:|-------------------:|-------------------:|--------------:|------|-----|
| 64  | 592 | 194 | 250 | 12288 | 8759570 |
| 128 | 603 | 277 | 338 | 24576 | 8759575 |
| 256 | 606 | **417** | 466 | 49152 | 8759594 |

- llama.cpp batches (gen scales 194→277→417 as B grows) but far below vLLM. Its server/batched path is
  not a paged-attention continuous-batching engine, so aggregate decode saturates early.
- (MoE-on-CPU / F4_hbm mode is single-stream-oriented; CPU MoE bottlenecks under heavy concurrency, so
  pure-GPU 2-tile is the right llama.cpp batching config.)

## Head-to-head (aggregate generation tok/s, one node)

| Concurrency | vLLM (best TP) | llama.cpp gpu2 | vLLM advantage |
|-------------|---------------:|---------------:|---------------:|
| ~64  | 1031 (TP4) | 194 | 5.3× |
| ~128 | 2049 (TP8) | 277 | 7.4× |
| ~256 | **3597 (TP8)** | 417 | **8.6×** |

## Verdict

- **High-concurrency serving on one Aurora node: use vLLM (frameworks 0.15, TP=8, large `max_num_seqs`).**
  ~3600 tok/s aggregate output at 256-way concurrency, all quality-OK.
- **Single-stream / lowest latency / single-tile: use llama.cpp** (F4_hbm 41.6 tok/s decode).
- The engines are complementary: llama.cpp wins single-stream decode (41.6 vs 31.9); vLLM wins
  concurrency by ~5–9× and prefill throughput.

## Full-node deployment: use ALL tiles (the real "12 llama.cpp instances vs vLLM" question)

gpt-oss `num_attention_heads=64`, `num_key_value_heads=8` → valid vLLM TP ∈ {2,4,8} (**TP=12 invalid**;
KV heads cap clean sharding at 8). To use all **12 tiles**, vLLM needs **data parallelism**: N
independent engines on disjoint `ZE_AFFINITY_MASK` tile groups. Truest analogue of 12 single-tile
llama.cpp instances = **3× TP=4** (=12 tiles). Harness: `vllm_dp_node.pbs`.

### vLLM full-node (3× TP=4 = 12 tiles, summed aggregate output tok/s)
| per-replica mns | prompts/replica | node agg output tok/s | node req/s | quality | job |
|-----------------|-----------------|-----------------------|-----------|:-------:|-----|
| 64  | 256 | 2939 | 23.0 | 3/3 replicas OK | 8759967 |
| **128** | 512 | **4565** | 35.7 | 3/3 OK | 8760023 |

### llama.cpp full-node (N single-tile instances, each MoE→CPU + batched)
| instances | NP/inst | node agg gen tok/s | notes | job |
|-----------|---------|--------------------|-------|-----|
| 12 (mmap-shared weights) | 16 | **CRASH** | GPU segfault: mmap + CPU tensor-override incompatible with GPU offload | 8759985 |
| 6 (`--no-mmap`, 16 thr) | 16 | **109.9** | per-inst only 15–23 tok/s — CPU-MoE offload contends for CPU/HBM bandwidth; does NOT scale | 8760000 |

**The 12-instance MoE-offload plan does not scale.** Each instance's speed comes from the CPU handling
MoE experts; running 6–12 of them saturates shared CPU compute + memory bandwidth, so per-instance
throughput collapses (41.6 → ~18 tok/s) and 12× resident copies also blow host RAM (mmap-share crashes
the GPU path). Node aggregate (~110 tok/s at 6 inst) is **far below** even a single 2-tile pure-GPU
llama.cpp batched run (417 tok/s), and ~40× below vLLM full-node.

## Full-node head-to-head (gpt-oss-120b MXFP4, one node, aggregate gen tok/s)

| Deployment | tiles | agg gen tok/s |
|------------|:-----:|--------------:|
| **vLLM 3× TP=4, mns=128** | 12 | **4565** |
| vLLM 1× TP=8, mns=256 | 8 | 3597 |
| llama.cpp 2-tile pure-GPU batched, NP=256 | 2 | 417 |
| llama.cpp 6× single-tile MoE→CPU, NP=16 | 6 | 110 |
| 12× single-tile llama.cpp (hypothetical 12×41.6 single-stream) | 12 | ~500* |

\* *Optimistic upper bound assuming perfect scaling; measured 6-instance scaling shows it does NOT
hold — real 12-instance MoE-offload aggregate would be ~110–200 tok/s due to CPU/HBM contention.*

**Answer: yes, vLLM uses all 12 tiles (as 3× TP=4 data-parallel), and at full node it delivers ~4565
tok/s — ~9–40× the llama.cpp full-node options.** vLLM keeps MoE on-GPU (IPEX Marlin) so it scales
with tiles; llama.cpp's single-tile speed relies on CPU-MoE offload that does not replicate across
instances.

## Recipes
- vLLM single instance: `qsub -v MODEL=<mxfp4>,TP=8,MML=4096,MNS=256,NPROMPTS=1024,MAXTOK=128,KVGIB=40 vllm_conc_fw.pbs`
- **vLLM full node (12 tiles):** `qsub -v MODEL=<mxfp4>,NREP=3,TPE=4,MNS=128,NPROMPTS=512,KVGIB=20 vllm_dp_node.pbs`
- llama.cpp 2-tile GPU batched: `qsub -v MODE=gpu2,NP=256,CTX=65536 llama_batched.pbs`
- llama.cpp N MoE-offload instances: `qsub -v NINST=6,NP=16,THREADS=16 llama_12inst_moecpu.pbs`

## Notes
- vLLM `per_stream_out_tok_s` drops as concurrency rises (3.5 tok/s/stream at 256-way) — expected
  latency/throughput tradeoff; aggregate is what matters for serving.
- All vLLM runs `quality_ok_frac=1.0`. Harnesses: `vllm_concurrency.py`, `vllm_conc_fw.pbs`,
  `llama_batched.pbs`.
