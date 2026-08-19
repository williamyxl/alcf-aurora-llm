# Best recipes — Nemotron-4-340B-Instruct on Aurora (PVC / Max 1550)

Two production recipes for **NVIDIA Nemotron-4-340B-Instruct** (`NemotronForCausalLM`, `model_type:
nemotron` — a **DENSE** 340B transformer, 96 layers, hidden 18432, 96 heads / 8 KV heads; NO MoE), via
**llama.cpp SYCL**. Measured 2026-08-19 on the `debug` queue, quant **i1-Q4_K_M** (~196 GB GGUF).

| Workload | Recipe | Throughput |
|----------|--------|-----------|
| **High throughput / concurrency (full node)** | **3× `llama-server` full-XPU, 4 tiles each (data parallel) + LB** | **~9.9 tok/s** agg gen, **0.077 req/s** |
| **Single user / lowest latency** | **1× `llama-server` full-XPU, 6 tiles** | **TTFT ~2.4 s, prefill ~11 tok/s, decode ~2.3 tok/s** |

**Dense model — key difference from MoE Nemotrons:** there is **no MoE**, so no `-ncmoe` CPU-expert
offload. All weights must be on GPU (full-XPU) or host RAM. A single tile cannot hold 340B, so the
single-serve recipe uses the **fewest tiles that fit** (6), not 1. Decode is ~3× slower than the
55B-active MoE Nemotron-3-Ultra because a dense 340B does full-model FLOPs every token.

Engine: self-built llama.cpp SYCL `../inkling/build-llamacpp-sycl/build/bin` (b10383) — supports the
`nemotron` arch (`nemotron.cpp`). Module `oneapi/release/2025.3.1`. Env: `ZE_FLAT_DEVICE_HIERARCHY=FLAT`,
`ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `GGML_SYCL_ENABLE_VMM=0`, `--no-mmap`.

Checkpoint: `models/gguf-n4/nemotron4-340b-i1-Q4_K_M.gguf` (assembled from mradermacher multipart GGUF;
stored on MatSciAI — note the IQC project quota is 1 MB and unusable). Download:
`download_nemotron4_gguf.sh` (streaming download+assemble, xet disabled).

HBM budget (~64 GB/tile): i1-Q4_K_M ~210 GB → ≥4 tiles (~52 GB/tile, tight but works); 6 tiles ~35, 8
tiles ~26. All three DP/single configs below loaded fine.

---

## Recipe A — High throughput: 3× full-XPU data-parallel (full node) ⭐

**Use for:** concurrency / max node throughput. 3 independent `llama-server` instances, 4 disjoint tiles
each ({0-3},{4-7},{8-11}), each full-XPU (`-ngl 99 --split-mode layer`), fronted by `serve/lb.py` →
one `/v1`. Each instance is self-contained in HBM (dense, no shared bottleneck) → node aggregate ≈ 3×.

**Launch / bench** (`serve-llamacpp-n4/dp_bench_debug.pbs`):
```bash
cd /lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/nemotron
qsub -q debug -v NREP=3,TPE=4,QUANT=i1-Q4_K_M,CTX=2048,NPAR=8,MAXTOK=128,LEVELS="1 4 8" \
  serve-llamacpp-n4/dp_bench_debug.pbs
```

**Measured (3× 4-tile full-XPU, i1-Q4_K_M, job 8765549):**
| per-instance concurrency | node agg gen tok/s | node req/s | mean TTFT p50 |
|--------------------------|-------------------:|-----------:|--------------:|
| 1 (×3) | 6.6 | 0.051 | 2.83 s |
| 4 (×3) | 9.1 | 0.072 | 7.88 s |
| **8 (×3)** | **9.9** | **0.077** | 10.6 s |

Alternative **2× 6-tile** (more HBM headroom, job 8765521): 4.4 / 6.0 / **6.6** tok/s agg at c=1/4/8.

**Serve as one endpoint:** launch 3 instances (ports 8001/8002/8003), then
`serve/lb.py --host 0.0.0.0 --port 8000 --backends http://127.0.0.1:8001 ...:8002 ...:8003`.

**Caveats:** 3×4 is ~52 GB/tile — keep per-instance `-c`/`--parallel` modest (KV + compute must fit
alongside weights). If a load OOMs, drop to `NREP=2,TPE=6`.

---

## Recipe B — Single user, lowest latency: 1× full-XPU, 6 tiles

**Use for:** one interactive stream, lowest TTFT. Full-XPU on 6 tiles (`-ngl 99 --split-mode layer`),
`numactl --interleave=all`, `-fa on`, `--no-mmap`. (6 tiles = safe fit at ~35 GB/tile; the other 6 tiles
stay free.)

**Launch (server + bench)** (`serve-llamacpp-n4/single_bench_debug.pbs`):
```bash
qsub -q debug -v QUANT=i1-Q4_K_M,NTILES=6,CTX=8192,NPAR=1,MAXTOK=128,LEVELS="1" \
  serve-llamacpp-n4/single_bench_debug.pbs   # writes serve-llamacpp-n4/ENDPOINT.txt
```

**Measured (1× 6-tile full-XPU, i1-Q4_K_M, single stream, job 8765501):**
| metric | value |
|--------|------:|
| TTFT (p50) | **2.39 s** |
| prefill tok/s | **11.0** |
| decode tok/s | **2.3** |

Single instance does not scale with concurrency (layer split = pipeline; agg stays ~2–3 tok/s). For
concurrency use Recipe A.

---

## Why these two — see the sweep

| Config (i1-Q4_K_M) | node agg gen tok/s | verdict |
|--------------------|-------------------:|---------|
| **3× 4-tile full-XPU (DP)** | **9.9** | ⭐ best throughput (Recipe A) |
| 2× 6-tile full-XPU (DP) | 6.6 | throughput, more HBM headroom |
| 1× 6-tile full-XPU | 3.3 (c=8) / TTFT 2.4 s | best single-stream latency (Recipe B) |

Data parallelism scales ~linearly with instances (2×→6.6, 3×→9.9) because the dense model is
self-contained per instance in HBM — same DP pattern that worked for Nemotron-3-Ultra and gpt-oss.

## Quick reference

| Metric | Recipe A (3× 4-tile DP) | Recipe B (1× 6-tile) |
|--------|-------------------------|----------------------|
| Tiles used | 12 (full node) | 6 |
| Best for | concurrency / throughput | single-stream latency |
| Node agg gen tok/s | ~9.9 | ~3.3 (single ~2.3 decode) |
| Node req/s | ~0.077 | ~0.018 |
| TTFT | ~2.8–10.6 s | ~2.4 s |
| Quant | i1-Q4_K_M (196 GB) | i1-Q4_K_M |
| Launcher | `serve-llamacpp-n4/dp_bench_debug.pbs` | `serve-llamacpp-n4/single_bench_debug.pbs` |

Notes: a smaller quant (e.g. i1-IQ3_M ~155 GB, i1-IQ2_M ~117 GB) loads on fewer tiles / faster and would
raise tokens/s at some quality cost. Long-term hosting: raise walltime + use the `capacity` queue,
keep instances running, front DP instances with `serve/lb.py`. Test only on `debug`/`debug-scaling`.
