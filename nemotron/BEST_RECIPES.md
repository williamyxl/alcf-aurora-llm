# Best recipes — Nemotron-3-Ultra-550B-A55B on Aurora (PVC / Max 1550)

Two production recipes for **NVIDIA Nemotron-3-Ultra** (`nemotron_h_moe`: hybrid Mamba2 + attention +
non-gated relu² MoE), chosen by workload. Both use **llama.cpp SYCL** — vLLM 0.15 on XPU cannot run this
model (non-gated MoE has no XPU kernel; see `FINDINGS.md`). All numbers measured this campaign
(2026-08-19), quality-OK, on the `debug` queue, UD-IQ2_M GGUF unless noted.

| Workload | Recipe | Throughput |
|----------|--------|-----------|
| **High throughput / concurrency (full node)** | **3× `llama-server` full-XPU, 4 tiles each (data parallel) + LB** | **~15.6 tok/s** agg gen, **0.13 req/s** |
| **Single user / lowest latency (1 tile)** | **1× `llama-server`, 1 tile, MoE experts → CPU** | **TTFT ~2.7 s, ~7 tok/s decode** |

Engine (both): self-built llama.cpp SYCL at `../inkling/build-llamacpp-sycl/build/bin` (b10383, ggml
0.19), which already implements `nemotron_h_moe`. No rebuild needed. Module: `oneapi/release/2025.3.1`.
Env (both): `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`,
`GGML_SYCL_ENABLE_VMM=0` (required on PVC), `--no-mmap` (sequential load off Lustre).

---

## Recipe A — High throughput: 3× full-XPU data-parallel (full node) ⭐

**Use for:** many concurrent requests, maximum node throughput.
**Why:** each instance is **self-contained in HBM** (experts on GPU, no shared CPU/DDR bottleneck), so 3
independent instances on disjoint tiles run truly in parallel and the node aggregate ≈ 3× one instance.
This is the only config that scales node throughput (single-instance configs are CPU-bound or pipeline-
serial — see `FINDINGS.md`). Mirrors the gpt-oss "3× TP=4 across 12 tiles" data-parallel pattern.

**Layout:** 3 `llama-server` instances × 4 tiles = 12 tiles, disjoint `ZE_AFFINITY_MASK`
{0-3},{4-7},{8-11}, each `-ngl 99 --split-mode layer` and **no `-ncmoe`** (experts on GPU). UD-IQ2_M is
~48.5 GB/tile at 4 tiles → fits ~64 GB HBM. Front the 3 ports with `serve/lb.py` (round-robin) → one `/v1`.

**Launch / bench** (`serve-llamacpp/dp3_bench_debug.pbs`):
```bash
cd /lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/nemotron
qsub -q debug -v NREP=3,TPE=4,MODE=fullxpu,QUANT=UD-IQ2_M,CTX=4096,NPAR=8,MAXTOK=128,LEVELS="1 4 8" \
  serve-llamacpp/dp3_bench_debug.pbs
```

**Measured (3× 4-tile full-XPU, UD-IQ2_M, job 8765318):**
| per-instance concurrency | node agg gen tok/s | node req/s | mean TTFT p50 |
|--------------------------|-------------------:|-----------:|--------------:|
| 1 (×3) | 12.0 | 0.095 | 6.05 s |
| 4 (×3) | 13.6 | 0.123 | 13.4 s |
| **8 (×3)** | **15.6** | **0.131** | 12.5 s |

**Serving it as one endpoint:** launch the 3 instances (ports 8001/8002/8003), then
`serve/lb.py --host 0.0.0.0 --port 8000 --backends http://127.0.0.1:8001 ...:8002 ...:8003` → one `/v1`.

**Caveats:**
- Use **UD-IQ2_M** (fits full-XPU at 4 tiles). **MXFP4_MOE (88 GB/tile at 4 tiles) does NOT fit** full-
  XPU; for MXFP4 you'd need MoE→CPU per instance, which reintroduces shared-CPU contention (limited DP gain).
- Keep per-instance `-c`/`--parallel` modest so KV + compute fit alongside the ~48.5 GB/tile weights.
- DP raises **aggregate** node throughput / req-s, not per-request latency (each instance ~4–5 tok/s).

---

## Recipe B — Single user, lowest latency: 1 tile, MoE → CPU

**Use for:** one interactive stream at the lowest latency, using only **one GPU tile** (leaves the other
11 free for other jobs). Best single-stream TTFT and decode measured.
**Why:** keep dense/attention/Mamba on one tile and offload the MoE experts to CPU (`-ncmoe 99`); the
single stream isn't CPU-bandwidth-bound, so this is the fastest first-token + decode for one user.
(Directly analogous to the gpt-oss F4_hbm single-tile recipe.)

**Recipe:** 1 tile (`ZE_AFFINITY_MASK=0`), `-ngl 99 -sm none`, **`-ncmoe 99`** (experts→CPU), `-fa on`,
`-t 64`, `numactl --interleave=all`, `--no-mmap`.

**Launch (server)** (`serve-llamacpp/serve_test_debug.pbs`) or **smoke (cli)** (`llama_smoke.pbs`):
```bash
qsub -q debug serve-llamacpp/serve_test_debug.pbs        # OpenAI /v1 llama-server + self-smoke
qsub -q debug llama_smoke.pbs                             # llama-cli single-stream smoke
# tune: -v QUANT=UD-IQ2_M,CTX=4096,NPAR=1
```

**Measured (1 tile, MoE→CPU, UD-IQ2_M, single stream, job 8765192/8765130):**
| metric | value |
|--------|------:|
| TTFT (p50) | **2.67 s** |
| prefill tok/s | **11.9** |
| decode tok/s | **7.1** |

Do NOT expect this to scale to concurrency — a single instance's aggregate is flat ~6–7 tok/s (experts
on shared CPU DDR). For concurrency use Recipe A.

---

## Why these two (not the alternatives) — see `FINDINGS.md`

| Config | node agg gen (UD-IQ2_M) | verdict |
|--------|------------------------:|---------|
| **3× 4-tile full-XPU (DP)** | **15.6 tok/s** | ⭐ best throughput (Recipe A) |
| 1× 1-tile MoE→CPU | 6.9 tok/s | best single-stream latency (Recipe B) |
| 1× 8-tile full-XPU (layer split) | 5.8 tok/s | pipeline-serial, no gain — avoid |
| 1× tensor-parallel (`--split-mode row`) | — | **crashes** on `nemotron_h_moe` — unsupported |
| vLLM 0.15 XPU | — | **blocked**: non-gated relu² MoE has no XPU kernel |

## Quick reference

| Metric | Recipe A (3× full-XPU DP) | Recipe B (1-tile MoE→CPU) |
|--------|---------------------------|---------------------------|
| Tiles used | 12 (full node) | 1 |
| Best for | concurrency / throughput | single-stream latency |
| Node agg gen tok/s | ~15.6 | ~6.9 (single stream ~7 decode) |
| Node req/s | ~0.13 | ~0.06 |
| TTFT | ~6–13 s under load | ~2.7 s |
| Quant | UD-IQ2_M | UD-IQ2_M (or MXFP4_MOE for quality) |
| Launcher | `serve-llamacpp/dp3_bench_debug.pbs` | `serve-llamacpp/serve_test_debug.pbs` |

Full data & analysis: `FINDINGS.md`. Serving/hosting details: `serve-llamacpp/README.md`.
Long-term hosting: raise walltime and use the `capacity` queue (≤168 h); keep the server(s) running and
front DP instances with `serve/lb.py`.
