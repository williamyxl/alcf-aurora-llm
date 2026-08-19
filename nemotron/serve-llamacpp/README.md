# Hosting Nemotron-3-Ultra on Aurora with llama.cpp SYCL (OpenAI-compatible)

Serves **NVIDIA Nemotron-3-Ultra-550B-A55B** (`nemotron_h_moe`: hybrid Mamba2 + non-gated MoE) via the
self-built llama.cpp SYCL `llama-server`, on **one Aurora GPU tile** with MoE experts offloaded to CPU.
This is the working path (vLLM's XPU MoE kernels don't support this model — see `../FINDINGS.md`).

## Architecture (single tile)
```
client ──> llama-server :8000  (OpenAI /v1)   tile 0 (dense/attn/Mamba on GPU)
                                              MoE experts (-ncmoe 99) -> CPU DDR
```
`llama-server` exposes `/v1/chat/completions` and `/v1/models` natively — no load balancer needed for a
single instance.

## Engine
Self-built llama.cpp SYCL at `inkling/build-llamacpp-sycl/build/bin` (b10383, ggml 0.19) — already has
`nemotron_h_moe`. No rebuild required.

## 1. Download a GGUF quant
```bash
cd /lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/nemotron
bash download_nemotron_gguf.sh                 # UD-IQ2_M (~194 GB, fast; default)
QUANT=MXFP4_MOE bash download_nemotron_gguf.sh # native MoE quant (~352 GB, higher quality)
```

## 2. Validate on the debug queue (single node, <=1 h)
```bash
qsub -q debug serve-llamacpp/serve_test_debug.pbs
# tune: -v QUANT=MXFP4_MOE,CTX=8192,NPAR=4,PORT=8000
tail -f serve-llamacpp/logs/serve_test.out     # model load (min), then SMOKE curl output
```
When ready, `serve-llamacpp/ENDPOINT.txt` shows `url=http://<node_ip>:8000/v1`.

**Validated (2026-08-19, job 8765130, UD-IQ2_M):** `/v1/chat/completions` returns a correct MOF answer;
prefill ~10.3 tok/s, decode ~5.0 tok/s.

## Performance (job 8765192, UD-IQ2_M, max_tokens=128)
Single-stream is good; **throughput does not scale with concurrency** (CPU-MoE-offload bottleneck).

| Concurrency | TTFT p50 | per-req decode tok/s | agg gen tok/s |
|-------------|---------:|---------------------:|--------------:|
| 1  | 2.67 s | 7.1 | 6.2 |
| 8  | 16.2 s | 1.0 | 6.9 |
| 32 | 66.2 s | 0.19 | 5.3 |

→ A single instance is **single-user / low-latency** (aggregate ~6–7 tok/s; experts on shared CPU DDR).

## High-throughput on one node: 3 independent instances × 4 tiles (data parallel)
The way to raise node throughput is **data parallelism**: 3 full-XPU `llama-server` instances, each on 4
disjoint tiles ({0-3},{4-7},{8-11}), experts on GPU (no `-ncmoe`). Each instance is self-contained in
HBM (no shared-CPU bottleneck), so the node aggregate ≈ 3× one instance:

| Config | node agg gen tok/s | node req/s |
|--------|-------------------:|-----------:|
| 1× 1-tile MoE→CPU | 6.9 | 0.064 |
| 1× 8-tile full-XPU | 5.8 | 0.049 |
| **3× 4-tile full-XPU (DP)** | **15.6** | **0.131** |

Benchmark / launch the DP node:
```bash
qsub -q debug serve-llamacpp/dp3_bench_debug.pbs    # 3 instances, node-aggregate bench (DP_NODE_JSON)
# -v NREP=3,TPE=4,MODE=fullxpu,QUANT=UD-IQ2_M,CTX=4096,NPAR=8,MAXTOK=128,LEVELS="1 4 8"
```
For a real endpoint, front the 3 instance ports (8001/8002/8003) with `../serve/lb.py` (round-robin) to
expose one `/v1`. Use **UD-IQ2_M** (fits full-XPU at 4 tiles; MXFP4_MOE 88 GB/tile does not).

Single-instance bench (for latency): `serve-llamacpp/bench_debug.pbs`. Full analysis and the ranked
throughput options are in `../FINDINGS.md`.

## 3. Reach the service
Compute nodes have no public IP — reach via a login node (UAN).
```bash
NODE_IP=$(grep '^ip=' serve-llamacpp/ENDPOINT.txt | cut -d= -f2)
curl http://$NODE_IP:8000/v1/models                       # intra-cluster
ssh -N -L 8000:$NODE_IP:8000 <you>@aurora.alcf.anl.gov     # from laptop, then:
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"nemotron-3-ultra","messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'
```
Any OpenAI SDK: `base_url=http://127.0.0.1:8000/v1`, `api_key="dummy"`, `model="nemotron-3-ultra"`.

## Recipe internals (why these flags)
- `ZE_AFFINITY_MASK=0 -ngl 99 -sm none` — one tile, all non-expert layers on GPU.
- `-ncmoe 99` — offload all MoE experts to CPU (the 55B active MoE is the bulk of the weights).
- `--no-mmap` + `numactl --interleave=all` — stream weights once sequentially into DDR (faster cold
  load off Lustre than mmap random page-faults; interleave because the model exceeds one HBM node).
  `--membind=<one HBM node>` OOMs for the big quants — do not use it here.
- `-fa on -t 64 --numa numactl` — flash attention, 64 CPU threads for the CPU-resident experts.
- `--jinja --cont-batching --parallel N` — chat template + continuous batching for N concurrent slots.

## Long-term hosting (beyond debug)
Raise `#PBS -l walltime` and switch `-q capacity` (<=168 h), keep `llama-server` running (remove the
self-smoke/`kill` tail), and refresh clients from `ENDPOINT.txt` at each (re)start. Mirror the gpt-oss
`serve/` chaining pattern for multi-week hosting if needed.

## Caveats
- Test on `debug` (1-2 nodes) / `debug-scaling` (>=2 nodes) only, <=1 h. `A MatSciAI -l filesystems=flare`.
- MXFP4_MOE (352 GB) cold-load can exceed the 30-min debug limit; use IQ2_M for quick checks or raise
  walltime. Second-run loads are faster (page cache warm).
- No auth; binds an intra-cluster IP only. Do not expose beyond ALCF.
- IQ2_M is ~2.7 bpw (aggressive) — use MXFP4_MOE / a higher UD-Qx quant for better quality.
