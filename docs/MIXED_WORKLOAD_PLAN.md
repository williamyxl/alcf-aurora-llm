# Plan: 1-Tile Dedicated Inference Service (F4_hbm) + Agentic Tooling on Aurora

Mixed workload: run a persistent **gpt-oss-120b MXFP4** inference service pinned to
**one GPU tile with MoE offloaded to socket-0 CPU HBM**, while leaving the other 11
tiles + socket 1 free for a scientific workload — all inside a single PBS allocation.
The service exposes an OpenAI-compatible API that both the local scientific loop
(over localhost) and agentic tools like Kilo / OpenCode / Claude (over an SSH tunnel)
can drive across many iterations.

---

## Grounding facts (from this tree)

- **Binary:** `.../gpt-oss-120b/build-llamacpp-sycl/build/bin/llama-server` (SYCL, PVC)
- **Model:** `.../gpt-oss-120b/models/openai-gpt-oss-120b-mxfp4.gguf` (63.4 GB MXFP4)
- **Recipe F4_hbm** (best short-context decode: **41.78 tok/s**, TTFT **364 ms**, 1 tile;
  RESULTS.md job 8757601, host x4112c1s3b0n0):
  `ZE_AFFINITY_MASK=0`, `-sm none`, `-ncmoe 99`, `-t 32`,
  `numactl --physcpubind=1-51,105-155 --membind=2`, `--numa numactl`,
  `-fa on`, `GGML_SYCL_ENABLE_VMM=0`, `--no-mmap`. Model load ~49 s.
- **NUMA map (Flat):** socket 0 = CPUs `1-51,105-155` (reserved 0/104 skipped),
  DDR NUMA 0, **HBM NUMA 2 (64 GB)**; GPU0 tiles 0/1 belong to socket 0.
  Socket 1 = CPUs `53-103,157-207`, DDR NUMA 1, HBM NUMA 3; GPUs 3-5.
- **Constraint:** `--membind=2` is HBM-only *hard* bind — MoE experts must fit in 64 GB
  or the job OOMs (gpt-oss MXFP4 fits). Fallback on OOM: `--preferred=2` (spills to DDR).

---

## Resource partition

```
One Aurora node
├── Inference fence (F4_hbm)
│     GPU0 tile 0  (ZE_AFFINITY_MASK=0)
│     socket-0 HBM (NUMA 2)  = MoE experts
│     socket-0 CPUs 1-51,105-155
└── Scientific workload
      tiles 1-11 (GPU0 tile1 + GPU1-5)
      socket-1 CPUs 53-103,157-207 + DDR/HBM NUMA 1,3
```

Service consumes **1 tile + socket-0 CPU/HBM**. Workload gets **11 tiles + socket 1**.
The only contention surface is socket-0 memory bandwidth — keep the workload on
socket 1 to stay clean.

---

## Files to create in `mixed_workload/`

```
mixed_workload/
├── PLAN.md             # this file
├── env.sh              # paths + F4_hbm binding + socket-1 workload fence
├── start_service.sh    # launch llama-server, wait for /v1/models
├── job.pbs             # 1 node: service + workload, socket-split
├── tunnel.sh           # convenience ssh -L from laptop
└── endpoint.txt        # (generated at runtime) node+port of the live service
```

### `env.sh` — single source of truth
```bash
#!/bin/bash
export AURORA_LLM_ROOT=/lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm
export LLAMA_SERVER=$AURORA_LLM_ROOT/gpt-oss-120b/build-llamacpp-sycl/build/bin/llama-server
export MODEL_GGUF=$AURORA_LLM_ROOT/gpt-oss-120b/models/openai-gpt-oss-120b-mxfp4.gguf
export SERVE_PORT=8080
export SERVE_API_KEY=aurora-local            # non-empty so the fabric listener isn't open

# F4_hbm inference fence
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=0
export GGML_SYCL_ENABLE_VMM=0
export LLM_NUMACTL="numactl --physcpubind=1-51,105-155 --membind=2"
export LLM_THREADS=32

# Scientific workload fence (socket 1)
export WORK_NUMACTL="numactl --physcpubind=53-103,157-207 --membind=1"
export WORK_ZE_MASK=2,3,4,5,6,7,8,9,10,11    # tiles 1-11
```

### `start_service.sh` — persistent server (backgrounded inside the job)
```bash
#!/bin/bash -l
source "$(dirname "$0")/env.sh"
module load oneapi/release/2025.3.1
cd "$AURORA_LLM_ROOT/gpt-oss-120b/build-llamacpp-sycl"

$LLM_NUMACTL "$LLAMA_SERVER" \
  -m "$MODEL_GGUF" \
  --host 0.0.0.0 --port "$SERVE_PORT" --api-key "$SERVE_API_KEY" \
  -ngl 99 -sm none -ncmoe 99 -fa on -t "$LLM_THREADS" \
  --numa numactl --no-mmap \
  -c 8192 \
  > "$PBS_O_WORKDIR/llama-server.log" 2>&1 &
echo $! > "$PBS_O_WORKDIR/llama-server.pid"

# Wait until the OpenAI endpoint answers (model load ~50 s)
until curl -sf -H "Authorization: Bearer $SERVE_API_KEY" \
      http://localhost:$SERVE_PORT/v1/models >/dev/null; do sleep 3; done
echo "SERVICE_UP host=$(hostname) port=$SERVE_PORT" | tee "$PBS_O_WORKDIR/endpoint.txt"
```

`llama-server` speaks the OpenAI API natively at `/v1/chat/completions` — that is what
lets it drop straight into agentic tools.

### `job.pbs` — mixed workload in one allocation
```bash
#!/bin/bash -l
#PBS -A MatSciAI
#PBS -N mixed-llm-work
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare
#PBS -l place=scatter
#PBS -q debug
#PBS -k doe
cd $PBS_O_WORKDIR
source ./env.sh

./start_service.sh                    # backgrounds llama-server, waits for readiness

# ---- scientific workload, fenced to socket 1 / tiles 1-11 ----
# ZE_AFFINITY_MASK=$WORK_ZE_MASK $WORK_NUMACTL ./your_app ...
# It calls the model at http://localhost:8080/v1 across many iterations.

# Keep the job alive for interactive attach if the workload is short:
wait $(cat llama-server.pid)          # or: sleep for remaining walltime
```

### `tunnel.sh` — run on your laptop
```bash
#!/bin/bash
# Reads the compute node from endpoint.txt (scp it down first, or paste the host).
NODE="$1"   # e.g. x4112c1s3b0n0
ssh -N -L 8080:${NODE}.hsn.cm.aurora.alcf.anl.gov:8080 xiaoliyan@aurora.alcf.anl.gov
```

---

## Two ways to consume the service

### A. Programmatic (scientific loop) — the "many iterations" case
Inside the job the workload hits `http://localhost:8080/v1/chat/completions` every
iteration. Weights load **once** (~49 s), so each call is pure decode (~42 tok/s) with
a warm KV cache — no per-call reload.

### B. Agentic tools from your laptop — optional live attach
```
Laptop (Kilo / OpenCode / Claude)
  --ssh -L 8080--> Aurora login --HSN--> compute node llama-server :8080
```
1. `scp` or read `endpoint.txt` to get the compute node hostname.
2. `./tunnel.sh <node>` from the laptop.
3. Configure the tool:

**Kilo / OpenCode** (OpenAI-Compatible provider):
- Base URL: `http://localhost:8080/v1`
- API key: `aurora-local`
- Model: `gpt-oss-120b` (id reported by `/v1/models`)

**Claude Code** targets Anthropic's API shape, not OpenAI. Put a small OpenAI→Anthropic
shim on your laptop (e.g. a LiteLLM proxy exposing an Anthropic endpoint backed by
`http://localhost:8080/v1`) and set `ANTHROPIC_BASE_URL` to the shim. Kilo/OpenCode
need no shim since they speak OpenAI-compatible directly — prefer those for the simplest
path, add the shim only if you specifically need the Claude Code CLI.

---

## Operational discipline

1. **Independent processes** — server backgrounded separately from the workload so a
   crash in either survives the other (important for debugging).
2. **Socket separation** — workload on socket 1 (`WORK_NUMACTL`, tiles 1-11); never let
   it touch NUMA 2 or tile 0.
3. **HBM budget** — `--membind=2` is a hard 64 GB bind; if a larger model OOMs, switch to
   `--preferred=2`.
4. **Benchmark exception** — for *clean* performance measurement of the scientific code,
   run it on a separate node; co-location pollutes socket-0 bandwidth numbers.
5. **Ephemeral endpoint** — the node changes every job; always re-read `endpoint.txt` and
   re-tunnel.

---

## References
- F4_hbm recipe & numbers: `../gpt-oss-120b/build-vllm-xpu/opus4.8-new_campaign/RESULTS.md`
- Historical recipe: `../gpt-oss-120b/build-llamacpp-sycl/BEST_RECIPE.md`
- NUMA/binding env: `../gpt-oss-120b/build-llamacpp-sycl/numa_moe_host.env.sh`
- ALCF binding docs: https://docs.alcf.anl.gov/aurora/running-jobs-aurora/#mpi-rank-and-thread-binding-to-cores-and-gpus
- ALCF HBM docs: https://docs.alcf.anl.gov/aurora/running-jobs-aurora/#using-the-hbm-on-the-sapphire-rapids-cpus
