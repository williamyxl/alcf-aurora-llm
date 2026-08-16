# vLLM on Aurora PVC — full results & recipes (gpt-oss-120b MXFP4)

**Date:** 2026-08-16
**Model:** `openai/gpt-oss-120b` MXFP4 (HF), single stream, quality-gated (coherent MOF answer).
**Metric:** two-call method (version-agnostic) — TTFT=prefill+1tok wall; prefill_tps=n_prompt/TTFT;
decode_tps=(N-1)/(t_N - t_1); e2e=N/wall. `warm2` = steady state. N=128 output tokens.

## Working stack (the one that runs)

Aurora **`frameworks/2025.3.1`** module (Aurora 26.26.0):
vllm **0.15.0** · torch **2.10.0a0+xpu** · triton **3.6.0** · ipex **2.10.10** · oneAPI 2025.3.
MoE backend = **IPEX Marlin** (`Using ipex marlin backend on XPU`); attention = Flash Attention (XPU).

> The self-built vllm 0.27.1 / torch 2.13 stack does NOT run on PVC (attention crashes: triton 3.7.2
> `get_native` SIGSEGV; SYCL FA2 xe2-only). See DEBUG_LOG.md P11–P14. oneAPI 2026 fixed that stack's
> ABI but not its attention. The frameworks 0.15 stack is the working path.

## Critical launch requirement

Run under the **full `module load frameworks` env**, not just `conda activate` — the module sets
`CCL_CONFIGURATION=cpu_gpu_dpcpp`, `CCL_ROOT`, `CMPLR_ROOT`, SYCL/oneAPI runtime on LD_LIBRARY_PATH,
`FI_CXI_*`, which the IPEX Marlin JIT needs to select the XPU (else `sycl::exception: No device`).
Launcher: `vllm_run_fw3.pbs`. vLLM 0.15 spawns its own XPU workers — **no mpiexec / external_launcher**.

```bash
module use /opt/aurora/26.26.0/frameworks/modulefiles && module load frameworks
FWPY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
export HF_HUB_OFFLINE=1 TMPDIR=/tmp VLLM_XPU_FUSED_MOE_USE_REF=1
export TRITON_CACHE_DIR=/tmp/triton_$PBS_JOBID SYCL_CACHE_DIR=/tmp/sycl_$PBS_JOBID
$FWPY vllm_bench2.py --model <MXFP4 dir> --tp 4 --max-model-len 4096 \
  --max-num-seqs 1 --gpu-mem-util 0.85 --max-tokens 128 --enforce-eager --kv-cache-memory-gib 8
```
Queue `debug-scaling`, `select=1`, `-A MatSciAI`, `filesystems=flare`.

---

## RESULTS — all vLLM tests

### TP sweep @ 4096 context (single stream, warm2)

| TP | decode tok/s (warm / cold) | prefill tok/s | TTFT (ms) | job |
|----|-----------------------------|---------------|-----------|-----|
| 2  | 29.6 / 30.1 | 1288 | 68 | 8759327 |
| **4** | **31.9 / 33.4** | **1671** | 52 | 8759337 |
| 8  | 29.3 / 30.8 | 1863 | 47 | 8759355 |

**TP=4 = best decode (31.9 warm / 33.4 cold, >30 target MET).** TP=8 best prefill/TTFT but decode
regresses (collective overhead). TP=2 is the minimum (2 tiles hold the ~62 GB model).

### Context-length sweep

| Config | ctx | decode tok/s (warm / cold) | prefill tok/s | TTFT (ms) | GPU KV cache | job |
|--------|-----|-----------------------------|---------------|-----------|--------------|-----|
| TP=4 | 4096 | 31.9 / 33.4 | 1671 | 52 | — | 8759337 |
| TP=4 | **131072 (max)** | **31.4 / 32.8** | 1656 | 53 | 1,942,976 tok | 8759384 |
| TP=2 | 4096 | 29.6 / 30.1 | 1288 | 68 | — | 8759327 |
| TP=2 | **131072 (max)** | **28.9 / 28.8** | 1270 | 69 | 482,048 tok | 8759404 |

**Max context (131072) has negligible decode penalty** (~1–2%) vs 4096 — gpt-oss `sliding_window=128`
keeps KV cost low. TP=4 sustains >31 tok/s even at 128K context.

### Single-tile + CPU-side offload (llama.cpp F4_hbm analogue) — NOT SUPPORTED

| Config | Result | job |
|--------|--------|-----|
| TP=1 `--cpu-offload-gb 35` +NUMA membind=2 | **FAIL**: `AttributeError: _C has no attribute get_cuda_view_from_cpu_tensor` | 8759377 |

vLLM's weight `cpu_offload_gb` uses a **CUDA-only UVA op** not implemented on XPU. vLLM XPU has no
weight CPU-offload and no `-ncmoe`-style CPU-expert offload. So the llama.cpp F4_hbm single-tile+CPU-HBM
recipe **has no vLLM equivalent on PVC**; minimum single-node config is TP=2. (`swap_space` exists but is
KV-cache CPU swap under memory pressure, not weight offload.)

---

## llama.cpp F4_hbm (1-tile MoE→CPU + HBM) — context sweep (the true single-tile+offload path)

| CTX | TTFT (ms) | prefill tok/s | decode tok/s | job |
|-----|-----------|---------------|--------------|-----|
| 4096 | 411 | 41.4 | **41.56** | 8759430 |
| 131072 (max) | 372 | 45.8 | **38.07** | 8759441 |

Recipe (`llama_f4hbm_ctx.pbs`): 1 tile (`ZE_AFFINITY_MASK=0`), `-ncmoe 99` (MoE experts on CPU),
`-sm none`, `numactl --physcpubind=1-51,105-155 --membind=2` (HBM NUMA), `-t 32`, `-fa on`, `--no-mmap`.
Decode drops ~8% at 128K (larger KV) but holds ~38 tok/s. **This is the single-GPU-tile deployment
vLLM cannot match on XPU** (vLLM has no weight/MoE CPU offload; P19).

## vLLM vs llama.cpp (same model, same PVC, decode, quality-OK)

| Engine / recipe | decode tok/s | prefill tok/s | notes |
|-----------------|-------------:|--------------:|-------|
| llama.cpp F4_hbm (1-tile MoE→CPU + HBM NUMA) | **41.8** | 54.9 | fastest; true single-tile+CPU offload |
| llama.cpp P14_tp2 (2-tile pure GPU) | 34.0 | 495 (pp512) | |
| **vLLM TP=4 (frameworks 0.15, IPEX Marlin)** | **31.9–33.4** | **1671** | best vLLM; >30 target met |
| vLLM TP=2 | 29.6 | 1288 | min config |
| vLLM (old REF-MoE path, historical) | ~1.2 | — | 25–35× slower; superseded |

**vLLM decode is now within ~5–20% of llama.cpp and ~26× faster than the old REF path.** vLLM's
**prefill is much higher** (1671 vs ~55 tok/s completion / ~495 pp512) — vLLM wins prefill-bound /
long-prompt serving; llama.cpp wins pure single-stream decode and single-tile deployment.

## Recommendations
- **Decode-bound / interactive:** vLLM TP=4 (31.9 tok/s) or llama.cpp F4_hbm (41.8) if single-tile.
- **Prefill-bound / long context / serving:** vLLM TP=4 (prefill 1671 tok/s, 128K ctx at 31 tok/s).
- **Single GPU tile:** only llama.cpp (F4_hbm); vLLM XPU cannot offload weights to CPU.

## Files
- `vllm_run_fw3.pbs` — working launcher (frameworks module; OFFLOADGB/NUMA/TP/MML knobs)
- `vllm_bench2.py` — bench (two-call metrics, quality gate)
- `VLLM_WORKING_RECIPE.md` — recipe summary; `DEBUG_LOG.md` — full P1–P20 with timestamps
- Logs: `logs/vllm_fw_tp{2,4,8}*.out`, `logs/vllm_fw_tp{2,4}_maxctx.out`, `logs/vllm_fw_tp1_off_4k.out`
