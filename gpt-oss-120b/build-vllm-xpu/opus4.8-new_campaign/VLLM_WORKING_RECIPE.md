# WORKING vLLM recipe — gpt-oss-120b MXFP4 on Aurora PVC (decode >30 tok/s)

**Date:** 2026-08-16
**Status:** WORKING. Decode **31.9 tok/s (warm) / 33.4 (cold)** at TP=4, quality OK. Target >30 MET.

## The stack that works: Aurora `frameworks/2025.3.1`

Do NOT use the self-built vllm 0.27.1 stack (attention crashes on PVC — see DEBUG_LOG P11–P14).
Use the site frameworks module env instead:

| Component | Version |
|-----------|---------|
| vllm | 0.15.0 |
| torch | 2.10.0a0 (xpu) |
| triton | 3.6.0 (vllm auto-disables triton GPU funcs — avoids the 3.7.2 crash) |
| ipex | 2.10.10 |
| MoE backend | **IPEX Marlin** (`mxfp4.py:167 Using ipex marlin backend on XPU`) |
| attention | Flash Attention (XPU) |
| oneAPI | 2025.3 (frameworks module) |

## The critical fix (why it works now)

vLLM must run under the **full `module load frameworks` environment**, not just `conda activate` of the
env. The module sets `CCL_CONFIGURATION=cpu_gpu_dpcpp`, `CCL_ROOT`, `CMPLR_ROOT`, the SYCL/oneAPI runtime
on `LD_LIBRARY_PATH`, and `FI_CXI_*` — all of which the **IPEX Marlin JIT kernel needs to select the XPU
at forward time**. With only conda-activate, the IPEX Marlin JIT threw
`sycl::exception: No device of requested type available` (DEBUG_LOG P16). With full module env → works.

## Launch recipe (`vllm_run_fw3.pbs`)

```bash
module use /opt/aurora/26.26.0/frameworks/modulefiles
module load frameworks
FWPY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
export HF_HUB_OFFLINE=1 TMPDIR=/tmp
export VLLM_XPU_FUSED_MOE_USE_REF=1
export TRITON_CACHE_DIR=/tmp/triton_$PBS_JOBID SYCL_CACHE_DIR=/tmp/sycl_$PBS_JOBID
# vLLM 0.15 spawns its own XPU workers; no mpiexec / external_launcher needed.
$FWPY vllm_bench2.py --model <gpt-oss-120b MXFP4 HF dir> \
  --tp 4 --max-model-len 4096 --max-num-seqs 1 \
  --gpu-mem-util 0.85 --max-tokens 128 --enforce-eager --kv-cache-memory-gib 8
```

Queue: `debug-scaling`, `select=1`, `-A MatSciAI`, `filesystems=flare`.

## Results (gpt-oss-120b MXFP4, MML=4096, single stream, quality OK)

| TP | decode tok/s (warm) | decode (cold) | prefill tok/s | TTFT (ms) | job |
|----|---------------------|---------------|---------------|-----------|-----|
| 2  | 29.6                | 30.1          | 1288          | 68        | 8759327 |
| **4** | **31.9**         | **33.4**      | **1671**      | **52**    | 8759337 |
| 8  | 29.3                | 30.8          | 1863          | 47        | 8759355 |

**Best decode: TP=4 = 31.9 tok/s warm, 33.4 cold.** TP=8 regresses on decode (comm overhead) but wins
prefill/TTFT — choose TP=8 for prefill-bound / long-prompt, TP=4 for decode-bound.

## Comparison to llama.cpp (same model, same PVC)

| Engine | Best decode tok/s | Notes |
|--------|-------------------|-------|
| llama.cpp SYCL F4_hbm (1-tile MoE→CPU) | 41.8 | fastest overall |
| llama.cpp SYCL P14_tp2 (2-tile GPU) | 34.0 | |
| **vLLM 0.15 frameworks TP=4 (this)** | **31.9–33.4** | now competitive; was ~1.2 (REF) / crashing before |

vLLM decode is now **~25–28× faster than the old REF-MoE path (~1.2 tok/s)** and within ~5–20% of
llama.cpp — the IPEX Marlin MXFP4 MoE + working attention closed almost the entire gap.

## Metrics method

`vllm_bench2.py` uses a **two-call** measurement (version-agnostic): t1 = prefill+1 token,
tN = prefill+N tokens; decode_tps = (N-1)/(tN-t1), prefill_tps = n_prompt/t1, ttft = t1.
(vLLM 0.15 `RequestOutput.metrics` is None in V1, so engine-metric extraction returns null.)

## Notes / next optimizations (optional, to push higher)
- Enable XPU graph capture: `VLLM_XPU_ENABLE_XPU_GRAPH=1` + drop `--enforce-eager` (may add ~10–20%).
- Larger `--max-num-seqs` for aggregate throughput (batched serving).
- The `repo_utils safetensors` ERROR lines are benign (HF offline local-path warning).
