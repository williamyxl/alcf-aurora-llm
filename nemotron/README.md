# Nemotron-3-Ultra inference on Aurora (vLLM / PVC)

Deploy an OpenAI-compatible vLLM inference endpoint for **NVIDIA Nemotron-3-Ultra-550B-A55B** on Aurora,
reusing the proven gpt-oss-120b vLLM-on-XPU serving kit
(`../gpt-oss-120b/build-vllm-xpu/opus4.8-new_campaign/`).

**Read `BEST_RECIPES.md` first** — the two production recipes (high-throughput + single-user).
Then `FINDINGS.md` for the full outcome, evidence, and both engine paths.

This dir now covers **two models**: Nemotron-3-Ultra-550B (MoE) and Nemotron-4-340B (dense).

## ⭐ Best recipes — Nemotron-3-Ultra-550B (MoE; see `BEST_RECIPES.md`)
| Workload | Recipe | Throughput | Launcher |
|----------|--------|-----------|----------|
| **High throughput / concurrency** | **3× `llama-server` full-XPU, 4 tiles each (data parallel) + LB** | **~15.6 tok/s agg, 0.13 req/s** | `serve-llamacpp/dp3_bench_debug.pbs` |
| **Single user / lowest latency** | **1× `llama-server`, 1 tile, MoE→CPU** | **TTFT ~2.7 s, ~7 tok/s decode** | `serve-llamacpp/serve_test_debug.pbs` |

## ⭐ Best recipes — Nemotron-4-340B (dense; see `BEST_RECIPES_N4.md`)
| Workload | Recipe | Throughput | Launcher |
|----------|--------|-----------|----------|
| **High throughput / concurrency** | **3× `llama-server` full-XPU, 4 tiles each (data parallel) + LB** | **~9.9 tok/s agg, 0.077 req/s** | `serve-llamacpp-n4/dp_bench_debug.pbs` |
| **Single user / lowest latency** | **1× `llama-server` full-XPU, 6 tiles** | **TTFT ~2.4 s, ~11 prefill / ~2.3 decode tok/s** | `serve-llamacpp-n4/single_bench_debug.pbs` |

Nemotron-4-340B is a **dense** 340B (no MoE → no CPU-expert offload; single-tile impossible). Checkpoint:
`models/gguf-n4/` (i1-Q4_K_M, ~196 GB; download via `download_nemotron4_gguf.sh`).

> **Outcome (2026-08-19): ✅ WORKING via llama.cpp SYCL.**
> - **llama.cpp SYCL `llama-server`** serves Nemotron-3-Ultra (`nemotron_h_moe`) on **one Aurora tile**
>   with MoE experts offloaded to CPU. OpenAI `/v1` endpoint validated on the debug queue (job 8765130):
>   correct output, prefill ~10.3 tok/s, decode ~5.0 tok/s (UD-IQ2_M). **This is the recommended path.**
> - **vLLM 0.15 (XPU)** is **blocked**: Nemotron-3-Ultra uses **non-gated relu² MoE**, which has no XPU
>   kernel in the frameworks vLLM (`is_act_and_mul=False` unsupported; both XPU MoE kernels are
>   SiLU/gated-only). Config/registration/INT4 all work (via `nemotron_h_shim.py`) but model build fails.

## Quickstart (llama.cpp — the working path)
```bash
cd /lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/nemotron
bash download_nemotron_gguf.sh                       # UD-IQ2_M GGUF (~194 GB) on a login node
qsub -q debug llama_smoke.pbs                        # single-tile smoke (cli)
qsub -q debug serve-llamacpp/serve_test_debug.pbs    # OpenAI /v1 llama-server + self-smoke
# endpoint -> serve-llamacpp/ENDPOINT.txt ; full guide -> serve-llamacpp/README.md
```

## TL;DR
- Model = 550B hybrid **Mamba2 + attention + MoE** (`NemotronHForCausalLM`), 55B active.
- Checkpoint = **`RedHatAI/...quantized.w4a16`** (INT4, ~293 GB) — the only single-node-viable option on
  PVC (BF16=1.1 TB too big; NVFP4 unsupported on PVC).
- Stack = Aurora `frameworks/2025.3.1` (vllm 0.15+xpu, torch 2.10, triton 3.6, ipex 2.10,
  compressed-tensors 0.13). `NemotronHForCausalLM` is registered; Mamba2 kernels are pure Triton (the
  XPU-capable path). **Run under full `module load frameworks`.**
- Serving = **single TP=8 engine** (8/12 tiles) + LB → one `/v1` endpoint.
- Test on **debug / debug-scaling** only (≤1 h).

## Workflow
```bash
cd /lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/nemotron

# 1. Download INT4 checkpoint (~293 GB) — on a login/UAN node
bash download_nemotron_w4a16.sh

# 2. Feasibility probe (debug queue, ~1 job): load TP=8 + 1 short gen; catches Mamba/quant/XPU issues
qsub -q debug probe_nemotron.pbs
tail -f logs/probe.out            # look for PROBE_JSON={...,"ok":true,...}

# 3. Serving validation (debug queue): engine + LB + self-smoke, writes serve/ENDPOINT.txt
qsub -q debug serve/serve_test_debug.pbs
tail -f serve/logs/serve_test.out
```

## Files
| File | Purpose |
|------|---------|
| `BEST_RECIPES.md` | **⭐ The two production recipes (high-throughput DP + single-user). Start here.** |
| `FINDINGS.md` | Full outcome (llama.cpp works, vLLM blocked), all benchmarks, evidence. |
| `download_nemotron_gguf.sh` | Download a GGUF quant (UD-IQ2_M / MXFP4_MOE) for llama.cpp |
| `llama_smoke.pbs` | llama.cpp SYCL single-tile smoke (MoE->CPU), debug queue — **validated** |
| `serve-llamacpp/` | `llama-server` OpenAI `/v1` endpoint kit + README — **validated** |
| `PLAN.md` | Model facts, quant/TP analysis, stack, queues (vLLM planning) |
| `nemotron_h_shim.py` | vLLM config + eager-registration shim (unblocks config/registration on XPU) |
| `probe_nemotron.{py,pbs}`, `serve/`, `smoke_bench.py` | vLLM path (blocked at MoE kernel; kept for record) |
| `download_nemotron_w4a16.sh` / `.pbs` | Resumable HF download of the INT4 checkpoint |
| `probe_nemotron.py` / `.pbs` | Feasibility probe (registry, INT4 load, Mamba Triton on XPU, 1 gen) |
| `smoke_bench.py` | Two-call decode/prefill/TTFT bench (version-agnostic) |
| `serve/serve_test_debug.pbs` | Single-node `vllm serve` TP=8 + LB + self-smoke (debug queue) |
| `serve/lb.py`, `serve/client_example.py`, `serve/README.md` | LB, client, hosting guide |

## Queues (test only)
`debug` (1–2 nodes, ≤1 h, 1 job/user) → single-node probe & serve test (`select=1`).
`debug-scaling` (2–256 nodes, ≤1 h) → multi-node only.
All jobs: `-A MatSciAI -l filesystems=flare`.

## Status
**Deployed via llama.cpp SYCL** — smoke (job 8765117) and OpenAI `/v1` server (job 8765130) both
validated on the debug queue with correct output. GGUF quants on disk: UD-IQ2_M (194 GB), MXFP4_MOE
(352 GB). vLLM path is blocked by the XPU non-gated-MoE kernel gap (kept documented in `FINDINGS.md`;
the dead vLLM INT4 checkpoint was removed to reclaim quota).
