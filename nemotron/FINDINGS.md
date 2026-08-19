# FINDINGS — Nemotron-3-Ultra inference on Aurora (PVC)

**Date:** 2026-08-18/19.
**Outcome:** ✅ **WORKING via llama.cpp SYCL** (OpenAI-compatible `llama-server`). vLLM is blocked by an
XPU MoE-kernel gap; llama.cpp supports `nemotron_h_moe` (hybrid Mamba2 + non-gated MoE) with SYCL and
serves the model on one Aurora tile with MoE experts offloaded to CPU.

## Result summary
| Engine | Status | Evidence |
|--------|--------|----------|
| **llama.cpp SYCL (`llama-server`)** | ✅ **works, OpenAI `/v1` endpoint** | jobs 8765117 (cli), 8765130 (server) |
| vLLM 0.15 (frameworks, XPU) | ❌ blocked: non-gated relu² MoE has no XPU kernel | job 8764932 |

**llama.cpp measured (single tile, MoE→CPU, UD-IQ2_M 2.7bpw):** single-stream prefill ~11.9 tok/s,
decode ~7 tok/s, TTFT ~2.7 s. Coherent correct output; endpoint returns proper OpenAI chat completions
with `reasoning_content`.

## Performance — single-stream vs high concurrency (job 8765192, UD-IQ2_M, max_tokens=128)

Measured against the loaded `llama-server` with `--parallel 32`, streaming client (TTFT = time to first
token; per-req decode from client; prefill/decode also from the server `timings` block):

| Concurrency | TTFT p50 | prefill tok/s (server) | per-req decode tok/s | **agg gen tok/s** | req/s |
|-------------|---------:|-----------------------:|---------------------:|------------------:|------:|
| **1**  | **2.67 s** | **11.9** | **7.1** | 6.2 | 0.048 |
| 4  | 7.6 s  | 3.1 | 2.2 | 6.9 | 0.064 |
| 8  | 16.2 s | 0.9 | 1.0 | 6.9 | 0.058 |
| 16 | 27.1 s | 0.6 | 0.56 | 7.4 | 0.063 |
| 32 | 66.2 s | 0.4 | 0.19 | 5.3 | 0.045 |

**Batched inference WAS enabled and IS functioning** — the flat throughput is a hardware bottleneck, not
a batching failure. Server loaded with `--parallel 32 --cont-batching` → `n_slots = 32, n_ctx_slot =
512`. The server log (`serve-llamacpp/logs/bench_server.out`) confirms genuine continuous batching: e.g.
at the c=8 wave, 8 slots `launch_slot_` within microseconds (tasks 275–282), their prefill overlaps, and
they `release` together — i.e. decoded in the same rolling batch, not serialized.

**Conclusion: this recipe is single-user / low-concurrency oriented — it does NOT scale with
concurrency.** Aggregate generation throughput is **flat at ~6–7 tok/s** across all concurrency levels
while TTFT and per-request decode degrade ~linearly. Root cause: with `-ncmoe 99`, every decoded token
routes top-22 experts through **CPU DDR**, and all batched slots contend for the **same CPU cores / memory
bandwidth** for the MoE FFN. Continuous batching grows the batch dimension, but CPU expert GEMMs don't
get the parallel-efficiency a GPU batched MoE kernel would, so the aggregate stays pinned at the CPU-MoE
ceiling. Same non-scaling llama.cpp MoE-offload behavior documented in the gpt-oss campaign (single-stream
fast, does not replicate for throughput). Under load, per-slot prefill also falls (~0.6–1.6 tok/s at c=8
vs ~11.9 single) — consistent with CPU-bound experts, confirmed in the server timing lines.

**Implications / options for higher concurrency:**
- Best fit today: **single-user, low-latency** serving (~2.7 s TTFT, ~7 tok/s) on one tile — leaves the
  other 11 tiles free.
- To raise node throughput: run **multiple independent single-tile server instances** (data parallel)
  behind a load balancer — but they contend for the **same CPU/DDR** for experts, so aggregate gains are
  limited (same caveat as gpt-oss llama.cpp MoE-offload; not a linear win).
- Keeping experts **on GPU** (drop `-ncmoe`, multi-tile TP) would scale better but needs the full model
  in HBM across tiles (MXFP4_MOE ~352 GB across ≥6 tiles) — untested here; a future experiment.
- The `MXFP4_MOE` quant (higher quality) was not benchmarked (352 GB cold-load vs debug walltime); rerun
  `serve-llamacpp/bench_debug.pbs -v QUANT=MXFP4_MOE` with extended walltime to compare.

Benchmark kit: `serve-llamacpp/bench_client.py` (streaming TTFT + concurrency sweep) +
`serve-llamacpp/bench_debug.pbs` (loads server once, sweeps 1/4/8/16/32).

## Full-XPU inference WITHOUT CPU offload (experts on GPU) — tested

**Question:** can Nemotron-3-Ultra run fully on the XPU (no `-ncmoe` CPU offload)?
**Answer: YES it runs fully on GPU, but it is NOT faster — slightly slower — and still does not scale
with concurrency.** Tested with llama.cpp SYCL, UD-IQ2_M spread across **8 tiles** of one node (weights
~24 GB/tile, fits ~64 GB HBM), `-ngl 99`, **no `-ncmoe`**, `--split-mode layer` (job 8765260).

Memory feasibility (~64 GB/tile, ~768 GB/node): UD-IQ2_M (194 GB) needs ≥4 tiles (safe 6–8);
MXFP4_MOE (352 GB) needs ≥6 (safe 8+). So full-GPU residency is feasible on one node.

**Full-XPU (8 tiles, layer split, experts on GPU) vs single-tile MoE→CPU offload — both UD-IQ2_M, 128 tok:**
| Concurrency | offload agg gen | **full-XPU agg gen** | offload decode/req | full-XPU decode/req | offload TTFT p50 | full-XPU TTFT p50 |
|-------------|----------------:|---------------------:|-------------------:|--------------------:|-----------------:|------------------:|
| 1  | 6.2 | **4.7** | 7.1 | 5.3 | 2.67 s | 3.21 s |
| 8  | 6.9 | **5.4** | 1.0 | 0.79 | 16.2 s | 17.3 s |
| 32 | 5.3 | **5.8** | 0.19 | 0.21 | 66.2 s | 43.8 s |

**Why full-XPU doesn't help:** `--split-mode layer` places different layers on different tiles and runs
them **sequentially (pipeline)** — 8 tiles add HBM capacity but **no parallel compute** for a single
forward pass. Aggregate stays flat ~5–6 tok/s (now GPU-bound on one active tile at a time + cross-tile
transfer + on-GPU MXFP4 dequant), essentially matching the CPU-offload ceiling. Single-stream is even a
bit slower than the 1-tile CPU-offload recipe.

**Tensor-parallel (`--split-mode row`) — does NOT work here.** Row split (which *would* give true
per-layer parallelism across tiles) **crashes the server at model load** for `nemotron_h_moe` on this
build (SIGABRT during load, no graceful error) — llama.cpp's SYCL row-split does not support this
hybrid Mamba2/recurrent + MoE architecture. Job 8765295.

**Net:** on Aurora PVC with this llama.cpp build there is **no configuration that scales Nemotron-3-Ultra
throughput with concurrency** — neither CPU-MoE-offload (CPU/DDR bound) nor full-XPU layer-split
(pipeline, single active tile) nor row-split (unsupported/crashes). The model is best served as a
**single-user / low-latency** endpoint; the **1-tile MoE→CPU recipe is the best single-stream option**
(fastest TTFT + decode) and frees the other 11 tiles for other work. True high-concurrency serving would
require either (a) a working tensor-parallel MoE path on XPU (vLLM once it adds non-gated MoE, or a
fixed llama.cpp row-split), or (b) multiple independent nodes behind a load balancer.

Full-XPU kit: `serve-llamacpp/full_xpu_bench_debug.pbs` (`-v NTILES=8,SPLIT=layer|row`).

## Data parallelism: 3 independent instances × 4 tiles on one node — DOES aggregate ✅

**Question:** launch 3 independent instances on the 12 tiles (4 tiles each) and aggregate throughput?
**Answer: YES — this is the only config that raises node throughput.** ~2.5–3× a single instance.
Tested with 3 independent `llama-server` instances, each **full-XPU on 4 disjoint tiles** (masks
{0-3},{4-7},{8-11}), UD-IQ2_M, experts on GPU (no `-ncmoe`), driven concurrently; node aggregate =
sum over instances (job 8765318). All 3 loaded fine (~48.5 GB/tile weights fits ~64 GB HBM).

| per-instance concurrency | **NODE agg gen tok/s** | NODE req/s | mean TTFT p50 |
|--------------------------|-----------------------:|-----------:|--------------:|
| 1 (×3) | 12.0 | 0.095 | 6.05 s |
| 4 (×3) | 13.6 | 0.123 | 13.4 s |
| **8 (×3)** | **15.6** | **0.131** | 12.5 s |

**Node throughput comparison (UD-IQ2_M, all on one node):**
| Config | best node agg gen tok/s | best node req/s |
|--------|------------------------:|----------------:|
| 1× 1-tile MoE→CPU | 6.9 | 0.064 |
| 1× 8-tile full-XPU (layer split) | 5.8 | 0.049 |
| **3× 4-tile full-XPU (data parallel)** | **15.6** | **0.131** |

**Why data parallelism works where the others didn't:** each 4-tile full-XPU instance is **self-contained
in HBM** with no shared CPU/DDR expert bottleneck, so the 3 instances run truly independently and the node
aggregate ≈ 3× one instance. This is exactly the gpt-oss "3× TP=4 data-parallel across 12 tiles" pattern,
now confirmed for Nemotron via llama.cpp. (Each instance is still single-user-latency-bound at ~4–5 tps;
DP raises *aggregate* node throughput and *req/s*, not per-request speed.)

**Recommended high-throughput recipe (one node):** 3 independent full-XPU `llama-server` instances of 4
tiles each behind a round-robin load balancer (reuse `serve/lb.py`), one `/v1` endpoint. Front them with
the same LB used for gpt-oss. Notes:
- Use **UD-IQ2_M** (fits full-XPU at 4 tiles). MXFP4_MOE (88 GB/tile at 4 tiles) does NOT fit full-XPU;
  for MXFP4 you'd need MoE→CPU per instance, which reintroduces shared-CPU contention (limited DP gain).
- Keep per-instance `-c`/`--parallel` modest so KV + compute fit alongside the ~48.5 GB/tile weights.

DP kit: `serve-llamacpp/dp3_bench_debug.pbs` (`-v NREP=3,TPE=4,MODE=fullxpu|offload,...`) — launches N
instances on disjoint tiles, drives them concurrently, prints `DP_NODE_JSON` node aggregate.

## Bottom line (throughput options, ranked)
1. **Node throughput / concurrency:** 3× 4-tile full-XPU data-parallel + LB → **~15.6 tok/s agg, 0.13 req/s** (best).
2. **Single-user latency:** 1× 1-tile MoE→CPU → TTFT ~2.7 s, ~7 tok/s decode (best single-stream; frees 11 tiles).
3. Avoid: single 8-tile full-XPU (pipeline, no gain) and `-sm row` (crashes on nemotron_h_moe).

---

# Part 2 — llama.cpp SYCL (the working path)

**Model:** `unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF`. Downloaded **UD-IQ2_M** (194 GB, fast to
load within debug walltime) and **MXFP4_MOE** (352 GB, native MoE quant, higher quality).

**Engine:** the existing self-built llama.cpp SYCL (`inkling/build-llamacpp-sycl/build/bin`, b10383,
ggml 0.19) — **already has `nemotron_h_moe`** (`src/models/nemotron-h-moe.cpp`, arch `LLM_ARCH_NEMOTRON_H_MOE`,
incl. MTP handling). No rebuild needed.

**Recipe (mirrors the gpt-oss F4_hbm recipe):** 1 GPU tile (`ZE_AFFINITY_MASK=0`, `-ngl 99 -sm none`),
**MoE experts → CPU** (`-ncmoe 99`), `-fa on`, `-t 64`, `--numa numactl`, and **`--no-mmap`** so the
weights stream once sequentially into host RAM (~1 TB/node) — far faster off Lustre than mmap random
page-faults for a cold run. `numactl --interleave=all` (model > one HBM node).

**Serving:** `llama-server` exposes OpenAI `/v1/chat/completions` + `/v1/models` directly (no LB needed
for a single instance), `--jinja --cont-batching --parallel N`, `--alias nemotron-3-ultra`.

**Key gotchas resolved:**
1. **Split GGUF:** point `-m` at the real `-00001-of-000NN.gguf` shard; llama.cpp derives the rest and
   rejects a renamed/symlinked path ("invalid split file name").
2. **OOM with `--no-mmap` on MXFP4_MOE (352 GB) + `--membind=2` (one HBM node ~128 GB):** the strict
   membind was too small. Use `--interleave=all` (all NUMA/DDR). IQ2_M (194 GB) loads comfortably.
3. **Debug walltime:** 352 GB cold-load + first forward can exceed 30 min; use IQ2_M for a fast
   validation, or request more walltime for MXFP4_MOE.

**Files (llama.cpp path):**
- `download_nemotron_gguf.sh` — download a GGUF quant (`QUANT=UD-IQ2_M` default, or `MXFP4_MOE`).
- `llama_smoke.pbs` — single-tile smoke (`llama-cli`), debug queue. **Validated (job 8765117).**
- `serve-llamacpp/serve_test_debug.pbs` — `llama-server` OpenAI endpoint + self-smoke, debug queue.
  **Validated (job 8765130).** `serve-llamacpp/README.md` for hosting/tunnel details.

**Scale-up notes:** for higher throughput or quality, run MXFP4_MOE (raise walltime), increase
`--parallel`/`-c`, or host long-term on the `capacity` queue (raise walltime, keep the server running,
drop the self-exit) as in the gpt-oss serving kit. Multi-tile GPU offload of experts is possible but the
single-tile MoE→CPU recipe (like gpt-oss F4_hbm) is the proven, node-friendly default.

---

# Part 1 — vLLM (blocked)

**Outcome:** Deployment kit built and driven end-to-end on the `debug` queue.
Everything up to model forward-graph construction works; a **hard vLLM-0.15 XPU MoE-kernel limitation
blocks execution**: Nemotron-3-Ultra uses **non-gated relu² MoE**, which has no XPU kernel in the
Aurora frameworks vLLM. Verified empirically (job 8764932).

---

## What was verified to WORK (on one Aurora node, debug queue, TP=8)

1. **Checkpoint**: `RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-quantized.w4a16` (INT4, 273 GB, 225
   shards) downloaded and gated OK.
2. **Config loading** — needed a shim (`nemotron_h_shim.py`):
   - transformers 4.57.6 lacks `nemotron_h` in AutoConfig; vLLM 0.15 ships `NemotronHConfig` but does
     NOT register `nemotron_h` in its config registry → falls through to HF AutoConfig and fails.
   - vLLM's shipped `NemotronHConfig` predates this NVIDIA schema (expects `hybrid_override_pattern`
     with a read-only `layers_block_type` property; new config gives explicit `layers_block_type`
     incl. `moe`, plus `n_groups`/`conv_kernel`, `num_hidden_layers: null`).
   - **Shim fix:** subclass registered for `nemotron_h` that builds `hybrid_override_pattern` from
     `layers_block_type` (mamba=M, attention=*, mlp=-, **moe=E**), migrates key names, derives
     `num_hidden_layers`. → config loads: 108 layers (48 mamba / 48 moe / 12 attention), n_groups=8,
     512 experts, latent-MoE. ✅
3. **Model registration / inspection** — needed the shim too:
   - vLLM inspects the model class in a subprocess (`python -m vllm...registry`) which **SIGSEGVs on
     PVC** (silent crash), even though importing `nemotron_h.py` in-process is fine.
   - **Shim fix:** `ModelRegistry.register_model("NemotronHForCausalLM", <class object>)` registers it
     as an *eager* model → inspection happens in-process, no subprocess, no crash. ✅
4. **INT4 quantization on XPU**: `Using XPUwNa16LinearKernel for CompressedTensorsWNA16` and
   `CompressedTensorsWNA16MarlinMoEMethod` selected on all 8 workers. compressed-tensors INT4 (w4a16)
   is supported on PVC. ✅
5. **8 TP workers spawn**, mamba page-size / attention block-size auto-config succeeds
   (`Setting attention block size to 2080 ... mamba page size ...`). ✅

## The BLOCKER (hard limitation)

During `NemotronHModel.__init__` → building a MoE layer → `SharedFusedMoE(...)`:

```
File ".../vllm/model_executor/layers/fused_moe/layer.py", line 606, in __init__
    raise NotImplementedError(
NotImplementedError: is_act_and_mul=False is supported only for CUDA and ROCm for now
```

Nemotron-3-Ultra's MoE experts are **non-gated** (`is_act_and_mul=False`) with **relu²** activation
(`mlp_hidden_act: relu2`; `nemotron_h.py:207-208` passes `activation=activation_without_mul("relu2"),
is_act_and_mul=False`). In frameworks vLLM 0.15 on XPU:

- `FusedMoE.__init__` **hard-rejects** `is_act_and_mul=False` on non-CUDA/ROCm (`layer.py:605-607`).
- Even if that guard is bypassed, **both XPU MoE kernels are SiLU/gated-only**:
  - `CompressedTensorsWNA16MarlinMoEMethod.apply` asserts `layer.activation == "silu"`
    ("Only SiLU activation is supported"; `compressed_tensors_moe.py`).
  - The Triton `fused_experts` path only supports `["silu","gelu","swigluoai"]`
    (`fused_moe.py:1950`) and assumes gated act-and-mul.

**There is no non-gated / relu² MoE kernel on the vLLM-0.15 XPU path.** This is an engine-kernel gap,
not a config/packaging issue. It cannot be closed without writing/porting a non-gated relu² MoE kernel
for XPU (or a newer vLLM-XPU that adds it) — outside the scope of a deployment task, and the campaign
rule is not to rebuild the vLLM stack.

Note the analogous non-MoE parts would likely have worked: Mamba2 kernels are pure Triton (the working
XPU path), attention is standard, and INT4 linear ran. It is specifically the **non-gated MoE** that is
unsupported on XPU today.

---

## Options going forward (pick per priority)

1. **Wait for / request a newer frameworks vLLM-XPU** that implements non-gated (relu²) MoE on XPU,
   then re-run `probe_nemotron.pbs` (the shim + kit are ready; only the kernel gap remains). *Lowest
   effort once available.* Consider filing an ALCF software request / vLLM-XPU issue referencing
   `layer.py:606` + `fused_moe.py:1950`.
2. **Serve a gated-MoE Nemotron instead.** Nemotron models whose MoE is gated SiLU (act-and-mul=True)
   should run on the existing XPU Marlin/Triton MoE path. (Nemotron-3-Ultra specifically is non-gated,
   so it is excluded.)
3. **llama.cpp SYCL** (as used for gpt-oss). If/when llama.cpp gains a `nemotron_h` (hybrid Mamba2 +
   non-gated MoE) GGUF path with SYCL kernels, it could run single-tile like the gpt-oss F4_hbm recipe.
   Not available/verified here.
4. **CPU / non-Aurora GPU** deployment — out of scope for this Aurora task.

The full deploy kit (download, probe, config shim, single-node TP=8 serve + LB + client, bench) is in
place and correct; re-running it after option 1/2/3 requires no new engineering beyond a model/kernel
that the XPU MoE path supports.

## Reproduce
```bash
cd /lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/nemotron
qsub -q debug probe_nemotron.pbs   # -> PROBE_JSON stage=construct_llm; engine log shows the MoE error
grep -E "is_act_and_mul|Only SiLU" logs/probe.out
```
Evidence: `logs/probe.out` (job 8764932), engine trace `NotImplementedError: is_act_and_mul=False ...`.
