# A1 — Hypothesis Scout: gpt-oss-120b slow inference on Aurora PVC / vLLM-XPU

**Agent:** A1 (Hypothesis Scout)  
**Date:** 2026-07-18  
**Scope:** Ranked hypotheses for ~343 s reported TTFT / ~0.37 e2e tok/s on cold Phase 5 PASS.  
**Constraint:** Hypotheses only — no solutions (A3), no code edits, no job submits.

---

## 1. Sources consulted

### Local

| Path | Relevance |
|------|-----------|
| `build-vllm-xpu/SUCCESS_INFER.md` | PASS recipe + METRICS_JSON (`ttft_s≈343`, `e2e_tok_s≈0.37`) |
| `build-vllm-xpu/PERF_PLAN.md` | Known bottlenecks (REF MoE, cold JIT, eager, TP=8/12, PVC FP4 reality) |
| `build-vllm-xpu/PHASE_STATUS.md` | Quality gate: fused MXFP4 → `!!!`; REF+TRITON_ATTN → coherent |
| `build-vllm-xpu/logs/test_gptoss.out` (job `8680184`, host `x4303c1s3b0n0`) | Warmup **434.94 s** then timed **343.31 s** (post-warmup!); SYCL/Triton JIT spam; `reduce_segments` JIT warn; CUDAGraph=0.0 GiB |
| `one_chat.py` / `infer_chat.pbs` | TP=8, `enforce_eager=True`, `TRITON_ATTN`, `VLLM_XPU_FUSED_MOE_USE_REF=1`, ephemeral `TRITON_CACHE_DIR`/`SYCL_CACHE_DIR`, `TORCHDYNAMO_DISABLE=1`, `CCL_WORKER_COUNT=1`, `disable_custom_all_reduce=True` |
| `bench_perf.py` / `bench_perf.pbs` | Phase-0 cold/warm/warm2 harness (warm baseline may not exist yet) |

### Web / upstream

| URL | Relevance |
|-----|-----------|
| https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive | TRITON_ATTN is portable default on XPU; not FlashInfer-class CUDA path |
| https://github.com/vllm-project/vllm-xpu-kernels | XPU MoE / MXFP4 / attention kernels; fused path exists separately from REF |
| https://github.com/vllm-project/vllm/issues/33214 | XPU kernel migration; MXFP4 MoE support landed as fused SYCL path |
| https://github.com/intel/intel-xpu-backend-for-triton/issues/7144 | Triton+IGC cold JIT measurement methodology on XPU |
| https://github.com/intel/intel-xpu-backend-for-triton/issues/5258 | MoE GEMM on PVC: FP8 often upcast; BF16 MoE still tunable / not free |
| https://next.redhat.com/2025/05/16/understanding-triton-cache-optimizing-gpu-kernel-compilation/ | Triton cache keys / cold compile cost; ~30% startup win with warm cache |
| https://vllm.ai/blog/2026-02-01-gpt-oss-optimizations | gpt-oss-120b needs native FP4/MoE co-design for SoTA; PVC lacks native FP4 TC |
| https://ai-muninn.com/en/blog/part2-gpt-oss-120b-serve-script | `--enforce-eager` roughly halves tok/s on GB10; graphs matter for decode |
| https://github.com/intel/llm-scaler/issues/383 | XPU `--enforce-eager` ~10–20% cost when graphs/compile unavailable |
| https://docs.vllm.ai/en/v0.23.0/design/debug_vllm_compile/ | `enforce_eager` disables torch.compile + CUDAGraphs |
| https://github.com/vllm-project/vllm-gaudi/pull/1567 | MXFP4 dequant-at-load vs native packed MoE is a known large cost axis |
| https://docs.alcf.anl.gov/aurora/data-science/inference/vllm/ | Aurora TP/Ray/ZE_FLAT guidance (comms topology context) |
| https://github.com/vllm-project/vllm/pull/42436 | Triton MoE TD path; XPU often routes MoE via SYCL kernels not Triton MoE |

---

## 2. Metric hygiene (before ranking)

Phase 5 PASS numbers are **easy to misread**:

| Observation | Implication |
|-------------|-------------|
| `one_chat.py` runs **warmup** then **timed** generate | Official `METRICS_JSON` is the **second** generate |
| Log: warmup `434.94 s/it`, timed `343.31 s/it` | **~0.37 tok/s survives after one full generate** → not “cold JIT alone” |
| `ttft_s == wall_s == 343.3` and `decode_tok_s="n/a"` | Engine `first_token_latency` missing → “TTFT” is **e2e wall**, not true first-token |
| Warmup log has heavy IGC/SYCL compile + `reduce_segments` JIT warn | Cold JIT **does** dominate first generate; timed still slow ⇒ **steady-state also broken** |
| `CUDAGraph memory: 0.0 GiB` | Graphs off (eager) confirmed in PASS log |

**WARM baseline required** before confirming any hypothesis that claims “cold-only” vs “steady-state.” `bench_perf.py` cold/warm/warm2 is the right instrument; results may not exist yet.

---

## 3. Ranked hypotheses

| ID | Hypothesis | Evidence / sources | Expected impact if true | Notes / warm-baseline? |
|----|------------|--------------------|-------------------------|------------------------|
| **H1** | **REF MoE path (`VLLM_XPU_FUSED_MOE_USE_REF=1`) dominates steady-state compute** — unfused / reference expert math (dequant + naive GEMM/gather) instead of fused XPU MXFP4 MoE | PASS requires REF for coherent text (`SUCCESS_INFER`, `PHASE_STATUS`); timed generate after warmup still ~343 s / 0.37 tok/s (`test_gptoss.out`); PERF_PLAN ranks MoE as P1; fused path previously → all `!` / token id 0 | **High** (decode + prefill) | **Needs warm baseline + MoE A/B** to separate from residual JIT. Log still prints `Using XPUExpertsMxFp4` under REF — confirm what REF actually disables inside kernels. |
| **H2** | **Cold Triton/IGC JIT (esp. `kernel_unified_attention` / related shapes) inflates first-generate wall and contaminates reported “TTFT”** | Massive SYCL `#include` compile spam during PASS warmup; `jit_monitor` + `reduce_segments` warn; SUCCESS_INFER attributes cold compile; Intel Triton issue #7144; Red Hat Triton cache article | **High for cold TTFT / first generate**; **Med for warm e2e** (unless re-JIT continues) | **Needs warm/warm2** to quantify residual. Ephemeral caches amplify (see H5). |
| **H3** | **MXFP4 → BF16/FP16 software dequant on every MoE (and possibly dense) matmul — PVC has no native FP4 tensor cores** | PERF_PLAN PVC table; vLLM gpt-oss Blackwell blog (native FP4 co-design); Gaudi PR #1567 (dequant-at-load vs packed); Intel Triton MoE #5258 (FP8 upcast on PVC) | **High** (structural ceiling vs Blackwell) | True even with fused kernels; sets upper bound. Warm baseline needed to measure *compute* ceiling without JIT. |
| **H4** | **`enforce_eager=True` + `TORCHDYNAMO_DISABLE`/`TORCH_COMPILE_DISABLE` eliminate graph capture / compile fusion → high CPU launch + dispatch overhead per token** | PASS args + log `CUDAGraph memory: 0.0 GiB`; vLLM compile debug docs; llm-scaler #383 (~10–20% on XPU); GB10 writeups (~2× with graphs) | **Med–High** on decode tok/s | **Needs warm baseline**; graph-on experiments are separate from JIT cache. |
| **H5** | **Ephemeral / non-persistent Triton+SYCL caches force full cold compile every job** (`TRITON_CACHE_DIR=$TMPDIR/triton_cache_$$`, unset `SYCL_CACHE_PERSISTENT`, per-job SYCL dir) | `infer_chat.pbs`; SUCCESS_INFER “per-job caches”; Triton cache docs | **High for cold start / job TTFT**; **Low–Med for in-process warm2** | Warm-within-job can still look good; **cross-job cold** stays bad until persistent cache. Flag: confirm worker processes share same cache dir. |
| **H6** | **`TRITON_ATTN` is a correctness workaround that is slower than FLASH_ATTN / XPU flash kernels for this model** | PASS forces `attention_backend=TRITON_ATTN` (FLASH garbles gpt-oss); portable Triton blog; earlier log lines used Flash backend on failing runs | **Med** | Quality-gated. Warm A/B only if a correct non-Triton attn exists. |
| **H7** | **TP underutilization / imbalance: TP=8 of 12 tiles leaves 4 idle; plus `disable_custom_all_reduce=True` + oneCCL sockets / `CCL_WORKER_COUNT=1` add per-layer sync cost** | `xpu_count 12`, TP=8 in PASS; PERF_PLAN P2; ALCF vLLM Aurora docs; xccl backend in log; CCL env in `infer_chat.pbs` | **Med** (tok/s and latency) | TP=12 may help or hurt (more comm). **Warm TP=8 vs TP=12** required. |
| **H8** | **Single-stream offline chat (`one_chat.py`) underuses continuous batching / serving scheduler — reported tok/s is worst-case BS=1, not serve aggregate** | Offline `LLM.generate` smoke; PERF_PLAN P4; industry MoE decode often needs batching to hide expert/comms latency | **Med for “production tok/s” narrative**; **Low for fixing BS=1 latency** | Serve bench needed; do not conflate with H1. |
| **H9** | **Metric artifact: missing per-token timestamps make “343 s TTFT” equal full generate wall, exaggerating first-token story** | `metrics_from_output` fallback (`ttft_s = wall_s`); METRICS_JSON equality; decode_tok_s n/a | **High for interpretation**; **Low for true hardware speed** | Warm baseline with proper TTFT/decode split (bench_perf) required before ranking cold vs steady. |
| **H10** | **Shape-specialized Triton re-JIT during decode (new seq lengths / reduce kernels) keeps “warm” runs partially cold** | `reduce_segments` JIT-during-inference warn on PASS; jit_monitor mode=warn; Triton specialization cache issues (#6053/#6790) | **Med** if many shapes; **Low** if only 1–2 extra compiles | **Needs warm2** + JIT warn count on timed run. |
| **H11** | **MoE without expert parallelism / suboptimal prepare-finalize (`MoEPrepareAndFinalizeNoDPEPModular`) serializes experts under TP-only** | PASS log: `Using MoEPrepareAndFinalizeNoDPEPModular`; Intel container notes EP/TP MoE gaps on XPU V1 | **Med** | Warm profile of MoE vs attn vs CCL time splits needed. |
| **H12** | **Weight load / Lustre prefetch (~105–126 s) and engine warmup stalls (`shm_broadcast` 60 s warnings) pollute operator perception of “inference slow,” but are outside generate metrics** | PASS load ~105 s; prefetch ~126 s; repeated shm_broadcast during compile | **Low for generate tok/s**; **Med for job walltime UX** | Separate from H1/H2 generate path. |

---

## 4. Hypotheses that **need a WARM baseline** before confirmation

Explicit **WARM / warm2 required** (do not treat cold Phase 5 METRICS as decisive):

| ID | Why warm is required |
|----|----------------------|
| **H1** (REF MoE) | Must show slow tok/s **after** JIT-free generates; else cold compile confounds MoE cost |
| **H2** (cold JIT) | Confirm magnitude by cold − warm Δ; SUCCESS_INFER over-attributes 343 s to JIT despite post-warmup timed run |
| **H3** (MXFP4 dequant ceiling) | Need steady-state GFLOPS/bandwidth vs theoretical BF16 XMX |
| **H4** (eager / no graphs) | Graph vs eager only meaningful on hot kernels |
| **H5** (cache dirs) | Distinguish in-process warm vs cross-job cold with persistent cache |
| **H6** (TRITON_ATTN) | Attn backend A/B on warm decode |
| **H7** (TP / CCL) | TP=8 vs 12 and CCL knobs on warm decode |
| **H10** (re-JIT) | Count JIT warns on warm vs warm2 |
| **H11** (MoE EP/modular) | Profile warm layer times |

**Can be assessed with less dependence on warm (still better with it):**

- **H8** (serve vs single-shot) — needs serve bench, not just warm single-stream  
- **H9** (metric artifact) — already evidenced by code + METRICS_JSON  
- **H12** (load/prefetch) — already timed separately from generate in logs  

---

## 5. Working priority order (for A2 measurement / A3 solutions)

Suggested investigation order **after** P0 warm baseline exists:

1. **H9 + H2 + H5** — sanitize metrics; quantify cold vs warm vs warm2; cache persistence  
2. **H1 + H3** — MoE path / dequant (largest likely algorithmic gap; quality-gated)  
3. **H4** — eager / graphs once kernels are hot  
4. **H6 + H10** — attention backend and residual JIT shapes  
5. **H7 + H11** — TP/CCL/EP scaling  
6. **H8** — serve/continuous-batch realism  
7. **H12** — job UX only  

---

## 6. Non-goals reminder

- No solutions, recipes, or patch proposals here (A3).  
- No code edits or PBS submits by A1.  
- Do not claim Blackwell-class native FP4 performance on PVC.
