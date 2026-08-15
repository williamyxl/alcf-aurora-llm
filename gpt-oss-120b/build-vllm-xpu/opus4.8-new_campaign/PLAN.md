# New campaign — verify why vLLM-XPU is ~25× slower than llama.cpp SYCL

**Date:** 2026-08-15
**Author:** Kilo (Claude Opus 4.8)
**Scope:** Experiment plan to confirm/rule out the ranked causes of the quality-OK
single-stream decode gap on Aurora PVC: **vLLM REF MoE ≈ 1.2 tok/s vs llama.cpp SYCL ≈ 30–34 tok/s**
(same silicon, same MXFP4 gpt-oss-120b checkpoint).
**Inputs:** `../BEST_PRACTICE.md`, `../PERF.md`, `../perf-team/{A1_hypotheses,PATHS_TO_EXPECTED_PERF,FUSED_MOE_QUALITY,HALFPREC_TP248}.md`, `../../build-llamacpp-sycl/BEST_RECIPE.md`, `../../bench_perf.py`, `../../bench_perf_tp2.pbs`, and `../../../cursor-opus4.8-med-diagnosis.md`.

> Analysis-and-design doc. Nothing here has been run yet. Every experiment obeys the standing
> rules in `../BEST_PRACTICE.md`: quality gate first; report TP=2/4/8 with P7 fields where a TP
> claim is made; never call e2e wall "TTFT"; keep fused/mxfp4_fp8 out of default scripts until
> `quality_ok`.

---

## 1. The gap and what is already ruled out

Same PVC, same MXFP4 weights, single stream, warm decode:

| Path | decode tok/s | quality | notes |
|------|-------------:|---------|-------|
| llama.cpp SYCL, 1-tile MoE→CPU + HBM NUMA (`F4_hbm`) | **~34** | PASS | `../../build-llamacpp-sycl/BEST_RECIPE.md` |
| llama.cpp SYCL, 2-tile pure GPU (`P14_tp2`) | **~30** | PASS | |
| **vLLM REF MoE, TP=2** (only quality-OK vLLM path) | **1.22** | PASS | current best practice |
| vLLM fused MXFP4 MoE, TP=2 | ~5.2 | **FAIL** (`!!!`/id-0) | `FUSED_MOE_QUALITY.md` |
| vLLM BF16/FP16 unquant, TP=4 | ~3.0 | **FAIL** | `HALFPREC_TP248.md` |

**Already ruled out (do not re-litigate):**
- MXFP4→BF16 "upcast/casting tax" as the *primary* cause — halfprec was still ~3 tok/s **and** quality FAIL.
- `enforce_eager=False` alone as a win — job 8680603 showed no change (but graph capture likely never engaged; see E6).
- Higher TP under REF — inverse scaling (TP2 > TP4 > TP8), confirmed.
- TP=12, OpenCL-in-selector, FLASH_ATTN, `module load frameworks`.

**Two facts constrain the explanation:** even the fast-but-broken fused path (5.2) is ~6× slower
than llama.cpp, and BF16 casting was ruled out — so this is a **software-stack** gap, not hardware
and not merely a quality workaround.

---

## 2. Candidate causes (ranked)

Two *different* bugs on two *different* code paths can both be true and are not mutually exclusive:

| ID | Cause | Path | Why suspected | Est. share of gap |
|----|-------|------|---------------|-------------------|
| **C1a** | **Fused MXFP4 MoE is broken** → forces REF fallback | fused (cutlass grouped GEMM) | Contiguous half-split `gemm1_clamp_limit` applied to gpt-oss **interleaved** gate/up SwiGLU (`fused_moe_interface.py` L376–381); kernel pins `g4002cea90`/`g109b736b8` predate upstream fixes ([vLLM #33679](https://github.com/vllm-project/vllm/pull/33679)) | root of *why we are on REF* |
| **C1b** | **REF MoE over-computes experts** (evaluates all 128, or loops experts in Python, instead of top-4 gather) | REF (`ref_fused_moe`) | 128 experts, top-4 routing → all-experts is ~**32× FLOPs** at BS=1, ≈ the observed 25× | **dominant magnitude** |
| **C2** | Dequant-to-bf16 kills decode bandwidth | REF | BS=1 decode is weight-streaming-bound; reading bf16 ≈ 4× bytes/token vs 4-bit-native | 2–4× multiplier |
| **C3** | Eager + per-op host / Level-Zero dispatch overhead | both | `enforce_eager=True`, `TORCHDYNAMO_DISABLE=1`, `TORCH_COMPILE_DISABLE=1`; thousands of tiny launches/token vs one SYCL graph | 10–20%+ (XPU) |
| **C4** | TP=2 multiprocess + oneCCL all-reduce per layer | both | inverse TP scaling is a comm signature; llama.cpp best is 1-tile, zero cross-tile comm | med |
| **C5** | TRITON_ATTN (JIT) vs tuned SYCL flash-attn | both | ~20× **prefill** gap points beyond MoE; quality-gated (FLASH garbles gpt-oss) | prefill-weighted |
| **C6** | Cold Triton/IGC JIT + ephemeral caches | both | per-job cache dirs, heavy first-generate compile | startup only (S2 showed warm≈warm2) |

**Relationship to prior team hypotheses:** C1a↔H1 (adds the clamp/interleave root), C1b↔H1/H11
(adds the all-128-vs-top-4 test), C2↔H3, C3↔H4, C4↔H7, C5↔H6, C6↔H2/H5.

---

## 3. Experiments

Ordered so the cheapest strong discriminators come first. **E2 (REF expert-count audit)** and
**E1 (one warm decode profile)** together adjudicate most of the ranking.

| # | Experiment | Targets | Decision rule |
|---|-----------|---------|---------------|
| **E1** | **Operator/timeline profile of one warm decode.** TP=2 REF, warm2 step, under `torch.profiler` (XPU) or Intel `unitrace`/PTI. Bucket wall time: MoE expert GEMM, attention, CCL/all-reduce, host-idle gaps. | C1b/C2/C3/C4/C5 | MoE-GEMM-dominated → C1b/C2. Large host gaps between kernels → C3. Large CCL slice → C4. Attn heavy → C5. Single most informative run. |
| **E2** | **Audit `ref_fused_moe` expert count.** Read the kernel/wrapper source in the vllm-xpu-kernels source tree; instrument (counter or one-layer microbench at BS=1) to measure whether it evaluates top-4 or all-128 experts, and whether experts are looped in Python. | **C1b** | All-128 (or Python loop) → confirms ~32× and a likely *fixable* bug. Decisive + cheap; no full model load needed. |
| **E3** | **Batch-scaling sweep.** `bench_perf.py --tp 2 --moe-mode ref --max-num-seqs {1,4,8,16}` with matching concurrent prompts; watch **aggregate** decode tok/s. | C1b/C2 vs C3/C4 | Near-linear aggregate scaling ⇒ launch/overhead-bound (C3/C4). Flat ⇒ compute/bandwidth-bound kernel (C1b/C2). |
| **E4** | **Achieved HBM bandwidth during decode.** From E1 memory counters (or tok/s × bytes/token) for REF vs fused vs llama.cpp vs PVC ~1.6 TB/s roofline. | **C2** | REF near roofline reading bf16 while llama.cpp near roofline reading 4-bit ⇒ precision/bandwidth is a real multiplier. REF far below roofline ⇒ blame C1b/C3, not bandwidth. |
| **E5** | **TP=1 vs TP=2-per-tile.** `bench_perf.py --tp 1 --kv-cache-memory-gib 4 --max-model-len 2048` (shrink KV/context to fit ~60 GB MXFP4 in one 64 GB tile). | **C4** | TP=1 ≫ TP=2/tile ⇒ CCL/multiprocess comm is a major tax (matches inverse scaling + llama.cpp 1-tile win). If TP=1 OOMs, fall back to E1's CCL bucket. |
| **E6** | **Force graph/compile capture and *verify it engaged*.** Re-run `enforce_eager=False` but assert Dynamo/graph capture is non-NONE (check compile logs / `CUDAGraph memory`), since 8680603 likely captured nothing. | **C3** | Rules out C3 only if capture is *confirmed active* and decode still flat. If capture stays NONE, C3 is real-but-unremovable in vLLM-XPU today. |
| **E7** | **NUMA/HBM binding on existing REF TP=2 job.** Wrap the bench with llama.cpp-style binding (`numactl --physcpubind=<socket0 HWTs 1-51,105-155> --membind=<HBM node 2>`); PBS launch-line only, no code change. | **C2/C4 host side** | REF decode rises materially (as 30→34 for llama.cpp) ⇒ host/memory affinity is a free multiplier worth folding into the recipe. |
| **E8** | **Fix + re-bench fused path (contingent on E1/E2).** (a) Patch: skip contiguous half-split clamp for gpt-oss interleaved SwiGLU (`fused_moe_interface.py` L376–381) and re-run quality gate at TP=2; **or** (b) build a **side env** with newer `vllm-xpu-kernels`+matching vLLM and re-run fused with the gate. | **C1a** | Fused becomes `quality_ok` ⇒ confirms C1a and unlocks ~5 tok/s immediately, with headroom toward ≥10. Keep the PASS stack untouched. |
| **E9** | **Parity / measurement audit.** Confirm both engines compared on: same prompt, same generated-token count, warm, **decode-only** metric (vLLM `decode_tok_s`, not e2e; llama.cpp `gen_tps`), same tiles/context. | measurement artifact | Sanity gate before trusting the 25×: rules out apples-to-oranges (e2e-incl-prefill vs pure gen). Run first. |

---

## 4. Execution sequence

1. **E9** (parity audit) — 30 min desk check; make sure the 25× is real and like-for-like.
2. **E2** (REF expert-count audit) — source read + one-layer microbench; cheapest decisive test of the dominant C1b hypothesis.
3. **E1** (profile one warm decode) — single run; adjudicates C1b/C2/C3/C4/C5 at once. Pair E7 binding in the same job to get E7 for near-free.
4. **E4** (bandwidth from E1 counters) and **E3** (batch sweep) — classify compute/bandwidth vs launch/comm bound.
5. **E5** (TP=1) — isolate CCL/multiprocess tax; direct structural analogue to llama.cpp 1-tile.
6. **E6** (verified graph capture) — close out C3 as removable or not.
7. **E8** (fused fix / newer kernels) — only after E1+E2 point at C1a; largest potential single win inside vLLM.

**Stop conditions / branch:**
- If **E2** shows REF evaluates all 128 experts → prioritize a top-4-gather REF fix; likely a bigger,
  easier win than the fused clamp bug, and it keeps quality PASS.
- If **E8(a)** clamp-skip makes fused `quality_ok` → new best practice at ~5 tok/s; then chase ≥10 via
  C2/C3/C4 multipliers.
- If both stall → the data supports the strategic conclusion that **llama.cpp SYCL is the right engine
  for gpt-oss-120b on Aurora today**, and vLLM parity needs a correct+genuinely-fused MXFP4 MoE plus
  working graph capture (substantial kernel-stack effort).

---

## 5. How to run (PBS conventions)

Base recipe is unchanged from `../../bench_perf_tp2.pbs` (REF MoE, TRITON_ATTN, eager, KV pin 8 GiB,
`ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `oneapi/release/2025.3.1` only, **no** frameworks). Queues:
`debug` / `debug-scaling`, `walltime=00:59:59`, `-A MatSciAI`, `filesystems=flare`, `#PBS -j oe`,
one running debug job per user.

**Conda activation (current path):**

```bash
source /lus/flare/projects/MatSciAI/xiaoliyan/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV"   # $ENV = build-vllm-xpu/env
```

> All tracked scripts have been migrated from the old `MOFA/.../workdir/llm` project path to
> `MatSciAI/xiaoliyan/workdir/alcf-aurora-llm` and to the `MatSciAI/xiaoliyan/miniforge3` conda base.

**Do NOT `module load frameworks`.** The site frameworks module does not provide a working
causal / gpt-oss-120b path for this project (no usable MoE/MXFP4 route; consistent with
`../BEST_PRACTICE.md` "Explicitly not best practice" and `../perf-team/BETTER_SOLUTIONS.md`). This
campaign runs entirely on the self-built `build-vllm-xpu/env` stack with `oneapi/release/2025.3.1`
only. E8(b)'s newer-kernels experiment is a **separate self-built conda env** — also not frameworks.

Per-experiment `bench_perf.py` invocations:

```bash
# E3 batch sweep (repeat per --max-num-seqs; supply matching concurrent prompts)
python bench_perf.py --tp 2 --moe-mode ref --max-tokens 128 \
  --gpu-memory-utilization 0.82 --max-num-seqs 16 --kv-cache-memory-gib 8

# E5 TP=1 (shrink KV + context to fit one 64 GiB tile)
python bench_perf.py --tp 1 --moe-mode ref --max-tokens 128 \
  --gpu-memory-utilization 0.90 --max-num-seqs 1 --kv-cache-memory-gib 4 --max-model-len 2048

# E6 verified graph capture (assert capture engaged in logs afterward)
python bench_perf.py --tp 2 --moe-mode ref --enforce-eager false \
  --gpu-memory-utilization 0.82 --max-num-seqs 2 --kv-cache-memory-gib 8
```

E1/E4 wrap the E-baseline invocation under the XPU profiler (or `unitrace`). E7 prefixes the
`mpiexec ... python bench_perf.py` line with `numactl --physcpubind=1-51,105-155 --membind=2`
(ALCF bind: GPU0 ↔ socket 0 HWTs `1-51,105-155`; DDR NUMA 0; HBM NUMA 2 —
`../../build-llamacpp-sycl/BEST_RECIPE.md`). E2 needs the vllm-xpu-kernels **source tree** (build-src),
not just the installed wheel, to read/instrument `ref_fused_moe`. E8(b) builds a **separate** conda
env; do not displace the PASS stack.

**Reporting:** every run emits `PERF_JSON` (`../../bench_perf.py` schema — `ttft_source=engine`,
`decode_tok_s`, `quality_ok`, per-run cold/warm/warm2). Log new results in `../PERF.md` and, for any
recipe change, `../BEST_PRACTICE.md`. Any path with token-id-0 / `!!!` is FAIL regardless of tok/s.

---

## 6. References

- Local: `../BEST_PRACTICE.md`, `../PERF.md`, `../perf-team/PATHS_TO_EXPECTED_PERF.md`,
  `../perf-team/FUSED_MOE_QUALITY.md`, `../perf-team/A1_hypotheses.md`,
  `../../build-llamacpp-sycl/BEST_RECIPE.md`, `../../../cursor-opus4.8-med-diagnosis.md`
- Suspect code: vllm-xpu-kernels `fused_moe_interface.py` L376–381 (half-split `gemm1_clamp_limit`)
- Upstream: [vLLM #33679](https://github.com/vllm-project/vllm/pull/33679) (MXFP4 MoE accuracy / `!!!!`),
  [RFC #33214](https://github.com/vllm-project/vllm/issues/33214) (XPU kernel migration),
  [vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels)
