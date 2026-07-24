# A2 — Hypothesis Review: gpt-oss-120b XPU perf (Aurora)

**Agent:** A2 (Hypothesis Reviewer)  
**Date:** 2026-07-18  
**Input:** `A1_hypotheses.md` (full); skim `PERF_PLAN.md`, `SUCCESS_INFER.md`  
**Constraint:** Keep/kill/merge + ranking only — no solutions (A3), no code edits, no job submits.

**Warm-baseline flag:** Job **8680399** (`bench_perf.pbs`) is still **queued** (`Q` on debug). Cold/warm/warm2 `PERF_JSON` does **not** exist yet. Any claim that separates cold JIT from steady-state compute remains **provisional** until that job finishes.

---

## 1. Verdict on A1 methodology / metric hygiene

**Methodology: solid.** A1 consulted PASS logs + recipe + PERF_PLAN + relevant upstream issues; ranked by expected impact; and correctly called out quality gates (fused MXFP4 → `!!!` / token-id-0; OpenCL in `ONEAPI_DEVICE_SELECTOR` → vLLM MP SEGV). Scope stayed hypothesis-only.

**Metric hygiene: A1 is right; SUCCESS_INFER is easy to misread.**

| Point | A2 take |
|-------|---------|
| Timed generate after warmup still ~343 s / ~0.37 e2e tok/s | Confirmed from A1 + PASS log narrative → **steady-state is broken**, not “cold JIT alone.” |
| `ttft_s == wall_s` and `decode_tok_s="n/a"` | Confirmed in `one_chat.py` fallback (`ttft_s = wall_s` when `first_token_latency` missing) → **“343 s TTFT” is e2e wall**, not true first-token. |
| SUCCESS_INFER note that JIT “dominated” timed metrics | **Over-attribution**; A1 correctly flags this. Cold JIT still matters for first generate / cross-job, but cannot be the sole explanation of post-warmup 0.37 tok/s. |
| Warm required before cold-vs-steady ranking | **Agree.** Do not treat Phase 5 `METRICS_JSON` as decisive for H1–H7/H10/H11 until 8680399 (or equivalent) lands. |

**Minor A1 gaps (non-blocking):** (1) H1 should explicitly note log still says `Using XPUExpertsMxFp4` under REF — “what REF disables” is a measurement prerequisite, not a separate hypothesis. (2) H3 is partly a **hardware ceiling** and partly a **current-path cost**; those should not be conflated when ranking chase order. (3) Suggested priority putting H9+H2+H5 before H1 is correct for *measurement*; for *impact once warm exists*, H1 remains the top algorithmic suspect.

Default constraint upheld: **full stack rebuild out of scope.**

---

## 2. Per-hypothesis table

| ID | Decision | Possible? | Likelihood | Effort to test | Rationale |
|----|----------|-----------|------------|----------------|-----------|
| **H1** | **keep** | Y | **H** | **M** | REF is quality-required; timed post-warmup still ~0.37 tok/s → MoE path is the strongest *steady-state* suspect. Needs warm baseline + careful A/B (quality gate: fused → `!!!`). Confirm what REF actually changes despite `XPUExpertsMxFp4` log line. |
| **H2** | **keep → merge** (JIT family) | Y | **H** (cold) / **M** (warm residual) | **S** | Cold Triton/IGC compile is evidenced in PASS warmup log. Explains first-generate inflation; **does not alone** explain timed 343 s. Magnitude = cold−warm Δ from 8680399. |
| **H3** | **keep** (pair with H1) | Y | **H** (ceiling) / **M** (sole cause of 0.37) | **M–L** | PVC has no native FP4 TC — structural fact. Always true under fused or REF; sets upper bound. Likelihood it is *the* current bottleneck is medium without profiling; high that dequant+naive REF math is expensive. Full GFLOPS proof is L; accepting as ceiling context is S. |
| **H4** | **keep** | Y | **M** | **M** | Eager + Dynamo/compile disable confirmed (`CUDAGraph=0.0 GiB`). XPU literature suggests ~10–20% not 2×; still worth chasing **after** warm + MoE clarity. Graph-on is quality/stability sensitive on this stack. |
| **H5** | **keep → merge** (JIT family) | Y | **H** (cross-job cold) / **L–M** (in-process warm2) | **S** | Ephemeral Triton/SYCL dirs in PASS recipe are explicit. High impact on every-job cold TTFT; low for within-job warm2 once kernels hot. Worker-share of cache dir is a check, not a new hypothesis. |
| **H6** | **keep (parked)** | ? | **L–M** | **M** | TRITON_ATTN is quality-gated (FLASH garbles gpt-oss). Possibility of a *correct* faster XPU flash path is unknown. Do not chase until a non-garbage alternative exists. |
| **H7** | **keep** | Y | **M** | **M** | TP=8/12 idle tiles + conservative CCL knobs are real. Unlikely to explain orders-of-magnitude gap; plausible for moderate warm tok/s. Needs warm TP=8 vs 12. |
| **H8** | **keep (later)** | Y | **H** (narrative) / **L** (fixes BS=1) | **M** | Correct: BS=1 offline is worst-case for MoE. Does not fix single-stream latency; serve bench is PERF_PLAN P4, not P0/P1. |
| **H9** | **keep** | Y | **H** | **S** | Already evidenced by code + equal `ttft_s`/`wall_s`. High for *interpretation*; low for hardware. Must stay first so A3 does not optimize against a fake TTFT. Warm bench should expose true TTFT/decode split. |
| **H10** | **keep → merge** (JIT family) | Y | **M** | **S** | `reduce_segments` JIT-during-inference warn supports residual shape compile. Likelihood of large warm hit is medium; warm2 JIT-warn counts decide keep vs demote. |
| **H11** | **keep (parked)** | ? | **L–M** | **L** | Log line `MoEPrepareAndFinalizeNoDPEPModular` is real; EP/TP MoE gaps on XPU are plausible. Too speculative without warm time-split profile. |
| **H12** | **kill** (for generate KPI) | Y (UX) | **L** (tok/s) / **M** (job wall UX) | **S** | Load/prefetch/shm stalls are outside generate metrics. Valid UX note; **not** a chase for 0.37 e2e tok/s. |

Legend: Possible = physically/logically coherent (Y/N/?). Likelihood = chance this is a *material* contributor to the observed pain (H/M/L). Effort = S/M/L to *test* (not to fix).

---

## 3. Verified set (keep + H/M likelihood) — ranked for A3

Max ~8. **Measurement-first ordering**; impact notes in parentheses. Items that still need warm data are marked.

| Rank | ID(s) | Why chase | Warm data? |
|------|-------|-----------|------------|
| **1** | **H9** | Sanitize TTFT vs wall vs decode before any optimization narrative | Helpful; already mostly proven from code/PASS |
| **2** | **H2+H5+H10** (JIT/cache family) | Quantify cold vs warm vs warm2; residual re-JIT; cross-job cache effect | **Required** — awaiting **8680399** |
| **3** | **H1** | Largest likely *steady-state* algorithmic gap (REF MoE); quality-gated | **Required** before declaring MoE dominates |
| **4** | **H3** | PVC MXFP4→BF16/FP16 dequant ceiling; interpret MoE A/B results | Warm compute ceiling useful; structural claim already Y |
| **5** | **H4** | Eager / no graphs — plausible decode overhead once kernels hot | **Required** |
| **6** | **H7** | TP=8→12 / CCL — moderate upside, clear A/B | **Required** |
| **7** | **H8** | Serve/continuous-batch realism for production tok/s story (P4) | Serve bench, not just warm single-stream |
| **8** | **H6** (conditional) | Only if a quality-safe non-TRITON attn candidate appears | Warm A/B + quality gate |

**Top 3 to chase first (after / with P0 warm):** **H9 → JIT family (H2+H5+H10) → H1** (with H3 as paired context, not a separate experiment wave).

---

## 4. Do not chase yet

| Item | Why park |
|------|----------|
| **H6** as primary lever | No known correct faster attn for gpt-oss on this stack; FLASH previously garbled |
| **H11** (EP / modular MoE prepare) | Needs warm layer-time profile first; speculative |
| **H12** for tok/s KPI | Outside generate path; UX-only |
| **Optimizing against Phase 5 “343 s TTFT” as true TTFT** | Metric artifact (H9); wait for bench_perf TTFT/decode fields |
| **Declaring “cold JIT fixed it” from SUCCESS_INFER alone** | Timed post-warmup still ~0.37 tok/s |
| **Fused MXFP4 MoE wins that reintroduce `!!!` / token-id-0** | Quality gate — FAIL regardless of speed |
| **OpenCL / dual `ONEAPI_DEVICE_SELECTOR`** | Known SEGV on vLLM MP |
| **Full Torch/IPEX/vLLM stack rebuild** | Out of scope per PERF_PLAN / team default |
| **Blackwell-class native FP4 expectations on PVC** | Hardware cannot deliver; do not frame H3 fixes as TC parity |
| **H4 graph/eager experiments before warm + MoE clarity** | Confounds + prior XPU compile pain |
| **H7/H8 as explanations of the 0.37 number** | May help later; will not explain orders of magnitude alone |
| **Any A/B until 8680399 warm/warm2 lands** | Cold Phase 5 metrics confound H1–H5/H7/H10 |

---

## 5. Merges

| Merge | Members | Family name | How to treat |
|-------|---------|-------------|--------------|
| **M1** | **H2 + H5 + H10** | **JIT / cache family** | One measurement wave via cold/warm/warm2 + cache persistence + JIT-warn counts. Do not open three separate solution tracks until Δs are known. |
| **M2** | **H1 + H3** | **MoE / MXFP4 compute family** | One investigation wave: REF vs fused vs mxfp4_fp8 under quality gate, interpreted against PVC dequant ceiling. H3 alone is not an experiment; it is the physics context for H1 results. |
| — | H9 | (standalone) | Hygiene prerequisite for all families — not merged into JIT or MoE. |
| — | H4, H7, H8 | (standalone) | Keep separate; lower priority than M1/M2. |

A1’s suggested order (H9+H2+H5 → H1+H3 → H4 → …) is **accepted** with the explicit merges above.

---

## 6. Warm-baseline dependency (8680399)

| Status | Detail |
|--------|--------|
| Job | **8680399** — `gpt-oss-b*` on debug, **queued** as of A2 review |
| Instrument | `bench_perf.py` cold / warm / warm2 → `PERF_JSON` |
| Blocks confirmation of | H1, H2, H3 (compute ceiling), H4, H5 (in-process vs residual), H7, H10, H11 |
| Does not block | H9 (already evidenced); H12 kill-for-KPI; structural half of H3 (no FP4 TC); quality gates |

**A3 should not propose irreversible recipe changes until warm numbers exist;** measurement and ranking refinements are fine.

---

## 7. A2 → A3 handoff summary

- **Verified set IDs:** H9, H2+H5+H10 (M1), H1, H3 (with H1 as M2), H4, H7, H8, H6(conditional)  
- **Killed / parked:** H12 (tok/s), H11 (until profile), H6 (until correct alt attn)  
- **Top 3 to chase first:** **H9 → M1 (H2+H5+H10) → H1** (+ H3 context)  
- **Blocking data:** warm baseline job **8680399** still queued  
