# A4 — Solution Review: gpt-oss-120b XPU perf (Aurora)

**Agent:** A4 (Solution Reviewer)  
**Date:** 2026-07-18  
**Input:** `A3_solutions.md` (full); `A2_review.md` (full); skim `bench_perf.py`, `bench_perf.pbs`, `infer_chat.pbs` (`infer_serve.pbs` absent)  
**Constraint:** Approve / defer / reject for A5 — no code edits, no job submits.

**Warm-baseline gate:** Job **8680399** may still be queued. Approve *implementation of experiment harness* now; do **not** treat MoE / eager / TP outcomes as decided, and do **not** flip the default PASS recipe until cold/warm/warm2 `PERF_JSON` lands.

---

## Hard-constraint check (A3 overall)

| Constraint | A3 status | A4 |
|------------|-----------|-----|
| Quality gate: no all-`!` / token-id-0 for tok/s | Explicit in S3–S7 | **Pass** |
| No OpenCL in `ONEAPI_DEVICE_SELECTOR` | L0-only called out | **Pass** |
| No full torch/IPEX/vLLM rebuild (unless exceptional + labeled) | S3 exceptional correctly labeled OOS | **Pass** |
| Prefer extending Phase 0 tools over parallel forks | Extends `bench_perf.py` / clones PBS; no rival KPI path | **Pass** (see S6 note) |
| Warm 8680399: no irreversible MoE/eager/TP as “done” | Sequence waits on P0; harness OK | **Pass** |

A3’s catalog is coherent with A2’s verified set (H9 → M1 → M2 → H4 → H7 → H8 → H6-parked). No hard-constraint violations that force a wholesale reject.

**Note for A5:** `bench_perf.py` `run_metrics` has the **same** `ttft_s = wall_s` fallback as `one_chat.py` — S1 must fix **both**, not only Phase 5 chat.

---

## 1. Per-solution review

### S1 — Metric hygiene (stop optimizing fake TTFT)

| Field | Value |
|-------|--------|
| **ID** | **S1** |
| **Verdict** | **approve** |
| **Maps** | H9 |

**Risk notes:** Script-only; zero hardware risk. Mis-labeling `ttft_source` would confuse A6 narrative — keep enum clear (`engine` vs `fallback_wall`). Do not invent TTFT from wall.

**Conditions for A5:**
1. Fix `one_chat.py` `metrics_from_output`: when `first_token_latency` missing/≤0, set `ttft_s: null` (or omit), `ttft_source: "fallback_wall"`, always keep `wall_s` / `e2e_tok_s`.
2. Fix the **same** fallback in `bench_perf.py` `run_metrics`; enrich each run + top-level `PERF_JSON` with `ttft_source` (and keep per-run `wall_s`).
3. Do **not** change PASS env / MoE / attn / TP as part of S1.
4. Docs: one-line annotation that Phase 5 “343 s TTFT” is **e2e wall**, not first-token (optional comment in `SUCCESS_INFER.md` or PERF_PLAN — keep minimal).

---

### S2 — JIT / cache family (cold vs warm vs cross-job)

| Field | Value |
|-------|--------|
| **ID** | **S2** |
| **Verdict** | **approve** |
| **Maps** | M1 = H2 + H5 + H10 |

**Risk notes:** Lustre-backed Triton/SYCL dirs can collide across concurrent jobs (rare bad kernels). **`SYCL_CACHE_PERSISTENT` must stay unset** (SEGV under TP). Cross-job win is measurement, not a guaranteed tok/s fix for warm2.

**Conditions for A5:**
1. **Keep** current per-job `$TMPDIR/..._${JOBTAG}` behavior as the **default** in `bench_perf.pbs` (Phase 0 / 8680399 continuity).
2. Add **`bench_perf_persist.pbs`** (clone of `bench_perf.pbs`) with durable:
   - `TRITON_CACHE_DIR=$WORKDIR/.cache/triton_xpu_gptoss`
   - `SYCL_CACHE_DIR=$WORKDIR/.cache/sycl_xpu_gptoss`
   - `unset SYCL_CACHE_PERSISTENT`
3. Optionally align `infer_chat.pbs` cache dirs to the same durable paths **or** document chat vs bench divergence — prefer durable for chat once persist PBS exists; still never set `SYCL_CACHE_PERSISTENT`.
4. Measurement-only H10: grep/count `reduce_segments` / `jit_monitor` / `JIT` in logs between cold→warm→warm2 (script helper or documented grep in PBS echo is fine). Optional fixed-shape dummy generate **only** behind an explicit flag (default off).
5. Primary KPI remains `warm2_*` + `quality_ok`; do not declare “JIT fixed” from cold alone.

---

### S3 — MoE path A/B (+ MXFP4 ceiling context)

| Field | Value |
|-------|--------|
| **ID** | **S3** |
| **Verdict** | **approve (harness only)** |
| **Maps** | M2 = H1 + H3 |

**Risk notes:** Fused / mxfp4_fp8 historically → all-`!` / token-id-0. Log may still say `Using XPUExpertsMxFp4` under REF — trust env + `moe_mode` in `PERF_JSON`. H3 is PVC dequant ceiling, **not** Blackwell FP4 parity. Exceptional kernel/stack rebuild correctly OOS — **reject** if A5 expands into unlabeled rebuild.

**Conditions for A5:**
1. Implement harness **now**: extend existing `bench_perf.py` / PBS pattern (env-driven MoE + `--moe-mode` label already present). Prefer **one** parameterized PBS or thin clones `bench_perf_moe_{ref,fused,fp8}.pbs` — not a second Python bench.
2. **Default recipe stays REF:** `VLLM_XPU_FUSED_MOE_USE_REF=1`; fused/fp8 only when PBS/env explicitly unset REF / set MXFP4_FP8.
3. **Do not submit** fused/fp8 jobs as the “winning” path, and **do not** change default `bench_perf.pbs` / `infer_chat.pbs` away from REF, until **8680399** (or equivalent) `PERF_JSON` is ingested **and** A6 can compare warm/warm2 under `quality_ok`.
4. Any `quality_ok=false` run is **FAIL** regardless of tok/s; discard from ranking.
5. Compare warm/warm2 across modes only (cold confounded by JIT). Frame wins as less REF overhead / better fusion, not native FP4 TC.

---

### S4 — Eager / graphs (decode launch overhead)

| Field | Value |
|-------|--------|
| **ID** | **S4** |
| **Verdict** | **approve (harness only)** |
| **Maps** | H4 |

**Risk notes:** Graph-on can SEGV/hang/OOM/wrong text on this XPU stack. Expected gain ~10–20%, not 2×. A2: do not chase graphs before warm + MoE clarity for *interpretation*; harness coding is OK now.

**Conditions for A5:**
1. Add `--enforce-eager {true,false}` to `bench_perf.py`; **default `true`** (PASS).
2. First graph-on trial PBS must keep `TORCHDYNAMO_DISABLE=1` and `TORCH_COMPILE_DISABLE=1`.
3. Optional second-wave unset of Dynamo/compile disables: **defer job submit** until S4 trial #1 is stable + quality_ok; still no rebuild.
4. Do **not** flip default `enforce_eager` in `one_chat.py` / PASS PBS until A6 reviews warm2 under quality_ok.
5. Submit graph-on jobs only **after** P0 warm ingest (and preferably after S3 REF warm baseline is clear).

---

### S5 — TP 8 → 12 (full-node tiles)

| Field | Value |
|-------|--------|
| **ID** | **S5** |
| **Verdict** | **approve (harness only)** |
| **Maps** | H7 |

**Risk notes:** OOM / CCL hang at TP=12; comm may worsen latency. Will not alone explain 0.37 e2e tok/s. Moderate upside only.

**Conditions for A5:**
1. Add `--gpu-memory-utilization` to `bench_perf.py` (default `0.82`); clone `bench_perf_tp12.pbs` with `--tp 12 --moe-mode ref` and PASS env otherwise.
2. Keep `CCL_WORKER_COUNT=1`, `CCL_ZE_IPC_EXCHANGE=sockets`, `disable_custom_all_reduce=True`.
3. **Do not submit** TP=12 as default/PASS until warm TP=8 baseline exists; harness may land now.
4. Compare same MoE/attn/eager recipe; KPI = `warm2_e2e_tok_s` / `warm2_ttft_s` / `quality_ok`.

---

### S6 — Serve / continuous-batch realism

| Field | Value |
|-------|--------|
| **ID** | **S6** |
| **Verdict** | **defer** |
| **Maps** | H8 |

**Risk notes:** Large surface (new `infer_serve.pbs` + `bench_serve.py`); OpenCL creep → MP SEGV; port/OOM at concurrency. Correct for PERF_PLAN P4, but too large for this A5 pass and must not replace offline `warm2_*` KPIs.

**Conditions to unpark later:** Offline warm KPIs exist; best quality-passing offline recipe known; then create serve from `infer_chat.pbs` L0/REF/Triton recipe + concurrent client bench with sample `quality_ok`.

---

### S7 — Attention backend (conditional / parked)

| Field | Value |
|-------|--------|
| **ID** | **S7** |
| **Verdict** | **defer** (remain parked) |
| **Maps** | H6 |

**Risk notes:** FLASH_ATTN previously garbled gpt-oss. No documented quality-safe non-TRITON candidate on this stack.

**Conditions to unpark:** A4/A5 cite a **documented** XPU backend that produced coherent gpt-oss text here; then CLI `--attn` A/B with forced `quality_ok`. Until then: keep `TRITON_ATTN`; **reject** any FLASH re-enable as a perf lever.

---

## 2. Approved implement set for A5 (ordered)

Code and submit what is listed. Jobs that need warm baseline are marked **gated**.

| Order | ID | Scope | Files to change / add | Jobs to submit |
|-------|-----|--------|------------------------|----------------|
| **1** | **S1** | Full | `one_chat.py` (`metrics_from_output`); `bench_perf.py` (`run_metrics` + `PERF_JSON` / `runs.*` `ttft_source`, null TTFT on fallback); optional one-line SUCCESS_INFER/PERF_PLAN note | None required (hygiene). Re-run chat smoke only if convenient. |
| **2** | **S2** | Full (default cache unchanged) | Add `bench_perf_persist.pbs`; keep `bench_perf.pbs` per-job TMPDIR; optional `infer_chat.pbs` durable cache align; optional JIT-warn count helper / PBS log greps | After code: `qsub bench_perf_persist.pbs` when ready for H5 cross-job check (**gated** if 8680399 still Q — prefer ingest P0 first, then persist cold vs prior warm). |
| **3** | **S3** | **Harness only** | Thin PBS clones or one parameterized MoE PBS (`ref` default / `fused` / `mxfp4_fp8`); ensure env + `--moe-mode` recorded; no new parallel Python KPI tool | **Gated:** submit MoE A/B/C only after 8680399 `PERF_JSON` ingested. Default `bench_perf.pbs` stays REF. |
| **4** | **S4** | **Harness only** | `bench_perf.py`: `--enforce-eager` default `true`; optional `bench_perf_eager_false.pbs` with Dynamo/compile still disabled | **Gated:** graph-on submit after P0 (+ preferably S3 REF warm clarity). |
| **5** | **S5** | **Harness only** | `bench_perf.py`: `--gpu-memory-utilization` default `0.82`; add `bench_perf_tp12.pbs` (`--tp 12`, REF) | **Gated:** TP=12 submit after warm TP=8 baseline on same recipe. |

**Not in this A5 set:** S6 (defer), S7 (parked).

---

## 3. What A5 must NOT do

1. **Accept** all-`!` / token-id-0 / `quality_ok=false` runs as tok/s wins.
2. Put **OpenCL** (or `*:gpu`) in `ONEAPI_DEVICE_SELECTOR` — L0 only.
3. Enable **`SYCL_CACHE_PERSISTENT=1`**.
4. Start a **full torch / IPEX / vLLM rebuild** or custom fused-MXFP4 kernel fix (exceptional — needs new A4 escalate).
5. Treat MoE / eager / TP experiment results as **done** or flip **default PASS** (`infer_chat.pbs` / `bench_perf.pbs` REF + eager + TP=8) without 8680399 (or equivalent) warm/warm2 baseline.
6. Fork a **parallel** KPI path that replaces `bench_perf.py` `PERF_JSON` (no rival smoke metric as ranking source).
7. Implement **S6** (`infer_serve.pbs` / `bench_serve.py`) or unpark **S7** / re-enable **FLASH_ATTN** in this pass.
8. Frame H3 / fused wins as **native FP4 TC** parity with Blackwell.
9. Chase Phase 5 **“343 s TTFT”** as true first-token after S1 lands.
10. Submit irreversible recipe changes while **8680399** is still the only pending P0 and has not produced `PERF_JSON`.

---

## 4. Success criteria for A6 review

A6 should accept this cycle if:

1. **S1 shipped:** missing engine TTFT no longer masquerades as TTFT (`ttft_source` present; `ttft_s` null/omitted on fallback); `PERF_JSON` remains the ranking KPI.
2. **S2 shipped:** default bench still per-job TMPDIR; persist PBS exists with durable caches and **unset** `SYCL_CACHE_PERSISTENT`; JIT-warn observation path documented or scripted.
3. **S3–S5 harness shipped:** CLI/PBS for MoE modes, `--enforce-eager`, TP=12 / `gpu_memory_utilization` — all **default to PASS** (REF, eager true, TP=8, util 0.82).
4. **No** quality-gate regressions in default PBS/scripts; no OpenCL selector; no stack rebuild; no FLASH.
5. If any gated jobs ran: A6 gets comparable `PERF_JSON` rows with `quality_ok`, `moe_mode`, warm/warm2 fields, and narrative that does **not** claim irreversible recipe victory without P0 baseline.
6. **Deferred** S6/S7 explicitly left undone (not half-implemented).

**A6 fail if:** default recipe silently moved to fused/graphs/TP12; TTFT still equals wall without `ttft_source`; OpenCL/`SYCL_CACHE_PERSISTENT` reintroduced; quality-fail runs ranked as wins; full rebuild attempted.

---

## A4 → A5 handoff

**Ordered approved S-ids:** **S1 → S2 → S3(harness) → S4(harness) → S5(harness)**  
**Deferred:** **S6**, **S7**  

**A5 brief:** Implement S1 metric hygiene in `one_chat.py`+`bench_perf.py`, then S2 persist PBS (keep default TMPDIR caches), then MoE/eager/TP harness flags+PBS defaulting to PASS REF/eager/TP8 — submit gated A/B jobs only after 8680399 `PERF_JSON`; do not touch S6/S7 or rebuild.
