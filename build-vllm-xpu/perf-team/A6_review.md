# A6 — Implementation Review: gpt-oss-120b XPU perf (Aurora)

**Agent:** A6 (Implementation Reviewer)  
**Date:** 2026-07-18  
**Input:** `A4_review.md` §4 success criteria + “must NOT”; `A5_implement.md`; full review of listed A5 artifacts  
**Constraint:** Review only — no production code edits (PASS with nits only).

---

## 1. Verdict

**PASS**

A5 shipped S1–S5 as approved: metric hygiene, persist-cache PBS (default TMPDIR unchanged), MoE/eager/TP12 harnesses with PASS defaults intact. No S6/S7 half-work, no OpenCL / `SYCL_CACHE_PERSISTENT` / FLASH / rebuild, no experimental job claimed as recipe victory. Job **8680399** remains queued Phase 0 baseline.

---

## 2. Checklist (S1–S5 + gates)

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | **S1:** missing engine TTFT → `ttft_s` null/omitted + `ttft_source`; `wall_s`/`e2e` kept; no fake TTFT=wall | **PASS** | `one_chat.py` `metrics_from_output` L37–62: null + `fallback_wall` / `engine`; always `wall_s`/`e2e_tok_s`. `bench_perf.py` `run_metrics` L63–87 same; PERF_JSON L186–191 `*_ttft_source`; per-run `wall_s` in `runs.*`. |
| 2 | **S2:** default bench per-job TMPDIR; persist PBS durable; `SYCL_CACHE_PERSISTENT` unset; never OpenCL selector | **PASS** | `bench_perf.pbs` L45–49 still `$TMPDIR/..._${JOBTAG}`. `bench_perf_persist.pbs` L48–49 durable `$WORKDIR/.cache/{triton,sycl}_xpu_gptoss`; L45 `unset SYCL_CACHE_PERSISTENT`; JIT greps L71–80. `infer_chat.pbs` L50–55 aligned durable + unset. All PBS: `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`. |
| 3 | **S3–S5 harness:** MoE/eager/TP12 PBS exist; defaults PASS (REF, eager true, TP=8, util 0.82) | **PASS** | MoE clones `bench_perf_moe_{ref,fused,fp8}.pbs`; `bench_perf_graphs.pbs` (`--enforce-eager false`, Dynamo/compile still disabled); `bench_perf_tp12.pbs` (`--tp 12`, REF, util 0.82). Defaults: `bench_perf.py` `--enforce-eager` default True, `--gpu-memory-utilization` 0.82, `--tp` 8; default/pass PBS keep `VLLM_XPU_FUSED_MOE_USE_REF=1`. `one_chat.py` still `enforce_eager=True`. |
| 4 | No S6/S7 half-impl; no FLASH_ATTN; no stack rebuild | **PASS** | No `infer_serve.pbs` / `bench_serve.py`. Attn remains `TRITON_ATTN` only (FLASH mentioned only as forbidden comment). No rebuild PBS/scripts touched. |
| 5 | Git allowlist covers new PBS | **PASS** | `.gitignore` L50–55 + `FILES.md` L34–39 list all new `bench_perf_*.pbs`. |
| 6 | `quality_ok` gate still present | **PASS** | `bench_perf.py` `quality_ok()` L44–55; aggregated `all_ok`; `SystemExit(2)` L211–212 on fail. |
| 7 | No experimental job submit claimed as recipe victory | **PASS** | A5: no persist/MoE/graphs/TP12 submits. `qstat`: only **8680399** (`gpt-oss-bench`) still **Q**. Defaults not flipped. |

---

## 3. Issues / nits

| Severity | Item |
|----------|------|
| — | **No blockers.** |
| nit | `ttft_source="fallback_wall"` with `ttft_s: null` matches A4 enum but can be misread as “wall used as TTFT.” Consumers must treat null/`fallback_wall` as *no engine TTFT*; ranking KPIs remain `warm2_e2e_tok_s` + `quality_ok` (+ `wall_s` when needed). |
| nit | `SUCCESS_INFER.md` still embeds historical METRICS_JSON with `ttft_s == wall_s` (pre-S1); the following note correctly re-labels it as e2e wall. Optional: refresh after next chat smoke. |
| nit | Default `bench_perf.pbs` relies on Python CLI defaults for `--enforce-eager` / `--gpu-memory-utilization` (does not pass them explicitly). Values match PASS; optional explicit flags for clarity. |
| nit | `.gitignore` / `FILES.md` still allowlist `infer_serve.pbs` “if present” — pre-existing slot, not an S6 half-implementation (file absent). |

---

## 4. FAIL → A5 fix requests

**None.** (Would list exact file/line patches here if verdict were FAIL.)

---

## 5. PASS — what remains gated

Do **not** flip default PASS recipe (`bench_perf.pbs` / `infer_chat.pbs`: REF + eager + TP=8) until baselines exist under `quality_ok`.

| Gate | Action | When |
|------|--------|------|
| **P0 / 8680399** | Ingest `PERF_JSON` (cold/warm/warm2, `quality_ok`, `moe_mode`, `*_ttft_source`) into `PERF.md` | Job leaves Q and completes |
| **S2 persist** | `qsub bench_perf_persist.pbs` — compare cross-job cold vs prior warm + JIT warn counts | After P0 ingest (prefer) |
| **S3 MoE A/B/C** | Submit `bench_perf_moe_{ref,fused,fp8}.pbs`; discard any `quality_ok=false` from ranking | After P0 ingest |
| **S4 graphs** | `qsub bench_perf_graphs.pbs` (Dynamo/compile still disabled) | After P0 (+ preferably S3 REF warm clear) |
| **S5 TP12** | `qsub bench_perf_tp12.pbs` | After warm TP=8 baseline on same recipe |
| **Deferred** | S6 serve / S7 attn | Remain parked per A4 |

---

## A6 → next handoff

**Verdict:** **PASS**  
**Ordered remaining work:** ingest **8680399** → optional persist → gated MoE → gated graphs → gated TP12.  
**Do not:** treat experiment jobs as default recipe wins without P0 warm/warm2 + `quality_ok`.
