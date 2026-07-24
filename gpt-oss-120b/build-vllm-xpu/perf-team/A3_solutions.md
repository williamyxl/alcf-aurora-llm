# A3 — Solutions: gpt-oss-120b XPU perf (Aurora)

**Agent:** A3 (Solution Designer)  
**Date:** 2026-07-18  
**Input:** `A2_review.md` verified set; `A1_hypotheses.md`; PASS recipes + `PERF_PLAN.md`  
**Constraint:** Solutions only — no code edits, no job submits (A5 implements after A4 approval).

**Warm-baseline gate:** Job **8680399** may still be queued. Do **not** make irreversible PASS-recipe changes until cold/warm/warm2 `PERF_JSON` lands. Measurement hygiene (S1) and cache instrumentation (S2) may proceed in parallel with awaiting P0.

**Quality / safety gates (all solutions):**
- `quality_ok=true` required (no all-`!`, no token-id-0)
- `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` only — never OpenCL / `*:gpu`
- Prefer env / flags / PBS / script changes over kernel rewrites
- Full torch/IPEX/vLLM rebuild out of scope (exceptional only, labeled below)

**Instrument:** `bench_perf.py` / `bench_perf.pbs` → cold / warm / warm2 + `PERF_JSON` + `quality_ok`

---

## Solution catalog

### S1 — Metric hygiene (stop optimizing fake TTFT)

| Field | Value |
|-------|--------|
| **Solution ID** | **S1** |
| **Maps to** | **H9** |
| **Effort** | **S** |
| **Order** | **1 (first)** |

**Concrete change**

1. Treat `bench_perf.py` `PERF_JSON` as the only KPI source for ranking (not Phase 5 `METRICS_JSON`).
2. In `one_chat.py` `metrics_from_output` (`workdir/llm/gpt-oss-120b/one_chat.py`): when `first_token_latency` is missing, **do not** set `ttft_s = wall_s`. Emit:
   - `ttft_s: null` (or omit)
   - `ttft_source: "fallback_wall"` vs `"engine"`
   - always keep `wall_s` and `e2e_tok_s`
3. Optionally enrich `bench_perf.py` `PERF_JSON` with per-run `wall_s` and `ttft_source` under `runs.cold|warm|warm2` (fields already partly present).
4. Docs: annotate Phase 5 “343 s TTFT” as **e2e wall**, not first-token.

**Expected effect + measure**

- Effect: correct ranking of cold JIT vs steady-state; prevents chasing a fake TTFT.
- Measure: after 8680399, require `warm_ttft_s` / `warm_decode_tok_s` / `warm_e2e_tok_s` populated separately where engine stats exist; if `ttft_s ≈ wall_s` still, flag `ttft_source=fallback` in narrative — do not call it TTFT.

**Risk / rollback**

- Risk: none for hardware (script-only).
- Rollback: revert `one_chat.py` / `bench_perf.py` metric fields.

---

### S2 — JIT / cache family (cold vs warm vs cross-job)

| Field | Value |
|-------|--------|
| **Solution ID** | **S2** |
| **Maps to** | **M1 = H2 + H5 + H10** |
| **Effort** | **S–M** |
| **Order** | **2 (with / right after S1)** |

**Concrete change** (PBS / env only; keep `SYCL_CACHE_PERSISTENT` **unset**)

**A. Within-job stability (already partially in `bench_perf.pbs`)** — keep:

```bash
JOBTAG=${PBS_JOBID:-$$}
export SYCL_CACHE_DIR=$TMPDIR/sycl_cache_bench_${JOBTAG}
export TRITON_CACHE_DIR=$TMPDIR/triton_cache_bench_${JOBTAG}
mkdir -p "$SYCL_CACHE_DIR" "$TRITON_CACHE_DIR"
unset SYCL_CACHE_PERSISTENT   # PersistentDeviceCodeCache SEGV under TP
```

**B. Cross-job persistence (H5)** — change `infer_chat.pbs` / add variant `bench_perf_persist.pbs`:

```bash
# Lustre-backed shared caches (survive job end; node /tmp does not)
export TRITON_CACHE_DIR=$WORKDIR/.cache/triton_xpu_gptoss
export SYCL_CACHE_DIR=$WORKDIR/.cache/sycl_xpu_gptoss
mkdir -p "$TRITON_CACHE_DIR" "$SYCL_CACHE_DIR"
unset SYCL_CACHE_PERSISTENT
# Do NOT set SYCL_CACHE_PERSISTENT=1 (known SEGV with CCL/TP)
```

Align `infer_chat.pbs` (today: `TRITON_CACHE_DIR=$TMPDIR/triton_cache_$$`) with the same durable dirs once S2 is approved.

**C. Residual shape JIT (H10)** — measurement only in first pass:

- Grep job log for `reduce_segments` / `jit_monitor` / `JIT` between cold → warm → warm2.
- Optional A5 follow-up: fixed `max_tokens` + one dummy prefill length cover in a short warmup generate before timed runs (no new kernel APIs).

**Expected effect + measure**

| Claim | PERF_JSON / log check |
|-------|------------------------|
| Cold JIT dominates first generate | `cold_*` ≫ `warm_*` (esp. `*_ttft_s` / `*_e2e_tok_s`) |
| Steady-state residual | `warm_*` ≈ `warm2_*` |
| Cross-job cache win | Job N+1 `cold_*` closer to prior `warm_*` when dirs persist |
| Re-JIT still active | warm2 log still shows JIT-during-inference warns |

Primary KPI after hot kernels: **`warm2_e2e_tok_s`** / **`warm2_decode_tok_s`** with `quality_ok=true`.

**Risk / rollback**

- Risk: Lustre cache corruption / permission conflicts across jobs → rare bad kernels; **do not** enable `SYCL_CACHE_PERSISTENT` (SEGV).
- Rollback: revert to per-job `$TMPDIR/..._${JOBTAG}` dirs as in current `bench_perf.pbs`; delete `$WORKDIR/.cache/triton_xpu_gptoss` / `sycl_xpu_gptoss` if suspect.

---

### S3 — MoE path A/B under quality gate (+ MXFP4 ceiling context)

| Field | Value |
|-------|--------|
| **Solution ID** | **S3** |
| **Maps to** | **H1 + H3** (A2 merge **M2**) |
| **Effort** | **M** |
| **Order** | **3 (after P0 warm + S1/S2)** |

**Concrete change** — env-driven MoE modes via existing `bench_perf.py --moe-mode` label + env (no new kernel APIs):

| Exp | Env | CLI label |
|-----|-----|-----------|
| **A (baseline)** | `VLLM_XPU_FUSED_MOE_USE_REF=1` | `--moe-mode ref` |
| **B (fused)** | `unset VLLM_XPU_FUSED_MOE_USE_REF` (and unset MXFP4_FP8) | `--moe-mode fused` |
| **C (mxfp4_fp8)** | `unset REF`; `VLLM_XPU_FUSED_MOE_USE_MXFP4_FP8=1` | `--moe-mode mxfp4_fp8` |

PBS snippet pattern (clone `bench_perf.pbs` → `bench_perf_moe_{ref,fused,fp8}.pbs`):

```bash
# Exp A — PASS quality path
export VLLM_XPU_FUSED_MOE_USE_REF=1
unset VLLM_XPU_FUSED_MOE_USE_MXFP4_FP8
mpiexec -n 1 --ppn 1 python bench_perf.py --tp 8 --max-tokens 128 --moe-mode ref

# Exp B — fused MXFP4 (quality gate critical)
unset VLLM_XPU_FUSED_MOE_USE_REF
unset VLLM_XPU_FUSED_MOE_USE_MXFP4_FP8
mpiexec -n 1 --ppn 1 python bench_perf.py --tp 8 --max-tokens 128 --moe-mode fused

# Exp C — MXFP4→FP8 recipe
unset VLLM_XPU_FUSED_MOE_USE_REF
export VLLM_XPU_FUSED_MOE_USE_MXFP4_FP8=1
mpiexec -n 1 --ppn 1 python bench_perf.py --tp 8 --max-tokens 128 --moe-mode mxfp4_fp8
```

**H3 interpretation (not a separate flag flip):** PVC has **no native FP4 TC**. Even a quality-passing fused path is dequant→BF16/FP16 (or FP8 upcast) compute. Frame wins as “less REF overhead / better fusion,” **not** Blackwell FP4 parity. If fused/fp8 quality-PASS but `warm2_e2e_tok_s` still low, treat H3 ceiling as binding and stop chasing FP4 TC myths.

Log check (A2 note): under REF, log may still say `Using XPUExpertsMxFp4` — record env + `moe_mode` in `PERF_JSON`; do not trust log string alone.

**Expected effect + measure**

- Effect if H1 true: Exp B or C raises **`warm2_e2e_tok_s` / `warm2_decode_tok_s`** vs Exp A by a large factor (not 10–20%).
- Gate: `quality_ok` must be true for all of cold/warm/warm2; else experiment = **FAIL** regardless of speed.
- Compare only **warm/warm2** across MoE modes (cold confounded by JIT).

**Risk / rollback**

- Risk: fused / mxfp4_fp8 → all-`!` / token-id-0 (historically confirmed); possible SEGV/OOM unlikely but watch.
- Rollback: restore `VLLM_XPU_FUSED_MOE_USE_REF=1`; discard FAIL `PERF_JSON` from recipe.

**Exceptional (out of scope unless A4 escalates):** custom fused MXFP4 kernel fix / stack rebuild — label **exceptional**, not default S3.

---

### S4 — Eager / graphs (decode launch overhead)

| Field | Value |
|-------|--------|
| **Solution ID** | **S4** |
| **Maps to** | **H4** |
| **Effort** | **M** |
| **Order** | **4 (after S3 clarity on warm MoE)** |

**Concrete change**

1. Extend `bench_perf.py` LLM ctor with CLI `--enforce-eager {true,false}` (default `true` = PASS).
2. Keep `TORCHDYNAMO_DISABLE=1` and `TORCH_COMPILE_DISABLE=1` in PBS for first graph-on trial (vLLM graphs without full inductor — safer on this XPU stack).
3. PBS A/B:

```bash
# Baseline (PASS)
# LLM(..., enforce_eager=True)  via --enforce-eager true

# Trial
# LLM(..., enforce_eager=False) via --enforce-eager false
# Still:
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
```

4. Only if trial is stable + quality_ok: optional second wave unset Dynamo/compile disables (higher risk; still **no** full rebuild).

**Expected effect + measure**

- Effect: literature suggests ~10–20% on XPU when graphs help — **not** 2×. Look at **`warm2_decode_tok_s`** and **`warm2_e2e_tok_s`**.
- Confirm log: `CUDAGraph memory` > 0 when graphs engage (PASS had `0.0 GiB`).

**Risk / rollback**

- Risk: SEGV / hang / wrong text on XPU compile paths; OOM if graph memory grows.
- Rollback: `--enforce-eager true` + restore Dynamo/compile disable exports.

---

### S5 — TP 8 → 12 (full-node tiles)

| Field | Value |
|-------|--------|
| **Solution ID** | **S5** |
| **Maps to** | **H7** |
| **Effort** | **M** |
| **Order** | **5** |

**Concrete change**

```bash
# Clone bench_perf.pbs → bench_perf_tp12.pbs
# Keep REF + TRITON_ATTN + L0 recipe; only change TP:
mpiexec -n 1 --ppn 1 python bench_perf.py --tp 12 --max-tokens 128 --moe-mode ref
```

If OOM: lower `gpu_memory_utilization` in `bench_perf.py` (PASS uses `0.82`) to e.g. `0.75` via a new `--gpu-memory-utilization` flag (A5). Keep `CCL_WORKER_COUNT=1`, `CCL_ZE_IPC_EXCHANGE=sockets`, `disable_custom_all_reduce=True`.

**Expected effect + measure**

- Effect: moderate tok/s / latency change — **not** expected to fix 0.37 alone.
- Measure: same MoE/attn/eager; compare `n_tiles=8` vs `12` on **`warm2_e2e_tok_s`**, `warm2_ttft_s`, `quality_ok`.

**Risk / rollback**

- Risk: OOM / CCL hang at TP=12; worse latency if comm dominates.
- Rollback: `--tp 8`.

---

### S6 — Serve / continuous-batch realism (not BS=1 smoke)

| Field | Value |
|-------|--------|
| **Solution ID** | **S6** |
| **Maps to** | **H8** |
| **Effort** | **M** |
| **Order** | **6 (PERF_PLAN P4; after offline warm KPIs exist)** |

**Concrete change**

`infer_serve.pbs` is not present yet — create from `infer_chat.pbs` env recipe:

1. **Server PBS** (`infer_serve.pbs`): same L0 / REF / Triton / CCL env as PASS; launch `vllm serve` (or project-equivalent OpenAI API entry) with TP=8 (then best of S3/S5), `attention_backend` via vLLM CLI/`VLLM_*` as supported by this build, `enforce_eager` matching best offline recipe.
2. **Client bench** (`bench_serve.py`): concurrent clients N∈{1,4,8}, same MOF prompt, report **aggregate** tok/s + per-request latency; still enforce `quality_ok` on samples.
3. Do **not** replace offline `warm2_*` KPIs with serve aggregate in P0/P1 narratives.

**Expected effect + measure**

- Effect: production tok/s story improves via continuous batching; **BS=1 latency may stay similar**.
- Measure: serve JSON (aggregate tok/s, p50/p95 latency) **plus** spot `quality_ok`; contrast with offline `PERF_JSON` BS=1 `warm2_e2e_tok_s`.

**Risk / rollback**

- Risk: MP SEGV if OpenCL selector creeps in; port conflicts; OOM at higher concurrency (`max_num_seqs`).
- Rollback: stop serve job; fall back to `bench_perf.py` offline recipe.

---

### S7 — Attention backend (conditional / parked)

| Field | Value |
|-------|--------|
| **Solution ID** | **S7** |
| **Maps to** | **H6 (conditional)** |
| **Effort** | **M** (when unparked) |
| **Order** | **7 — only if a quality-safe non-TRITON candidate appears** |

**Concrete change (parked by default)**

- Keep PASS: `attention_backend="TRITON_ATTN"` in `bench_perf.py` / `one_chat.py`.
- **Do not** re-enable `FLASH_ATTN` as a perf lever (previously garbled gpt-oss).
- Unpark only if A4/A5 identify a **documented** XPU backend that previously produced coherent gpt-oss text on this stack; then A/B via CLI `--attn` (to add) with forced `quality_ok` gate:

```bash
# Hypothetical — only when candidate is known quality-safe
mpiexec -n 1 --ppn 1 python bench_perf.py --tp 8 --attn <CANDIDATE> ...
```

**Expected effect + measure**

- Effect: unknown; possible warm decode gain if a correct faster path exists.
- Measure: `warm2_*` + `quality_ok`; FAIL if garbage.

**Risk / rollback**

- Risk: garbled output (known for FLASH); high.
- Rollback: `attention_backend="TRITON_ATTN"`.

---

## Recommended implementation sequence (A4 → A5)

Ordered list for approval / implementation. **Gate:** ingest 8680399 `PERF_JSON` before irreversible MoE/eager/TP recipe changes.

1. **S1** — Metric hygiene in `one_chat.py` / `PERF_JSON` narrative (no hardware risk).
2. **S2** — Durable Triton/SYCL cache dirs + JIT-warn counting; keep `SYCL_CACHE_PERSISTENT` unset.
3. **Wait / ingest P0** — Job **8680399** cold/warm/warm2; refine whether residual is MoE-dominated (`warm≈warm2` but still slow) vs JIT.
4. **S3** — MoE A/B/C (`ref` / `fused` / `mxfp4_fp8`) under `quality_ok`; interpret with H3 PVC dequant ceiling.
5. **S4** — `enforce_eager=False` with Dynamo/compile still disabled; then optional compile-enable wave.
6. **S5** — TP=12 vs TP=8 on best quality-passing MoE/eager recipe.
7. **S6** — Serve + concurrent clients for production aggregate tok/s (P4).
8. **S7** — Attn A/B **only if** a quality-safe candidate appears; else remain parked.

**Do not implement (A2 parked/killed):** H11 EP/modular MoE (needs profile first); H12 for tok/s KPI; OpenCL selector; full stack rebuild; native-FP4 expectations on PVC.

---

## Quick reference — flags used

| Lever | Mechanism |
|-------|-----------|
| MoE REF | `VLLM_XPU_FUSED_MOE_USE_REF=1` |
| MoE mxfp4_fp8 | `VLLM_XPU_FUSED_MOE_USE_MXFP4_FP8=1` (REF unset) |
| MoE fused | both unset |
| TP | `bench_perf.py --tp {8,12}` |
| Eager | `LLM(..., enforce_eager=True/False)` |
| Attn | `attention_backend="TRITON_ATTN"` (FLASH parked) |
| Caches | `TRITON_CACHE_DIR`, `SYCL_CACHE_DIR`; **unset** `SYCL_CACHE_PERSISTENT` |
| Selector | `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` |

---

## A3 → A4 handoff

**Ordered S-ids for A4:** **S1 → S2 → S3 → S4 → S5 → S6 → (S7 conditional)**
