# RESUME — performance work (session recovery)

**Paused:** 2026-07-20 (~18:00 UTC)  
**Resumed:** 2026-07-21 — P7 + TP scaling **done**; **TP=2 is best practice**; next = fused MoE quality.  
**Purpose of this file:** Enough context for a **new chat / new engineer / new agent** to continue without the prior conversation.

Stack bring-up (Phases **0–6**) is **CLOSED**. Only **performance** work is open.

---

## 0. Start here (60-second orientation)

| | |
|--|--|
| Workdir | `/lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b` |
| Conda env | `build-vllm-xpu/env` (Python 3.12) — **no** `module load frameworks` |
| Model | `models/openai-gpt-oss-120b` (MXFP4 MoE, ~60.8 GiB on disk) |
| Account / PBS | `-A MatSciAI`; typical queues `debug` / `debug-scaling` (max walltime **01:00:00**) |
| Git | repo `https://github.com/williamyxl/alcf-aurora-llm.git`, branch `main` |
| Last local commit | `1b8efdb` (RESUME/perf docs); branch may be synced with origin — verify `git status` |
| Push status | Check `git status -sb`; push if still ahead |
| Best practice | [`BEST_PRACTICE.md`](BEST_PRACTICE.md) — **TP=2**, warm2 ≈ **1.15** e2e tok/s |
| Active P7 job | **8681016** — **PASS** (engine TTFT); see PERF.md §P7 |
| Scaling jobs | TP=2 **8681063** / TP=4 **8681062** / TP=8 **8681016** — **COMPLETE** |
| Best BS=1 REF | **TP=2** warm2 e2e **1.15** tok/s ≫ TP=4 0.66 ≫ TP=8 0.37 (inverse scaling) |
| Prior chat | Cursor transcript: `482a0618-af7b-4bb9-bb9f-ac69719a9c03` under agent-transcripts (optional) |

**Next ordered work:**

1. ~~**P7**~~ — PASS  
2. ~~**TP=2/4/8 scaling**~~ — COMPLETE → `BEST_PRACTICE.md` + `SCALING_TP248.md`  
3. **Optimization** — fused MoE quality (`bench_perf_moe_fused{,_tp2,_tp4}.pbs`) → serve (P4) → long context (P6)

---

## 1. What already works (do not regress)

### Phase 5 inference PASS

- Artifact: `SUCCESS_INFER.md`  
- Job: **8680184**, host `x4303c1s3b0n0`  
- Coherent MOF isotherm text (not all-`!` / token-id-0)

### Phase 6 LoRA train PASS

- Artifact: `SUCCESS_TRAIN.md`  
- Adapter: `checkpoints/lora-smoke/adapter/`

### Best quality-passing *speed* recipe (still slow)

- Artifact: `SUCCESS_PERF.md`  
- Baseline job: **8680399** (`bench_perf.pbs`), warm2 **≈0.372 e2e tok/s**, `quality_ok=true`  
- Log backup: `build-vllm-xpu/logs/bench_perf.out.pre_p7_8680399`  
- **Not** a production speed win — REF MoE dominates steady-state

### Required runtime recipe (copy for every job)

| Setting | Value |
|---------|--------|
| Module | `oneapi/release/2025.3.1` only |
| `ONEAPI_DEVICE_SELECTOR` | `level_zero:gpu` (**never** OpenCL / `*:gpu` — SEGV in vLLM MP) |
| `TRITON_INTEL_DEVICE_EXTENSIONS` | `cl_intel_subgroup_matrix_multiply_accumulate cl_intel_subgroup_matrix_multiply_accumulate_tensor_float32 cl_intel_subgroup_2d_block_io cl_intel_bfloat16_conversions` |
| MoE | `VLLM_XPU_FUSED_MOE_USE_REF=1` |
| Attention | `TRITON_ATTN` |
| TP (default) | **8** |
| dtype | `bfloat16` |
| `enforce_eager` | `True` |
| `max_model_len` | `4096` (until P6) |
| Dynamo/compile | `TORCHDYNAMO_DISABLE=1`, `TORCH_COMPILE_DISABLE=1` |
| CCL | `CCL_WORKER_COUNT=1`, `CCL_ZE_IPC_EXCHANGE=sockets` |
| Triton patch | `build-vllm-xpu/patches/` — OpenCL-optional `driver.c` |

Scripts already encode this: `infer_chat.pbs`, `bench_perf.pbs`, `bench_perf_tp{2,4}.pbs`, etc.

---

## 2. Memory / TP facts (why TP=2/4 OOM’d)

Aurora tile HBM ≈ **64 GiB**. Measured weight load:

| TP | Weights / tile | Result without KV pin |
|----|----------------|------------------------|
| 8 | **8.2 GiB** | PASS; util planner reserves ~49 GiB KV |
| 4 | **16.3 GiB** | OOM at init (~63 GiB used) |
| 2 | **31.1 GiB** | OOM at init (~62 GiB used) |
| 1 | (earlier) | OOM |
| 12 | — | **Invalid** (`64 attn heads % 12 != 0`) |

**Root cause of OOM:** XPU memory profiler makes `available_KV ≈ gpu_memory_utilization × HBM` (negative `non_torch` cancels weights). Planner printed ~51 GiB KV even with 31 GiB weights → alloc dies.

**Fix (in PBS, not yet re-validated at scale):**

```bash
# already in bench_perf_tp2.pbs / bench_perf_tp4.pbs
python bench_perf.py --tp {2|4} ... --kv-cache-memory-gib 8 --max-num-seqs 2
```

`--kv-cache-memory-gib` sets `kv_cache_memory_bytes` and **bypasses** util-based KV sizing. Single-stream 4K needs ≪1 GiB KV; 8 GiB is ample.

Checkpoint size on Lustre: **~60.77 GiB**.

---

## 3. Metrics / P7 (GATE — incomplete validation)

### Problem

Offline `LLM()` defaults `disable_log_stats=True` → `RequestOutput.metrics is None` → no `first_token_latency`.

S1 correctly stopped faking TTFT as `wall_s`. So 8680399 reports:

- `ttft_s=null`, `ttft_source=fallback_wall`
- Only trustworthy rate: **e2e tok/s = n_out / wall_s**

Phase 5’s “343 s TTFT” was **e2e wall**, not first-token.

### Code already landed (commit `f60a2bb`)

| File | Change |
|------|--------|
| `bench_perf.py` | `disable_log_stats=False`; `prefill_tok_s` / `decode_tok_s` in metrics + `PERF_JSON`; CLI `--kv-cache-memory-gib`, `--max-num-seqs` |
| `one_chat.py` | `disable_log_stats=False` |

### Definitions (P7 acceptance)

| Field | Formula |
|-------|---------|
| `ttft_s` | engine `metrics.first_token_latency` |
| `ttft_source` | must be **`engine`** (not `fallback_wall`) |
| `prefill_tok_s` | `n_prompt_tokens / ttft_s` |
| `decode_tok_s` | `(n_out - 1) / (last_token_ts - first_token_ts)` |
| `e2e_tok_s` | `n_out / wall_s` (keep) |

### How to validate P7

```bash
cd /lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b
qstat -u xiaoliyan          # should be empty before submit if possible
qsub bench_perf.pbs         # queue: debug, ~1h wall
# log:
tail -f build-vllm-xpu/logs/bench_perf.out
```

**PASS when** log / `PERF_JSON` shows for warm2:

- `ttft_source=engine` (or `"ttft_source":"engine"`)
- numeric `prefill_tok_s` and `decode_tok_s`
- `quality_ok=true`
- `ttft_s` ≪ `wall_s` when `max_tokens=128`

Useful greps:

```bash
rg "ttft_source=|PERF_JSON|prefill_tok_s|decode_tok_s|quality_ok|OOM|FAIL" \
  build-vllm-xpu/logs/bench_perf.out | tail -40
```

Then update `PERF.md` + `SUCCESS_PERF.md` narrative if TTFT story changes.

**Paused job:** validation attempt **8680906** was still Q (“Not enough free nodes”) and was **qdel’d** on pause — must resubmit.

---

## 4. Standing rule — TP scaling on every campaign

**Every** future performance metric campaign (baseline, MoE A/B, serve, 131k, etc.) must run and report:

| TP | Script | Queue | Extra flags |
|----|--------|-------|-------------|
| 8 | `bench_perf.pbs` | `debug` | default util 0.82 |
| 4 | `bench_perf_tp4.pbs` | `debug` | `--kv-cache-memory-gib 8 --max-num-seqs 2` |
| 2 | `bench_perf_tp2.pbs` | `debug-scaling` | same KV pin |

Ingest into `build-vllm-xpu/perf-team/SCALING_TP248.md` (append a **dated** section).  
Do **not** claim a recipe is faster from TP=8 alone.

**Queue tip:** user often limited to ~2 jobs in `Q` across debug*; submit TP=4 + TP=8 first, then TP=2, or stagger. Concurrent TP=4 (`debug`) + TP=2 (`debug-scaling`) is the intended pattern.

**PBS flakiness seen Jul 18:** recurring `Execution server rejected request` / offline nodes — qdel + resubmit if stuck >~15–20 min with that comment.

---

## 5. Experiment ledger (S2–S5) — do not redo blindly

| ID | Job | Result | Keep? |
|----|-----|--------|-------|
| P0 baseline | 8680399 | warm2 ≈0.372, quality OK; TTFT null | Baseline |
| S2 persist cache | 8680469 | no warm2 win vs TMPDIR | Ops only |
| S3 fused MoE | 8680525 | ~1.47 tok/s, **quality FAIL** | Discard |
| S3 mxfp4_fp8 | 8680546 | ~1.47 tok/s, **quality FAIL** | Discard |
| S4 eager=false | 8680603 | ≈0.37, no win | Discard |
| S5 TP=12 | 8680623 | heads % 12 ≠ 0 | Invalid |
| TP=2 OOM | 2026-07-20 log | util KV over-alloc | Retry with KV pin |
| TP=4 OOM | 2026-07-20 log | same | Retry with KV pin |

Logs (not in git): `build-vllm-xpu/logs/bench_perf*.out`

**Largest remaining speed lever:** make **fused MoE** quality-correct (was ~4× e2e vs REF but garbage text). Until then keep `VLLM_XPU_FUSED_MOE_USE_REF=1`.

---

## 6. Exact resume command sequence

```bash
cd /lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b

# 0) hygiene
git status -sb
git log -3 --oneline
qstat -u xiaoliyan
# push if still ahead: git push origin main

# 1) P7 GATE
qsub bench_perf.pbs
# wait for build-vllm-xpu/logs/bench_perf.out → PERF_JSON with ttft_source=engine

# 2) Scaling (after P7 PASS) — watch Q-slot limit
qsub bench_perf_tp4.pbs
qsub bench_perf_tp2.pbs
# TP=8 already from step 1; if needed re-run bench_perf.pbs for apples-to-apples same day

# 3) Ingest
# - Append dated table to build-vllm-xpu/perf-team/SCALING_TP248.md
# - Update build-vllm-xpu/PERF.md

# 4) Next opt (example: fused MoE quality investigation)
# Use bench_perf_moe_fused.pbs only with quality gate; discard if all-!
# Always include TP=2/4/8 for any “win” claim.
```

### Ingest template (paste into SCALING_TP248.md)

```markdown
## Campaign YYYY-MM-DD — <recipe label>

| TP | Job | warm2_ttft_s | ttft_source | warm2_prefill_tok_s | warm2_decode_tok_s | warm2_e2e_tok_s | quality_ok |
|----|-----|--------------|-------------|---------------------|--------------------|-----------------|------------|
| 2 | | | | | | | |
| 4 | | | | | | | |
| 8 | | | | | | | |

Notes: ...
```

---

## 7. Doc map

| Doc | Role |
|------|------|
| **This file (`RESUME.md`)** | Session recovery + ordered next steps |
| `BEST_PRACTICE.md` | **Current recommended recipe (TP=2)** + scaling summary |
| `PERF_PLAN.md` | Strategy / workstreams / standing rules |
| `PERF.md` | Living experiment log |
| `SUCCESS_PERF.md` | S2–S5 quality-gated closure |
| `SUCCESS_INFER.md` / `SUCCESS_TRAIN.md` | Bring-up gates (CLOSED) |
| `PHASE_STATUS.md` | Chronological phase notes |
| `perf-team/SCALING_TP248.md` | TP=2/4/8 results |
| `perf-team/A1`–`A6_*.md` | Hypotheses / solutions / reviews |
| `../README.md` | Project entry |
| `../FILES.md` | Git allowlist (no models/env/logs) |

---

## 8. Pause snapshot (2026-07-20)

- Agent progress loop **stopped**
- PBS jobs: **none** left for user after qdel of 8680906  
- Debug / debug-scaling often showed **0 Running** while still refusing new jobs (“Not enough free nodes”) — site/routing issue, not a long local run queue  
- Uncommitted work was committed as `f60a2bb`; **push may still be needed**

When work resumes, update this section’s date and clear “paused” language in `PERF.md` / `PHASE_STATUS.md`.
