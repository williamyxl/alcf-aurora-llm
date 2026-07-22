# gpt-oss-120b XPU performance log

Living experiment + hypothesis/solution log for Aurora PVC (self-built vLLM-XPU stack).

**Closure:** S2–S5 complete — see [`SUCCESS_PERF.md`](SUCCESS_PERF.md).  
**Current best practice:** [`BEST_PRACTICE.md`](BEST_PRACTICE.md) — **TP=2**, warm2 e2e ≈ **1.15** tok/s (≃3× historical TP=8).

**Resumed 2026-07-21** — [`RESUME.md`](RESUME.md).  
**P7 PASS** — job **8681016**.  
**TP=2/4/8 REF scaling COMPLETE** — [`SCALING_TP248.md`](perf-team/SCALING_TP248.md): TP=2 best (**1.15** e2e).  
**Failed campaigns (document):** fused MXFP4 + BF16/FP16 unquant — both quality FAIL; see [`FAILED_ATTEMPTS.md`](perf-team/FAILED_ATTEMPTS.md).  
**Casting hypothesis killed:** half-prec ~3 tok/s max, still all-`!`.  
**Checkpoint policy:** **MXFP4 only** (`models/openai-gpt-oss-120b`) — no more BF16/FP16.  
**Next:** llama.cpp SYCL compile-from-scratch — [`../build-llamacpp-sycl/README.md`](../build-llamacpp-sycl/README.md).  
**Standing rule:** every future vLLM perf campaign runs **TP=2/4/8** with P7 fields.

## Failed — BF16/FP16 unquant MoE (2026-07-21)

Downloaded community BF16/FP16 checkpoints; REF unset; KV pinned by TP. Full table: [`perf-team/HALFPREC_TP248.md`](perf-team/HALFPREC_TP248.md).

| Dtype | TP | Job | warm2 e2e | warm2 decode | quality |
|-------|----|-----|-----------|--------------|---------|
| bf16 | 8 | 8681162 | 1.41 | 1.45 | **FAIL** |
| bf16 | 4 | 8681163 | **2.96** | **2.98** | **FAIL** |
| fp16 | 8 | 8681177 | 1.51 | 1.58 | **FAIL** |
| fp16 | 4 | 8681178 | 2.77 | 2.80 | **FAIL** |
| fp16 | 2 | 8681207 | — | — | **OOM** |

**Verdict:** Discard. Same garbage as fused; not a path to ≫13 tok/s.

## Failed — fused MXFP4 MoE TP=2/4/8 (2026-07-21)

See [`perf-team/FUSED_MOE_QUALITY.md`](perf-team/FUSED_MOE_QUALITY.md). Best ignore-quality speed ~5.2 decode @ TP=2 (8681118); TP=8 job **8681141** warm2 e2e 1.41 / decode 1.46, quality FAIL.

## P7 — engine TTFT / prefill / decode — PASS (8681016)

`disable_log_stats=False` populates `RequestOutput.metrics`. Host `x4408c7s2b0n0`, TP=8, REF MoE, same PASS recipe.

| Run | wall_s | e2e tok/s | ttft_s | ttft_source | prefill tok/s | decode tok/s | quality_ok |
|-----|--------|-----------|--------|-------------|---------------|--------------|------------|
| cold | 441.4 | 0.290 | **69.4** | engine | 2.48 | 0.341 | true |
| warm | 349.7 | 0.366 | **32.0** | engine | 5.37 | 0.400 | true |
| warm2 | 349.7 | 0.366 | **32.1** | engine | 5.36 | 0.400 | true |

**Interpretation:** Prior “343 s TTFT” was e2e wall. True warm TTFT ≈ **32 s**; decode ≈ **0.40 tok/s** dominates the remaining ~318 s of a 128-token generate. Steady-state bottleneck is decode (REF MoE), not missing TTFT instrumentation.

Raw: `build-vllm-xpu/logs/bench_perf.out` (append after Jul 18 baseline; backup `bench_perf.out.pre_p7_8680399`).

## Schema (`PERF_JSON`)

Minimum fields from `bench_perf.py`:

| Field | Meaning |
|-------|---------|
| `n_tiles` | TP size (tiles used) |
| `moe_mode` | `ref` / `fused` / `mxfp4_fp8` |
| `attn` | attention backend |
| `cold_ttft_s` / `warm_ttft_s` / `warm2_ttft_s` | TTFT per run (`null` if engine TTFT missing) |
| `*_ttft_source` / `runs.*.ttft_source` | `engine` or `fallback_wall` |
| `*_prefill_tok_s` | `n_prompt / ttft` when engine TTFT present |
| `*_decode_tok_s` | `(n_out-1)/(t_last-t_first)` when timestamps present |
| `runs.*.wall_s` | full generate wall (always) |
| `cold_e2e_tok_s` / `warm_e2e_tok_s` / `warm2_e2e_tok_s` | end-to-end tok/s |
| `n_output_tokens` | last-run output length |
| `n_prompt_tokens` | encoded prompt length (packed when `--prefill-tokens` set) |
| `max_model_len` | vLLM context window |
| `text_preview` | quality spot-check |
| `quality_ok` | false if all-`!` or token-id-0 |

## Phase 0 baseline — COMPLETE

Job **8680399** host `x4303c1s3b0n0`, TP=8, REF MoE, TRITON_ATTN, eager, util=0.82.

| Run | wall_s | e2e_tok_s | ttft_s | ttft_source | quality_ok |
|-----|--------|-----------|--------|-------------|------------|
| cold | 435.01 | 0.294 | null | fallback_wall | true |
| warm | 343.93 | 0.372 | null | fallback_wall | true |
| warm2 | 343.71 | 0.372 | null | fallback_wall | true |

**Verdict:** warm ≈ warm2 ≈ Phase 5 timed (~344 s / ~0.37 tok/s). Slowdown is **steady-state**, not cold-JIT-only. Cold pays extra ~90 s JIT. Engine TTFT unavailable (`ttft_s=null`).

Raw: `build-vllm-xpu/logs/bench_perf.out` (`PERF_JSON=...`).

Phase 5 note: reported “343 s TTFT” was **e2e wall**, not true first-token.

## Follow-on experiments (gated)

### S2 persist — COMPLETE (8680469, host earlier `x4310c4s0b0n0`)

| Run | wall_s | e2e_tok_s | quality_ok |
|-----|--------|-----------|------------|
| cold | 435.93 | 0.294 | true |
| warm | 343.09 | 0.373 | true |
| warm2 | 343.87 | 0.372 | true |

Within-job warm ≈ Phase 0 TMPDIR baseline — durable cache alone does **not** fix steady-state ~0.37 tok/s. Cross-job cold win not measured yet (would need a 2nd persist job).

### Jobs

| Job | Script | Status | Purpose |
|-----|--------|--------|---------|
| 8680469 | `bench_perf_persist.pbs` | **done** | S2 durable caches |
| 8680525 | `bench_perf_moe_fused.pbs` | **DONE FAIL** — warm2≈1.47 tok/s, all-`!`/id0 | Discard; keep REF |
| 8680546 | `bench_perf_moe_fp8.pbs` | **DONE FAIL** — warm2≈1.47 tok/s, all-`!`/id0 | Discard; keep REF |
| 8680603 | `bench_perf_graphs.pbs` | **DONE** — warm2≈0.37, quality OK, ≈eager REF | S4: no win from `enforce_eager=false` alone |
| 8680623 | `bench_perf_tp12.pbs` | **DONE FAIL** — exit 1 at LLM init | S5: `64 heads % TP=12 != 0` — **TP=12 invalid for gpt-oss** |
| 8680703 | `bench_perf_ctx131k.pbs` | **qdel'd** (freed debug-scaling for TP scaling) | Resubmit after TP=2/4/8 study |

Valid TP sizes for 64 heads: 1,2,4,8,16,… — **not 12**. Full-node tile use needs EP/other sharding, not attn TP=12.

**Queue policy:** for 2 concurrent benches, use `debug` + `debug-scaling` (`qsub -q debug-scaling …`).

### Planned after open items close

| Phase | Item | Status |
|-------|------|--------|
| P6 | Perf bench at **`max_model_len=131072`** | **paused** — 8680703 qdel'd for priority TP=2/4/8 scaling; resubmit after |
| **PRIORITY** | TP=2 / 4 / 8 scaling | **in flight** — TP4=`8680707` (debug), TP2=`8680711` (debug-scaling); TP8 baseline warm2≈0.372; see [SCALING_TP248.md](perf-team/SCALING_TP248.md) |

Current default remains `max_model_len=4096` (existing short-context PBS unchanged).

### S3 MoE summary (REF stays default)

| Mode | Job | warm2 e2e | quality |
|------|-----|-----------|---------|
| ref (baseline) | 8680399 / 8680469 | ~0.37 | PASS |
| fused | 8680525 | ~1.47 | **FAIL** |
| mxfp4_fp8 | 8680546 | ~1.47 | **FAIL** |

## Team artifacts

Closure doc: [`SUCCESS_PERF.md`](SUCCESS_PERF.md). See [`perf-team/`](perf-team/) for A1–A6 reports. A6 **PASS**.

## Hypothesis / solution tracker

| ID | Hypothesis | Status | Solution | Implemented | Verified |
|----|------------|--------|----------|-------------|----------|
| H9 | Metric hygiene | Confirmed | S1 | A5 | A6 PASS |
| M1 JIT/cache | Cold +90s; warm still ~0.37 | S2 | A5 | within-job: no steady-state win |
| H1 REF MoE | Steady-state bottleneck vs fused | S3 | A5 | fused **FAIL quality** (warm2 ~1.47 tok/s garbage); keep REF |
| H3 MXFP4 ceiling | Context + fp8 path | S3 | A5 | fp8 **FAIL quality** (like fused); keep REF |
| H4 eager | Graph-on trial | S4 job 8680603 | A5 | **DONE**: warm2≈0.37 ≈eager; no win (compile/cudagraph still NONE) |
| H7 TP=12 | Full-node tiles | S5 | A5 | **FAIL config**: 64 attn heads not divisible by 12; keep TP=8 |
| P6 | 131072 context bench | harness + qsub | `bench_perf_ctx131k.pbs` | **queued** (KV OOM risk) |
| H8/H6 | Deferred S6/S7 | — | — | — |
