# Current best practice — gpt-oss-120b on Aurora XPU

**As of:** 2026-07-21  
**Scope:** Single-stream inference (MOF-style chat / `bench_perf` / `infer_chat`), quality-gated.  
**Living detail:** [`SCALING_TP248.md`](perf-team/SCALING_TP248.md) · [`PERF.md`](PERF.md) · recovery: [`RESUME.md`](RESUME.md)

---

## Verdict

**Use 2 GPU tiles (TP=2).** For the quality-passing REF MoE recipe, TP=2 is the best latency/throughput choice today — about **3×** faster end-to-end than the historical TP=8 Phase 5 PASS setup.

| TP | Job (2026-07-21) | warm2 e2e tok/s | warm2 decode tok/s | warm2 TTFT (engine) | quality |
|----|------------------|-----------------|--------------------|---------------------|---------|
| **2** | **8681063** | **1.15** | **1.22** | **7.5 s** | OK |
| 4 | 8681062 | 0.66 | 0.71 | 15.8 s | OK |
| 8 | 8681016 | 0.37 | 0.40 | 32.1 s | OK |

Inverse scaling under BS=1 REF MoE: more tiles → more sync/comm overhead, not more speed.

---

## Recommended recipe (do this)

| Setting | Value | Why |
|---------|--------|-----|
| **TP / tiles** | **2** | Best warm2 e2e / decode / TTFT (see table) |
| MoE | **REF** — `VLLM_XPU_FUSED_MOE_USE_REF=1` | Fused / mxfp4_fp8 → all-`!` / token-id-0 |
| Attention | `TRITON_ATTN` | FLASH_ATTN garbles gpt-oss on XPU |
| `enforce_eager` | `True` | Graphs/Dynamo: no win; keep off |
| dtype | `bfloat16` | PASS path |
| `max_model_len` | `4096` | Default; 131k is P6 |
| KV memory | **`--kv-cache-memory-gib 8`** | Required for TP=2/4 — util planner OOMs (~50 GiB KV) |
| `max_num_seqs` | `2` (single-stream bench) | Pair with KV pin |
| `gpu_memory_utilization` | `0.82` | Ignored for KV size when KV pin is set |
| Selector | `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` | No OpenCL (SEGV) |
| Triton | Intel device extensions + OpenCL-optional `driver.c` patch | AuroraBug#102 |
| Compile | `TORCHDYNAMO_DISABLE=1`, `TORCH_COMPILE_DISABLE=1` | XPU safety |
| Stats | `disable_log_stats=False` | Real engine TTFT / prefill / decode (P7) |
| Modules | `oneapi/release/2025.3.1` only | **No** `module load frameworks` |
| Env | `build-vllm-xpu/env` | Self-built stack |

### How to run

```bash
cd /lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b

# Recommended single-stream bench (TP=2, REF, KV pin)
qsub bench_perf_tp2.pbs
# log: build-vllm-xpu/logs/bench_perf_tp2.out → PERF_JSON

# Interactive / chat smoke (after updating one_chat to TP=2 + KV pin)
qsub infer_chat.pbs
```

Equivalent `bench_perf.py` args:

```bash
python bench_perf.py --tp 2 --max-tokens 128 --moe-mode ref \
  --gpu-memory-utilization 0.82 --max-num-seqs 2 --kv-cache-memory-gib 8
```

---

## Scaling performance (documented)

Campaign **2026-07-21**, REF MoE + TRITON_ATTN + eager + `max_model_len=4096`, P7 engine metrics, MOF prompt, cold/warm/warm2. TP=2/4 used KV pin 8 GiB.

### warm2 summary

| TP | wall_s | ttft_s | prefill_tok_s | decode_tok_s | e2e_tok_s | vs TP=2 e2e |
|----|--------|--------|---------------|--------------|-----------|-------------|
| 2 | 111.6 | 7.50 | 22.93 | 1.220 | **1.147** | 1.00× |
| 4 | 194.5 | 15.77 | 10.90 | 0.711 | 0.658 | 0.57× |
| 8 | 349.7 | 32.08 | 5.36 | 0.400 | 0.366 | 0.32× |

All runs: `ttft_source=engine`, `quality_ok=true`.

### Why TP=2 wins (for now)

1. Decode roughly halves each time TP doubles → communication / rank sync dominate REF MoE BS=1.
2. Warm TTFT worsens with TP (7.5 → 16 → 32 s).
3. Earlier “must use TP=8” was a **memory planner** story (TP=2/4 OOM without KV pin), not a performance optimum.
4. TP=12 is **invalid** (`64` attn heads `% 12 != 0`).

**When to reconsider TP>2:** concurrent serve / continuous batching needing more aggregate KV or throughput, or a quality-correct fused MoE path that changes the compute/comm balance. Until then, default to **2 tiles**.

Full table + notes: [`perf-team/SCALING_TP248.md`](perf-team/SCALING_TP248.md).

---

## Standing rules

1. **Quality gate first** — any path with token-id-0 / `!!!` is FAIL, regardless of tok/s.
2. **Every metric campaign** reports **TP=2 / 4 / 8** with P7 fields (`ttft`, `prefill_tok_s`, `decode_tok_s`, `e2e_tok_s`, `quality_ok`). Do not claim a recipe win from one TP alone.
3. **Never** call e2e wall “TTFT” — require `ttft_source=engine` (or stream).
4. Keep fused / mxfp4_fp8 **out** of default scripts until quality_ok.

---

## Explicitly not best practice (failed / discarded)

Full ledger: [`perf-team/FAILED_ATTEMPTS.md`](perf-team/FAILED_ATTEMPTS.md).

| Path | Status |
|------|--------|
| Fused MXFP4 MoE TP=2/4/8 | ~5.2 decode @ TP=2; **quality FAIL** all TP — `FUSED_MOE_QUALITY.md` |
| Unquant BF16/FP16 MoE | ~3 decode @ BF16 TP=4; **quality FAIL**; TP2 OOM — `HALFPREC_TP248.md` |
| `VLLM_XPU_FUSED_MOE_USE_MXFP4_FP8=1` | Same quality FAIL |
| `enforce_eager=False` | No speed win vs eager REF |
| TP=8 as default for single-stream | Slower under REF |
| TP=12 | Invalid for this model |
| OpenCL in device selector | SEGV |
| `module load frameworks` | No usable causal / gpt-oss path for this project |

---

## Doc map

| Doc | Role |
|-----|------|
| **This file** | Current recommended recipe + scaling verdict |
| `SUCCESS_INFER.md` | Historical Phase 5 quality PASS (TP=8) |
| `SUCCESS_PERF.md` | S2–S5 closure + pending opts |
| `PERF.md` | Living experiment log |
| `RESUME.md` | Cold-session recovery |
| `perf-team/SCALING_TP248.md` | TP=2/4/8 campaign tables |
| `perf-team/FAILED_ATTEMPTS.md` | Failed campaigns ledger |
