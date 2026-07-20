## Standing rule (2026-07-20)

**Every future performance metric campaign** must run **TP=2 / 4 / 8** with P7 fields and append results here (dated section). Do not claim a recipe win from TP=8 alone.

**Paused 2026-07-20** — resume via [`../RESUME.md`](../RESUME.md). Next campaign: P7-validated TP=8 + KV-pinned TP=2/4.

---

# TP scaling study: gpt-oss-120b on Aurora XPU (2 / 4 / 8 tiles)

**Priority analysis** — REF MoE + TRITON_ATTN + eager + `max_model_len=4096`, same MOF prompt, cold/warm/warm2.

## Verdict

**No continuous speedup curve.** At this recipe / context length, only **TP=8** loads and runs. **TP=2 and TP=4 OOM** during engine init (weights + KV exceed ~64 GiB/tile). Inference scaling among {2,4,8} is therefore: **TP=8 only viable point**; cannot compute speedup/efficiency vs TP=2.

## Config validity

| TP | 64 attn heads | 8 KV heads | Memory expectation | Result |
|----|---------------|------------|--------------------|--------|
| 2 | OK | OK | ~33 GiB weights/tile; KV tight | **OOM** |
| 4 | OK | OK | ~16.5 GiB weights/tile | **OOM** |
| 8 | OK (PASS baseline) | OK | ~8.2 GiB weights/tile; ~49 GiB KV | **PASS** |

## Jobs (final)

| TP | Job / host | Queue | util / KV | Status |
|----|------------|-------|-----------|--------|
| 2 | ran 2026-07-20 on `x4316c4s4b0n0` | debug-scaling | util 0.85 (no explicit KV) | **FAIL OOM** (~62.4 GiB allocated, need +2.83 GiB) |
| 4 | ran 2026-07-20 on `x4111c1s6b0n0` | debug | util 0.82 (no explicit KV) | **FAIL OOM** (~62.9 GiB allocated, need +2.74 GiB) |
| 8 | 8680399 / 8680469 | debug | util 0.82 | **PASS** warm2≈0.372 tok/s, quality_ok |

### Fix attempt (2026-07-20)

Root cause: XPU memory profiler reports `available_KV ≈ util × HBM` (negative non-torch cancels weights). Fix: `--kv-cache-memory-gib 8` + `--max-num-seqs 2` in `bench_perf_tp{2,4}.pbs` (bypasses util KV planner; 8 GiB ≫ single-stream 4K need).

| TP | Job | Settings | Status |
|----|-----|----------|--------|
| 2 | **8680903** | util 0.82, max_num_seqs 2, kv 8 GiB | queued (debug-scaling) |
| 4 | **8680902** | util 0.82, max_num_seqs 2, kv 8 GiB | queued (debug) |

Earlier Jul 18 submits hit PBS `Execution server rejected request` / offline nodes repeatedly; jobs that finally ran are the Jul 20 OOM logs above.

## Results table

| TP | cold wall_s | cold e2e | warm wall_s | warm e2e | warm2 e2e | quality_ok | speedup vs TP=2 | efficiency |
|----|-------------|----------|-------------|----------|-----------|------------|-----------------|------------|
| 2 | — | — | — | — | — | — | n/a (OOM) | n/a |
| 4 | — | — | — | — | — | — | n/a (OOM) | n/a |
| 8 | 435.0 | 0.294 | 343.9 | 0.372 | **0.372** | true | n/a (no TP=2 baseline) | n/a |

TP=8 `PERF_JSON` from 8680399: warm2_e2e_tok_s=0.3724, n_out=128, `fallback_wall` TTFT.

## Analysis

1. **Head divisibility** allows TP∈{1,2,4,8,…}; memory does not allow TP=2 or TP=4 at `max_model_len=4096` with this MXFP4 MoE + REF + util≤0.85 recipe.
2. OOM is at **init** (allocate ~2.7–2.8 GiB when tile already ~62–63 GiB full), not mid-generate — typical of weight shard + KV reservation too large for fewer tiles.
3. **TP=1** previously OOM’d the same way; **TP=12** invalid (`64 % 12 != 0`).
4. Steady-state throughput at the only working point remains **~0.37 tok/s** (REF MoE bound), not cold-JIT-only.
5. To get a real 2/4/8 curve would need either lower `max_model_len`, lower util/KV, or a smaller memory recipe — none of which match the current PASS quality setup at 4K.

## Logs

- `build-vllm-xpu/logs/bench_perf_tp2.out` (2026-07-20 13:52)
- `build-vllm-xpu/logs/bench_perf_tp4.out` (2026-07-20 13:54)
- `build-vllm-xpu/logs/bench_perf.out` (TP=8 baseline)
