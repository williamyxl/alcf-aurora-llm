## Standing rule (2026-07-20)

**Every future performance metric campaign** must run **TP=2 / 4 / 8** with P7 fields and append results here (dated section). Do not claim a recipe win from TP=8 alone.

---

# TP scaling study: gpt-oss-120b on Aurora XPU (2 / 4 / 8 tiles)

**Recipe:** REF MoE + TRITON_ATTN + eager + `max_model_len=4096`, MOF prompt, cold/warm/warm2, P7 engine metrics.  
**TP=2/4:** `--kv-cache-memory-gib 8 --max-num-seqs 2` (avoids util-planner OOM).

## Campaign 2026-07-21 — P7 metrics + KV-pin (COMPLETE)

| TP | Job | Host | warm2_ttft_s | ttft_src | warm2_prefill | warm2_decode | warm2_e2e | quality | vs TP=2 (e2e) |
|----|-----|------|--------------|----------|---------------|--------------|-----------|---------|---------------|
| **2** | **8681063** | `x4719c7s2b0n0` | **7.50** | engine | **22.9** | **1.220** | **1.147** | true | **1.00×** |
| **4** | **8681062** | `x4603c4s5b0n0` | 15.77 | engine | 10.9 | 0.711 | 0.658 | true | 0.57× |
| **8** | **8681016** | `x4408c7s2b0n0` | 32.08 | engine | 5.36 | 0.400 | 0.366 | true | 0.32× |

All three: `ttft_source=engine`, `quality_ok=true`. KV pin worked (no OOM).

### Full warm2 detail

| TP | wall_s | ttft_s | prefill_tok_s | decode_tok_s | e2e_tok_s |
|----|--------|--------|---------------|--------------|-----------|
| 2 | 111.6 | 7.50 | 22.93 | 1.220 | 1.147 |
| 4 | 194.5 | 15.77 | 10.90 | 0.711 | 0.658 |
| 8 | 349.7 | 32.08 | 5.36 | 0.400 | 0.366 |

### Analysis

1. **Inverse scaling** for this BS=1 REF MoE recipe: **TP=2 ≫ TP=4 ≫ TP=8** on TTFT, decode, and e2e.
2. Decode tok/s roughly halves as TP doubles (1.22 → 0.71 → 0.40) — consistent with **comm / sync overhead** dominating over parallel matmul gains under REF MoE.
3. Warm TTFT also worsens with TP (7.5 → 16 → 32 s) — more ranks to sync for first token.
4. Prior “TP=8 only” story was a **memory/planner artifact**, not a performance optimum. With pinned 8 GiB KV, TP=2/4 run cleanly.
5. **Practical recommendation (quality REF path):** prefer **TP=2** for single-stream latency/throughput (~1.15 e2e tok/s, ~3× TP=8). Revisit TP=8 only if batching/serve needs the extra KV headroom of more tiles.

### Earlier OOM campaign (2026-07-20, superseded)

Without `--kv-cache-memory-gib`, TP=2/4 OOM’d (util planner reserved ~50 GiB KV). Logs: `bench_perf_tp{2,4}.out` older sections.

---

## Config validity

| TP | 64 attn / 8 KV heads | Memory with KV pin 8 GiB | Result (2026-07-21) |
|----|----------------------|--------------------------|---------------------|
| 2 | OK | weights ~31 GiB + 8 GiB KV | **PASS** best tok/s |
| 4 | OK | weights ~16 GiB + 8 GiB KV | **PASS** mid |
| 8 | OK | weights ~8 GiB + util KV | **PASS** slowest BS=1 |
| 12 | INVALID | — | heads % 12 ≠ 0 |
