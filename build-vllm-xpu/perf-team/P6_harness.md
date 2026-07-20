# P6 — Long-context (131k) performance harness

**Goal:** Measure gpt-oss-120b cold/warm/warm2 at `max_model_len=131072` with a packed ~120k-token prefill, same PASS recipe (REF MoE, TRITON_ATTN, L0), quality gate on.

## Submit

```bash
cd /lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b
qsub bench_perf_ctx131k.pbs
```

- Queue: `debug-scaling`
- Walltime: `01:00:00` (queue max; 02:00:00 rejected by PBS)
- Log: `build-vllm-xpu/logs/bench_perf_ctx131k.out`
- Command: `python bench_perf.py --tp 8 --max-tokens 128 --max-model-len 131072 --prefill-tokens 120000 --moe-mode ref`

## Env (PASS defaults)

| Knob | Value |
|------|-------|
| MoE | `VLLM_XPU_FUSED_MOE_USE_REF=1` |
| Attn | `TRITON_ATTN` |
| Selector | `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` (no OpenCL) |
| Eager | `enforce_eager=true` (default) |
| util | `0.82` (default) |
| Caches | durable `$WORKDIR/.cache/{triton,sycl}_xpu_gptoss` |
| `SYCL_CACHE_PERSISTENT` | **unset** (never set to 1) |

Short-context jobs keep `max_model_len=4096` (CLI default).

## Risks

1. **KV OOM** — 131k context + TP=8 + util=0.82 may fail at LLM init or first generate. Still a P6 data point; note OOM in `PERF.md`.
2. **Walltime** — cold JIT + long prefill may approach the 1h `debug-scaling` cap; warm/warm2 may not finish if cold is huge.
3. **Quality** — packed filler precedes the MOF question; gate still rejects all-`!` / token-id-0.
4. **Log size** — `bench_perf.py` truncates long prompts in stdout; `PERF_JSON` still has `n_prompt_tokens` + `max_model_len`.

## Ingest

After the job ends, grep `PERF_JSON=` from the log and append a P6 results row to `PERF.md`. If OOM, record the failure mode instead of metrics.
