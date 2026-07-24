# Inkling — llama.cpp SYCL on Aurora

**Quant:** Unsloth UD-IQ1_S (~270 GB). **Queues:** debug / debug-scaling only.  
**Metrics every run:** TTFT_ms, prefill_tps, gen_tps (+ quality).

## Order (mandatory)
1. Short ctx: **min pure-GPU tiles** → **scale to 12 tiles** → **min MoE→CPU tiles** (no offload scale)
2. Max ctx: same three steps
3. gpt-oss knobs: `VMM=0`, pure GPU `-sm tensor -fa on` even `-ts`, MoE `-ncmoe 99` + sock0 HBM `--preferred=2`

## Paths
- Models: `models/inkling-UD-IQ1_S.gguf` → first shard
- Build: `build-llamacpp-sycl/build/bin/`
- Ledger: `build-llamacpp-sycl/CYCLE_LOG.md`

## Submit helpers
```bash
qsub download_inkling_ud_iq1_s.pbs
qsub build_llamacpp_sycl.pbs
qsub -v CYCLE=PG1 -N ll-PG1 -o build-llamacpp-sycl/logs/perf_PG1.out bench_llamacpp_sycl_perf.pbs
bash build-llamacpp-sycl/harvest_once.sh          # status
bash build-llamacpp-sycl/harvest_once.sh --submit # next cycle
```

No login-node long monitors.
