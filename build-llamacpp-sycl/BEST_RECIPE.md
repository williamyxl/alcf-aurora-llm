# Best recipes — llama.cpp SYCL gpt-oss-120b MXFP4 (Aurora)

Frozen after Phase F + G completion (2026-07-22). Queues: debug / debug-scaling only.

## Short context (decode ≥30)

### A — 2-tile pure GPU
```bash
ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0,1 GGML_SYCL_ENABLE_VMM=0
llama-completion -m openai-gpt-oss-120b-mxfp4.gguf -ngl 99 -sm tensor -fa on -ts 0.5/0.5 --no-mmap
# measured: gen ≈ 29.95–30.01 (P11 / P14_tp2)
```

### B — 1-tile MoE→CPU + sock0 NUMA (faster decode)
```bash
ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0 GGML_SYCL_ENABLE_VMM=0
numactl --physcpubind=1-51,105-155 --membind=0 \   # or --preferred=2 for HBM
  llama-completion ... -ngl 99 -sm none -fa on -ncmoe 99 -t 32 --numa numactl --no-mmap
# F4 DDR: gen 33.28 ; F4h HBM-pref: gen 33.78
```

## Long context (`-c 131072`)

| Arm | gen_tps | prefill_tps | Notes |
|-----|--------:|------------:|-------|
| **G1** 1-tile MoE+NUMA | **32.66** | 326 | **prefer for decode** |
| G0 2-tile GPU | 28.61 | **465** | better prefill |

ALCF bind: GPU0 ↔ socket 0 HWTs `1-51,105-155`; DDR NUMA 0; HBM NUMA 2.
Docs: https://docs.alcf.anl.gov/aurora/running-jobs-aurora/#mpi-rank-and-thread-binding-to-cores-and-gpus
