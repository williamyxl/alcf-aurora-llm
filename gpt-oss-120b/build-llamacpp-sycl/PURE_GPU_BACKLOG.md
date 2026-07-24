# Hard rule: -ngl 99 (or all layers). NO partial CPU offload until pure-GPU options exhausted.

## Pure-GPU OOM / smoke backlog (before any -ngl < 99)

| ID | Experiment | Status |
|----|------------|--------|
| G0 | COMPOSITE `0.0,0.1` + layer + ngl99 | S1 FAIL OOM |
| G1 | COMPOSITE `0` implicit + sm none + ngl99 | S2 FAIL OOM |
| G2 | FLAT `0,1` + layer + completion -v --no-mmap + ngl99 | **S3 FAIL** — 2 devices + weight load OK; OOM warmup `mul_mat` |
| G3 | FLAT `0,1` + `-sm tensor` + ngl99 (same GPU 2 tiles) | **S4 FAIL** VMM=1; **P5 PASS** VMM=0 @ **29.13 tg** |
| G4 | FLAT `0,1` + layer + `GGML_SYCL_ENABLE_VMM=1` + ngl99 | **S5 FAIL** (VMM already default) |
| G5/G6 | tiny ctx/ubatch | folded into S6 |
| G7 | L0_API=0 | P6 (layer) / P10 (tensor) |
| **G8** | **`GGML_SYCL_ENABLE_VMM=0` + tiny ub** | **S6 PASS @ 26.82 eval tok/s** |
| G9 | COMPOSITE + VMM=0 | **P3** 27.46 — worse than FLAT |
| G10 | Rebuild / newer tip / IGC | P4 layer noop; P11 tensor pending; tip has **no** ggml-sycl delta |
| **P24** | **MXFP4 MoE SoA reorder** (code) | **DONE** gen 29.15 — no win vs P11 29.95 |
| **P25** | P24 binary + P11 IGC knobs | env ready; long shot after P24 |
| **P14** | **TP=2/4/8 dense pack** | **P14_tp2 DONE gen=30.01 TARGET MET**; tp4/8 in flight |

**Working baseline (use for all further pure-GPU runs):**
```bash
export GGML_SYCL_ENABLE_VMM=0
# FLAT 0,1 + -sm tensor -ngl 99 -fa on   # P7b best: 29.24 tg128
# prior: -sm layer → 28.09 tg128
```

**Parked (Phase E):** partial `-ngl`, MoE-on-CPU hybrids — **moved to Phase F** in [`PLAN.md`](PLAN.md).  
**P14 note:** Max 1550 has only **2 tiles/GPU** — same-GPU max is TP=2; TP4/8 add whole GPUs (both tiles), never spread one tile per GPU.

## Phase F unlock (1 tile + MoE→CPU + NUMA)

| ID | Experiment | Status |
|----|------------|--------|
| **F0** | 1 tile + `-ncmoe 99` (**no NUMA** baseline) | **DONE** gen=28.56 — `cycles/F0.env` |
| **F1** | 1 tile + `-ncmoe 8` + local-DDR NUMA | in flight — `cycles/F1.env` |
| **F2** | 1 tile + `-ncmoe 16` + local-DDR NUMA | in flight — `cycles/F2.env` |
| **F3** | 1 tile + `-ncmoe 32` + local-DDR NUMA | planned — `cycles/F3.env` |
| **F4** | `-ncmoe 99` + local-DDR NUMA (vs F0) | planned — `cycles/F4.env` |
| **F4h** | `-ncmoe 99` + CPU-HBM preferred | planned — `cycles/F4h.env` |
| F5 | Best NUMA F* + IGC | planned |

Default MoE host policy (ALCF Aurora): `--physcpubind=1-51,105-155 --membind=0` (sock0 DDR; GPU0) + `--numa numactl` — see [`numa_moe_host.env.sh`](numa_moe_host.env.sh).  
HBM preferred (F4h): `--preferred=2` (sock0 HBM NUMA node **2**, not 8–15).  
Docs: [GPU–CPU binding](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/#mpi-rank-and-thread-binding-to-cores-and-gpus), [SPR HBM](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/#using-the-hbm-on-the-sapphire-rapids-cpus).

Submit: `qsub -v CYCLE=F4 -N ll-F4 -o .../logs/perf_F4.out ../bench_llamacpp_sycl_perf.pbs`

## Phase G FINAL — max content length (131072) + NUMA

| ID | Experiment | Status |
|----|------------|--------|
| **G0** | 2-tile pure GPU · `N_CTX=131072` · local NUMA optional | planned — `cycles/G0.env` |
| **G1** | 1-tile + `-ncmoe 99` · **local-DDR NUMA required** | planned — `cycles/G1.env` |
| G0s / G1s | same arms @ 65536 (G1s NUMA required) | planned |
| G0t / G1t | same arms @ 32768 (G1t NUMA required) | planned |

Submit (debug/debug-scaling ≤59 m only — **no capacity**):  
`qsub -q debug-scaling -v CYCLE=G0 … bench_llamacpp_sycl_phaseG.pbs`  
`qsub -q debug -v CYCLE=G1 …`
