# Cycle ledger — llama.cpp SYCL (MXFP4, same-GPU 2 tiles)

**Target:** gen ≥ 30 tok/s + meaningful text. **Max cycles:** 100.  
**Plan:** [`PLAN.md`](PLAN.md)  
**Pure-GPU rule:** `-ngl 99` until [`PURE_GPU_BACKLOG.md`](PURE_GPU_BACKLOG.md) exhausted.

**Required metrics every cycle:** TTFT_ms, prefill_tps, gen_tps (+ optional bench).  
**Breakthrough:** `GGML_SYCL_ENABLE_VMM=0`.

| Cycle | Job | Change | TTFT_ms | prefill_tps | gen_tps | bench_ttft_ms | bench_prefill_tps | bench_gen_tps | Notes |
|-------|-----|--------|--------:|------------:|--------:|--------------:|------------------:|--------------:|-------|
| S6 | **8682042** | VMM=0 smoke | **435.46** | **45.93** | **26.82** | — | — | — | quality PASS |
| P2 | **8682591** | FLAT layer FA | —† | —† | —† | **776.85** | **659.07** | **28.09** | |
| P3 | **8682725** | COMPOSITE none | —† | —† | —† | **960.04** | **533.31** | **27.46** | worse |
| P4 | **8682840** | IGC + layer | —† | —† | —† | **767.55** | **667.06** | **28.09** | |
| P5 | **8682873** | tensor FA | —† | —† | —† | **1014.28** | **504.79** | **29.13** | |
| P6 | 8682900 | layer L0 | — | — | — | — | — | — | qdel |
| P7 | 8682966 | tensor+GRAPH+cli | — | — | — | — | — | — | FAIL walltime |
| P7b | **8683141** | tensor+GRAPH | —‡ | —‡ | —‡ | **1011.82** | **506.02** | **29.24** | TTFT lost (`rg`) |
| **P7c** | **8683158** | tensor FA; grep TTFT | **453.03** | **44.15** | **29.20** | **1020.86** | **501.54** | **29.07** | quality PASS MOF text |
| P8 | **8683201** | tensor FA **off** | — | — | — | — | — | — | **FAIL**: `SPLIT_MODE_TENSOR requires flash_attn` |
| P9 | **8683228** | tensor + FUSION=0 | **452.05** | **44.24** | **28.27** | **997.45** | **513.31** | **28.10** | worse; keep fusion ON |
| P10 | **8683321** | tensor + L0_API=0 | **461.12** | **43.37** | **29.22** | **1023.08** | **500.45** | **29.04** | ~flat vs P7c |
| **P11** | **8683523** | tensor + IGC/GRF | **449.30** | **44.51** | **29.95** | **1028.96** | **497.59** | **29.59** | quality PASS |
| P11b | (loop) | P11 IGC re-run | **451.03** | **44.34** | **29.98** | **1021.47** | **501.24** | **29.00** | quality PASS |
| P11c | **8686118** | P11 IGC re-run | **456.13** | **43.85** | **29.53** | **1018.16** | **502.87** | **29.06** | |
| P12 | **8684027** | ts 0.55/0.45 | **448.54** | **44.59** | **29.14** | **1004.57** | **509.67** | **28.86** | worse than even |
| P13 | **8683632** | ts 0.45/0.55 | **456.24** | **43.84** | **28.82** | **1011.66** | **506.10** | **27.92** | worse |
| P15 | **8684757** | -t 8 | **450.29** | **44.42** | **29.87** | **994.04** | **515.07** | **29.87** | ~flat |
| P16 | **8684763** | PRIORITIZE_DMMV=1 | **455.55** | **43.90** | **28.76** | **1027.82** | **498.14** | **28.63** | worse |
| P17 | **8684311** | ts+mg+t16 | **448.19** | **44.62** | **29.72** | **984.26** | **520.19** | **29.54** | below P11 |
| P18 | **8685013** | ctk/ctv q8_0 | **678.61** | **29.47** | **26.17** | **1019.66** | **502.13** | **26.07** | abandon |
| P19 | **8685267** | ASYNC_MEM_OP=1 | **458.60** | **43.61** | **29.49** | **1009.74** | **507.06** | **29.27** | no win |
| P23 | **8685318** | SCHED/SYCL debug | **496.58** | **40.28** | **22.11** | — | — | — | debug overhead; not speed |
| P24 | **8685017** | MXFP4 MoE SoA | **449.85** | **44.46** | **29.15** | **996.09** | **514.01** | **28.96** | quality PASS; no win |
| **P14_tp2** | **8685384** | even `-ts` 2-tile | **449.59** | **44.49** | **30.01** | **995.02** | **514.56** | **29.79** | 1× ≥30; quality PASS |
| P14_tp2b | **8685831** | confirm P14_tp2 | **459.63** | **43.51** | **28.66** | **1023.57** | **500.21** | **28.55** | confirm low |
| P14_tp2c | **8685995** | 3rd confirm | **449.03** | **44.54** | **29.97** | **1005.46** | **509.22** | **29.78** | ~30 |
| P14_tp2_igc | **8685991** | recipe + IGC | **457.39** | **43.73** | **29.43** | **1039.59** | **492.50** | **29.16** | no win |
| P14_tp4 | **8685832** | 4 tiles / 2 GPUs | **681.16** | **29.36** | **17.10** | **1458.77** | **350.98** | **16.99** | abandon multi-GPU TG |
| P14_tp8 | **8686110** | 8 tiles / 4 GPUs | **1145.36** | **17.46** | **7.01** | **2623.76** | **195.14** | **7.11** | much worse |
| P14_tp2d | **(confirm)** | P14_tp2 4th confirm | **452.69** | **44.18** | **29.18** | **1010.70** | **506.58** | **29.71** | quality PASS |

† pre-TTFT harness. ‡ `rg` missing on compute.

**Best real gen_tps:** **30.01** (P14_tp2, once); next **29.98** (P11b), **29.97** (P14_tp2c), **29.95** (P11).  
**Dead ends:** uneven `-ts`, FUSION=0, L0_API=0, DMMV, KV q8, SoA reorder alone, TP4/TP8 for decode.

### P14 — dense pack (2 tiles/GPU)
| Cycle | TP | Affinity | GPUs | Outcome |
|-------|---:|----------|-----:|---------|
| `P14_tp2` | 2 | `0,1` | 1 | gen 30.01 once |
| `P14_tp4` | 4 | `0,1,2,3` | 2 | gen 17.10 |
| `P14_tp8` | 8 | `0..7` | 4 | gen 7.01 |

**Winning-ish recipe (2 tiles):**
```bash
ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0,1 GGML_SYCL_ENABLE_VMM=0
-ngl 99 -sm tensor -fa on -ts 0.5/0.5 --no-mmap
# optional IGC: IGC_EnableGEPLRUCache=1 IGC_ForceOCLSIMDWidth=16 NEO_CACHE_PERSISTENT=1
```

### Phase F — planned (1 tile + MoE→CPU + NUMA)

See [`PLAN.md`](PLAN.md) Phase F. MoE→CPU uses **ALCF sock0 bind** (`--physcpubind=1-51,105-155 --membind=0`) except F0 baseline.  
Ref: [ALCF GPU–CPU binding](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/#mpi-rank-and-thread-binding-to-cores-and-gpus).

| Cycle | Change | Status |
|-------|--------|--------|
| F0 | 1-tile + `-ncmoe 99` (**no NUMA**) | **DONE** gen=28.56 |
| F1 | `-ncmoe 8` (pre-NUMA job) | **DONE** gen=28.86 |
| F2 | `-ncmoe 16` (pre-NUMA job) | **DONE** gen=28.77 |
| F3 | `-ncmoe 32` + sock0 DDR NUMA | **DONE** gen=24.60 |
| F4 | `-ncmoe 99` + sock0 DDR NUMA | **DONE** gen=**33.28** (≥30) |
| F4h | `-ncmoe 99` + sock0 HBM preferred | **DONE** gen=**33.78** (best F) |

| Cycle | Job | Change | TTFT_ms | prefill_tps | gen_tps | bench_ttft_ms | bench_prefill_tps | bench_gen_tps | Notes |
|-------|-----|--------|--------:|------------:|--------:|--------------:|------------------:|--------------:|-------|
| **F0** | **8686725** | 1-tile + `-ncmoe 99` (unbound) | **315.86** | **63.32** | **28.56** | — | — | — | pre-NUMA baseline; bench fail |
| F1 | **8686948** | `-ncmoe 8` | **271.00** | **73.80** | **28.86** | — | — | — | bench abort; likely pre-NUMA |
| F2 | **8686949** | `-ncmoe 16` | **271.31** | **73.72** | **28.77** | — | — | — | bench abort |
| F3 | **8687060** | `-ncmoe 32` + DDR NUMA | **296.90** | **67.36** | **24.60** | — | — | — | NUMA on; worse gen |
| **F4** | **8687085** | `-ncmoe 99` + DDR NUMA + `-t 32` | **308.58** | **64.81** | **33.28** | — | — | — | **≥30**; sock0 DDR |
| **F4h** | **8687285** | `-ncmoe 99` + HBM pref + `-t 32` | **234.66** | **85.23** | **33.78** | — | — | — | **best F**; `--preferred=2` |
| **P25** | **8686713** | P24 SoA + IGC | **457.49** | **43.72** | **29.94** | — | — | — | quality PASS |

**Phase F verdict:** sock0 NUMA + full MoE→CPU clears **30 tok/s** (F4/F4h). HBM-preferred slightly ahead. Bench still aborts with `--numa` on llama-bench (harness now strips it).

### Phase G — FINAL (max content length + NUMA) — **DONE**

Queues: **debug / debug-scaling only**. G0/G1 used `N_CTX=131072` + `FILL_CTX_TOKENS=8192` (59 m budget).

| Cycle | N_CTX | Mode | Fill tok | TTFT_ms | prefill_tps | gen_tps | Quality | Notes |
|-------|------:|------|--------:|--------:|------------:|--------:|---------|-------|
| **G0** | 131072 | 2-tile pure GPU | ~5837 | **12562** | **464.65** | **28.61** | PASS | sock0 NUMA; bench fail |
| **G1** | 131072 | 1-tile MoE→CPU | ~5837 | **17887** | **326.32** | **32.66** | PASS | **best long-ctx decode**; ≥30 |
| G0s | 65536 | 2-tile | ~46k | **113265** | **406.95** | **24.24** | PASS | full-ish fill |
| G1s | 65536 | 1-tile MoE | ~46k | **173420** | **265.79** | **22.37** | PASS | full-ish fill |
| G0t | 32768 | 2-tile | ~22905 | **49269** | **464.89** | **26.16** | PASS | |
| G1t | 32768 | 1-tile MoE | ~22905 | **73777** | **310.46** | **27.16** | PASS | |

**Phase G verdict:** At max `N_CTX=131072` (8k prompt), **1-tile MoE→CPU + sock0 NUMA (G1) gen=32.66** beats **2-tile pure GPU (G0) gen=28.61**. Prefill favors 2-tile. Quality PASS on all arms.

**Campaign complete** (Phase F + G). Best recipes frozen in [`BEST_RECIPE.md`](BEST_RECIPE.md).
