# Inkling llama.cpp SYCL campaign plan

See Cursor plan + this file. **Queues: debug / debug-scaling only — use both in parallel.**

## Queue policy
| Queue | Max / user | Use for |
|-------|------------|---------|
| **debug** | 1 job (≤2 nodes) | Prefer **MoE** (`MO*`, `C_MO*`, `MO_HBM`); single-node overflow |
| **debug-scaling** | 1 job (≤256 nodes) | Prefer **pure-GPU**; **all multi-node RPC** (`PG14+`, `C_PG14+`) **must** use this queue |

`harvest_once.sh --submit` fills **both** free slots when possible. TP>12 always uses `bench_llamacpp_sycl_rpc_multinode.pbs` on **debug-scaling**.

## Tile policy (Inkling)
| Mode | Tile floor | Why |
|------|------------|-----|
| **Pure GPU** (`-ngl 99`, experts on GPU) | **TP ≥ 8** | Full weight fit; PG1/2/4 OOM; skip PG6 / C_PG1–C_PG6 |
| **MoE→CPU** (`-ncmoe 99`) | **TP &lt; 8 OK** (try **MO1→MO2** first) | Experts on host; attention/non-MoE on GPU — 1–2 tiles expected to load |

## Pure-GPU scale ladder (short ctx)
Test **TP = 8, 10, 12, 14, 16, 18, 20, 22, 24** (even steps). **Do not skip.** Record TTFT / prefill_tps / gen_tps each run.

| TP | Cycle | Nodes | Backend | Status |
|---:|-------|------:|---------|--------|
| 8 | **PG8** | 1 | single-node SYCL | **DONE** gen=6.41 |
| 10 | **PG10** | 1 | single-node SYCL | **DONE** gen=6.73 |
| 12 | **PG12** | 1 | single-node SYCL | **DONE** gen=6.55 |
| 14 | **PG14** | 2 | **SYCL+RPC multinode** | retest (prior run was node-local 12-tile only) |
| 16 | **PG16** | 2 | **SYCL+RPC** | pending |
| 18 | **PG18** | 2 | **SYCL+RPC** | pending |
| 20 | **PG20** | 2 | **SYCL+RPC** | pending |
| 22 | **PG22** | 2 | **SYCL+RPC** | pending |
| 24 | **PG24** | 2 | **SYCL+RPC** | pending |

Aurora: **12 tiles / node**. TP≤12 → single-node `build-llamacpp-sycl`.  
TP≥14 → **separate** tree `build-llamacpp-sycl-rpc` (`-DGGML_RPC=ON` + SYCL) on **`debug-scaling`** only:
- `build_llamacpp_sycl_rpc.pbs` — build `ggml-rpc-server` + `llama-completion`
- `bench_llamacpp_sycl_rpc_multinode.pbs` — 1× rpc-server per node, client `--rpc host0:port,host1:port`

Max-ctx pure GPU mirrors the same ladder: **C_PG8 … C_PG24** (C_PG14+ also RPC on debug-scaling).

## Matrix
| Phase | Cycles | Rule |
|-------|--------|------|
| S1 min PG | **PG8** (done) | First pure-GPU PASS at ≥8 |
| S2 scale PG | **PG10→12→14→16→18→20→22→24** | Full ladder above |
| S3 min MO | **MO1→MO2** (TP&lt;8) | Min MoE tiles; **no** MO scale; NUMA `--preferred=2` |
| M1–M3 | **C_PG8…C_PG24** / **C_MO1→C_MO2** | Same at N_CTX=131072 (+ FILL) |
| **S4 HBM bind** | **MO_HBM** / **C_MO_HBM** | Winning min-MO + `--membind=2` |

## Hardware defaults
- Always `GGML_SYCL_ENABLE_VMM=0`
- Pure GPU: `-fa on -sm layer` even `-ts`; **TP≥8**; ladder to **24**
- MoE: `-ncmoe 99 -t 32` + `numactl --physcpubind=1-51,105-155 --preferred=2`; **TP&lt;8 allowed**
- Do not scale MoE past min working tile count

## Phase S4 — MoE on CPU-HBM (`--membind=2`) — **both models**
Same final arm for **Inkling** (this tree) and **gpt-oss-120b** (`workdir/llm/gpt-oss-120b/…`): force host MoE pages onto **Xeon Max HBM**, not DDR5.

Default MoE arms use `--preferred=2` (sock0 HBM preferred; overflow may land on DDR NUMA 0).  
**Final test:** same recipe as the S3 min-MO winner (Inkling) / F4h+G1 (gpt-oss), but pin with **`--membind=2`**:

```bash
# Aurora Flat NUMA: 0/1 = DDR sock0/1 ; 2/3 = HBM sock0/1 (64G each)
# GPU0 / ZE_AFFINITY_MASK=0 → socket 0 → HBM node 2
numactl --physcpubind=1-51,105-155 --membind=2 \
  llama-completion ... -ngl 99 -sm none -fa on -ncmoe 99 -t 32 --numa numactl --no-mmap
```

| Model | Short ctx | Max ctx | Notes |
|-------|-----------|---------|-------|
| **Inkling** | `MO_HBM` | `C_MO_HBM` | After S3/M3; retarget tiles to S3 winner if not MO1 |
| **gpt-oss-120b** | `F4_hbm` | `G1_hbm` | After F4h / G1; MoE ~50–60 G may fit in 64 G HBM |

- Expect OOM if MoE RSS ≫ 64 G HBM; still record TTFT/prefill/gen (or FAIL_OOM)
- Compare vs `--preferred=2` (F4h / MO*) and DDR `--membind=0` (F4 / G1)

## Success
Quality PASS + full TTFT/prefill/gen rows for every completed cycle in `CYCLE_LOG.md`, including the pure-GPU scale ladder through TP=24 (or documented multi-node blocker) and the final HBM-membind MoE arm.
