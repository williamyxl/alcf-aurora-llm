# Multi-node SYCL+RPC build (separate from single-node)

Path: `workdir/llm/inkling/build-llamacpp-sycl-rpc/`  
Source: symlink → `../build-llamacpp-sycl/llama.cpp` (Inkling PR #25731)  
Build flags: `-DGGML_SYCL=ON -DGGML_RPC=ON` (does **not** overwrite single-node `build/`)

## Queue
**debug-scaling only** for RPC build + all TP>12 jobs.

## Jobs
| Script | Role |
|--------|------|
| `../build_llamacpp_sycl_rpc.pbs` | Build `ggml-rpc-server` + `llama-completion` |
| `../bench_llamacpp_sycl_rpc_multinode.pbs` | 2-node: rpc-server per node, client `--rpc` |

## Ladder (restored — do not skip)
PG14 → PG16 → PG18 → PG20 → PG22 → PG24 (and C_PG14…C_PG24 after max-ctx single-node arms).
