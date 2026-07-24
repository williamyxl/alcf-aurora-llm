# Best recipes — llama.cpp SYCL Inkling UD-IQ1_S (Aurora)

Frozen after Phase S + Phase M harvest (2026-07-24). Queues: **debug / debug-scaling only**.

## Short context (`N_CTX=4096`)

| Arm | gen_tps | prefill_tps | TTFT_ms | Notes |
|-----|--------:|------------:|--------:|-------|
| **PG10** pure GPU 10-tile | **6.73** | 8.70 | 2300 | **best single-node decode** |
| **PG14** hybrid local12+RPC2 | **6.76** | 9.25 | 2161 | **best hybrid decode** |
| PG8–PG24 | 6.41→5.46 | — | — | ladder **complete**; more remote tiles → slower decode |

Min pure-GPU floor: **PG8**. Min MoE: **MO1** (`-ncmoe 99`, HBM `--preferred=2`) gen 6.43 / prefill **11.62**.

## Max context (`N_CTX=131072`, 8k FILL)

### Single-node — viable

| Arm | gen_tps | prefill_tps | TTFT_ms | Notes |
|-----|--------:|------------:|--------:|-------|
| **C_PG8** | 6.09 | 59.64 | 97870 | 8k fill PASS |
| **C_PG10** | 5.97 | 59.90 | 97439 | 8k fill PASS |
| **C_PG12** | 5.66 | 59.00 | 98927 | 8k fill PASS |
| **C_MO1** | 5.61 | 36.26 | 160980 | MoE max-ctx PASS |
| C_MO_HBM | NA | NA | NA | **FAIL_OOM** — `--membind=2` kill (exit 137) |

Prefer **C_PG8–10** for max-ctx single-node (best gen among max-ctx pure GPU).

### Multi-node hybrid RPC — not viable with fill

| Arm | Result | Notes |
|-----|--------|-------|
| C_PG14–20 | FAIL_OOM | FA-off V-pad 2048 @ 131k; remote OUT_OF_RESOURCES |
| **C_PG22** FILL (8693927) | **FAIL_OOM** | 8k fill wrote; load OK; OOM at generate |
| **C_PG24** FILL (8693776) | **FAIL_OOM** | same pattern |
| C_PG22 short20 | PASS* | alloc-only; **does not** count as max-ctx fill |

**Conclusion:** multi-node max-ctx with true fill is **not viable at TP 22/24** (nor 14–20) under current RPC FA-off + V-pad constraints.

## Constraints / hybrid recipe

- **RPC requires `-fa off`** (banded FA asserts over RPC).
- FA-off → **V-cache padded to 2048** at 131k (large KV; drives multi-node OOM).
- Hybrid: **local SYCL on dense node (≤12 tiles) + `--rpc` to remote `ggml-rpc-server` only** (do not put local tiles behind loopback RPC).
- `LLAMA_CACHE` on flare; `llama_max_devices` / `GGML_SCHED_MAX_BACKENDS` → 48 for TP>16.
- MoE: use `--preferred=2` (HBM), **not** `--membind=2`.
