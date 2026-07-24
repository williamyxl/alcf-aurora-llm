# Inkling CYCLE_LOG — TTFT / prefill_tps / gen_tps

Queues: **debug / debug-scaling only**. Do not condense without asking.

**Columns:** Cycle | N_CTX | tiles | mode | affinity | TTFT_ms | prefill_tps | gen_tps | quality | notes

## Setup

| Step | Status | Notes |
|------|--------|-------|
| Download UD-IQ1_S | **OK** | 270.2 GB, 7 shards; job 8688509 |
| SYCL build | **OK** | llama.cpp `cf51256` + PR **#25731** (inkling arch) |
| RPC build | **OK** | `build-llamacpp-sycl-rpc`; `llama_max_devices`/`GGML_SCHED_MAX_BACKENDS` → 48 |

---

## Phase S — short context (`N_CTX=4096`)

### Single-node pure GPU / MoE

| Cycle | N_CTX | tiles | mode | affinity | TTFT_ms | prefill_tps | gen_tps | quality | notes |
|-------|------:|------:|------|----------|--------:|------------:|--------:|---------|-------|
| PG1 (attempt1) | 4096 | 1 | pure GPU | 0 | NA | NA | NA | FAIL | `llama-completion` rejects `--chat-template-kwargs`; harness switched to `--reasoning off` |
| PG1 (attempt2) | 4096 | 1 | pure GPU | 0 | NA | NA | NA | FAIL | invalid split file name — must use `…-00001-of-00007.gguf` not flat symlink |
| PG1 (attempt3) | 4096 | 1 | pure GPU | 0 | NA | NA | NA | FAIL | `unknown model architecture: 'inkling'` — mainline tip lacks arch; rebuilding from **PR #25731** |
| **PG1** | 4096 | 1 | pure GPU | 0 | NA | NA | NA | FAIL_OOM | Inkling build OK; `UR_RESULT_ERROR_OUT_OF_RESOURCES` (1 tile insufficient) |
| **PG2** | 4096 | 2 | pure GPU | 0,1 | NA | NA | NA | FAIL | `LLAMA_SPLIT_MODE_TENSOR not implemented for architecture inkling` — switch to `-sm layer` |
| **PG2** (layer) | 4096 | 2 | pure GPU `-sm layer` | 0,1 | NA | NA | NA | FAIL_OOM | `-sm layer` + equal `-ts`; still `UR_RESULT_ERROR_OUT_OF_RESOURCES` → escalate **PG4** |
| **PG4** (layer) | 4096 | 4 | pure GPU `-sm layer` | 0–3 | NA | NA | NA | FAIL_OOM | same OOM during load/RMS_NORM |
| **PG6** | 4096 | 6 | — | — | — | — | — | SKIPPED | pure-GPU floor ≥8; MoE arms still use TP&lt;8 (MO1/MO2) |
| **PG8** | 4096 | 8 | pure GPU `-sm layer` | 0–7 | **2456** | **8.14** | **6.41** | PASS | **min pure-GPU**; load ~202 s; MOF-ish text OK; scale ladder → TP=10…24 |
| **PG10** | 4096 | 10 | pure GPU `-sm layer` | 0–9 | **2300** | **8.70** | **6.73** | PASS | vs PG8: slightly faster gen/prefill; load ~183 s |
| **PG12** | 4096 | 12 | pure GPU `-sm layer` | 0–11 | **2764** | **7.24** | **6.55** | PASS | full node; gen between PG8/PG10; load ~203 s |
| **MO1** | 4096 | 1 | MoE→CPU `-ncmoe 99` | 0 | **1722** | **11.62** | **6.43** | PASS | **min MoE**; HBM `--preferred=2`; load ~335 s; prefill best so far |
| **MO2** | 4096 | 2 | — | — | — | — | — | SKIPPED | MO1 sufficient; no MO scale |
| **MO_HBM** | 4096 | 1 | MoE→CPU HBM bind | 0 | NA | NA | NA | FAIL_OOM | `--membind=2` (sock0 HBM ~64 G); process **Killed** during load (exit 137) — MoE RSS ≫ HBM; MO1 `--preferred=2` still OK |

### Multi-node hybrid RPC (TP ≥ 14)

Hybrid recipe: local SYCL on dense node (≤12 tiles) + `--rpc` to remote `ggml-rpc-server` only. Force `-fa off` (banded FA asserts over RPC).

| Cycle | N_CTX | tiles | mode | affinity | TTFT_ms | prefill_tps | gen_tps | quality | notes |
|-------|------:|------:|------|----------|--------:|------------:|--------:|---------|-------|
| **PG14** | 4096 | 14→12 | pure GPU (2-node alloc, **no RPC**) | local 0–11 | **3334** | **6.00** | **6.28** | PASS* | *invalid for TP=14* — archived `perf_PG14.out.node_local_12tile` |
| **PG14** (RPC attempt1) | 4096 | 14 | RPC 2-node | 12+2 | NA | NA | NA | FAIL | node0 rpc: $HOME cache mkdir fail; node1 then `FLASH_ATTN_EXT_BANDED` assert (exit 134). Fix: `LLAMA_CACHE` on flare + readiness gate; resubmit |
| **PG14** (RPC attempt2) | 4096 | 14 | RPC 2-node | 12+2 | NA | NA | NA | FAIL | readiness false-timeout (stdout buffer); servers SIGTERM before client. Fix: stdbuf -oL + longer wait; resubmit |
| **PG14** (RPC attempt3) | 4096 | 14 | RPC 2-node | 12+2 | NA | NA | NA | FAIL | HSN self-connect miss on 12-tile node + `FLASH_ATTN_EXT_BANDED` on 2-tile server. Fix: loopback local RPC + `-fa off` |
| **PG14** (RPC attempt4) | 4096 | 14 | RPC 2-node | 12+2 | NA | NA | NA | FAIL | loopback OK; CLI rejected `--fa` (need `-fa off`) |
| **PG14** (RPC attempt5) | 4096 | 14→2 | RPC all-remote (broken) | 2 only | **2158** | **9.27** | **6.99** | PASS* | *invalid TP=14* — 12-tile local RPC got 0 accepts; only 2-tile remote ran. Switch to hybrid local SYCL+remote RPC |
| **PG14** | 4096 | 14 | hybrid local12+RPC2 `-fa off` | 0–11 + remote | **2161** | **9.25** | **6.76** | PASS | first valid TP=14; load ~273 s; remote rpc accepted=12 |
| **PG16** (attempt1) | 4096 | 16 | hybrid | 12+4 | NA | NA | NA | FAIL | `-ts` 16 entries rejected: `llama_max_devices()` hardcoded 16 (`size >= max`). Patch → 48; rebuild RPC |
| **PG16** (attempt2) | 4096 | 16 | hybrid | 12+4 | NA | NA | NA | FAIL | `GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS)` (default 16). Bump sched max→48; rebuild |
| **PG16** | 4096 | 16 | hybrid local12+RPC4 `-fa off` | 0–11 + remote | **2305** | **8.68** | **6.35** | PASS | load ~369 s; remote rpc accepted=20 |
| **PG18** | 4096 | 18 | hybrid local12+RPC6 `-fa off` | 0–11 + remote | **2458** | **8.14** | **6.06** | PASS | load ~432 s; remote rpc accepted=28 |
| **PG20** | 4096 | 20 | hybrid local12+RPC8 `-fa off` | 0–11 + remote | **2579** | **7.75** | **5.87** | PASS | load ~568 s; remote rpc accepted=36 |
| **PG22** | 4096 | 22 | hybrid local12+RPC10 `-fa off` | 0–11 + remote | **2716** | **7.36** | **5.65** | PASS | load ~642 s; remote rpc accepted=44 |
| **PG24** | 4096 | 24 | hybrid local12+RPC12 `-fa off` | 0–11 + remote | **2845** | **7.03** | **5.46** | PASS | load ~675 s; short ladder complete |

**Phase S short verdict:** ladder **PG8–PG24** complete. Best short pure-GPU gen: **PG10 6.73**. Best short hybrid: **PG14 6.76**. More remote tiles → slower decode (RPC overhead).

---

## Phase M — max context (`N_CTX=131072`, 8k FILL)

### Single-node

| Cycle | N_CTX | tiles | mode | affinity | TTFT_ms | prefill_tps | gen_tps | quality | notes |
|-------|------:|------:|------|----------|--------:|------------:|--------:|---------|-------|
| **C_PG8** | 131072 | 8 | pure GPU max-ctx | 0–7 | **97870** | **59.64** | **6.09** | PASS | 8k fill |
| **C_PG10** | 131072 | 10 | pure GPU max-ctx | 0–9 | **97439** | **59.90** | **5.97** | PASS | 8k fill; similar to C_PG8 |
| **C_PG12** | 131072 | 12 | pure GPU max-ctx | 0–11 | **98927** | **59.00** | **5.66** | PASS | 8k fill; full node |
| **C_MO1** | 131072 | 1 | MoE→CPU max-ctx | 0 | **160980** | **36.26** | **5.61** | PASS | 8k fill; load ~322 s; min MoE at max ctx |
| **C_MO_HBM** | 131072 | 1 | MoE→CPU HBM bind max-ctx | 0 | NA | NA | NA | FAIL_OOM | same `--membind=2` kill (exit 137) during load |

### Multi-node hybrid RPC (TP ≥ 14)

RPC forces `-fa off` → V cache padded to 2048 at 131k (large KV). `FILL_CTX` was missing from RPC PBS until after C_PG22; fixed thereafter.

| Cycle | N_CTX | tiles | mode | affinity | TTFT_ms | prefill_tps | gen_tps | quality | notes |
|-------|------:|------:|------|----------|--------:|------------:|--------:|---------|-------|
| **C_PG14** (attempt1) | 131072 | 14 | hybrid 12+2 FA-off f16 KV | 0–11 + remote | NA | NA | NA | FAIL_OOM | remote OUT_OF_RESOURCES RMS_NORM; FA-off V-pad 2048 at 131k |
| **C_PG14** (attempt2) | 131072 | 14 | hybrid 12+2 FA-off ctk+ctv q8 | 0–11 + remote | NA | NA | NA | FAIL | V cache quant requires flash_attn (incompatible with RPC FA-off) |
| **C_PG14** (attempt3) | 131072 | 14 | hybrid 12+2 FA-off ctk q8 | 0–11 + remote | NA | NA | NA | FAIL_OOM | same remote OUT_OF_RESOURCES; V-pad 2048 still dominates |
| **C_PG14** (attempt4) | 131072 | 14 | hybrid 12+2 FA-off `-nkvo` | 0–11 + remote | NA | NA | NA | FAIL_OOM | same remote OUT_OF_RESOURCES RMS_NORM; nkvo ineffective |
| **C_PG14** | 131072 | 14 | hybrid RPC max-ctx | — | NA | NA | NA | FAIL_OOM | exhausted mitigations; proceed C_PG16+ |
| **C_PG16** | 131072 | 16 | hybrid 12+4 FA-off | 0–11 + remote | NA | NA | NA | FAIL_OOM | remote OUT_OF_RESOURCES RMS_NORM (same as C_PG14) |
| **C_PG18** | 131072 | 18 | hybrid 12+6 FA-off | 0–11 + remote | NA | NA | NA | FAIL_OOM | remote OUT_OF_RESOURCES RMS_NORM |
| **C_PG20** | 131072 | 20 | hybrid 12+8 FA-off | 0–11 + remote | NA | NA | NA | FAIL_OOM | remote OUT_OF_RESOURCES OP SCALE |
| **C_PG22** (short20) | 131072 | 22 | hybrid 12+10 FA-off | 0–11 + remote | **3096** | **6.46** | **5.65** | PASS* | *alloc OK* but prompt was **20 tok** (no FILL) — archived `perf_C_PG22.out.short20tok_nofill` |
| **C_PG22** | 131072 | 22 | hybrid 12+10 FA-off + FILL | 0–11 + remote | NA | NA | NA | FAIL_OOM | job **8693927**; FILL_CTX 8192 (~8220 tok est) wrote OK; load OK (V-pad 2048) then `UR_RESULT_ERROR_OUT_OF_RESOURCES` at generate start (ggml-sycl.cpp:3665); COMPLETION_EXIT=1; N_PROMPT=NA |
| **C_PG24** | 131072 | 24 | hybrid 12+12 FA-off + FILL | 0–11 + remote | NA | NA | NA | FAIL_OOM | job **8693776**; FILL_CTX 8192 (~8220 tok est) wrote OK; load OK then `UR_RESULT_ERROR_OUT_OF_RESOURCES` at generate start (ggml-sycl.cpp:3665); COMPLETION_EXIT=1; N_PROMPT=NA |

**Phase M multi-node status:** C_PG14–20 FAIL_OOM under FA-off V-pad. **C_PG22** and **C_PG24** with true 8k FILL → FAIL_OOM at generate (`OUT_OF_RESOURCES`). Short-prompt C_PG22 PASS* (20 tok) does not count as max-ctx fill. **Multi-node max-ctx with fill not viable at TP 22/24** (nor 14–20).
