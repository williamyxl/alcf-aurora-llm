# Session recovery — vLLM-XPU bring-up + slowness investigation (Aurora PVC)

**Last updated:** 2026-08-15 ~21:45 UTC — **PAUSED.** No jobs queued/running.

## UPDATE 2 (21:45) — MXFP4 refocus + source-built PVC kernels

Per user: **keep MXFP4, forget FP16** (casting is NOT the 30× cause). Investigation deepened:

- **Isolation proved** the prebuilt `vllm 0.27.1` / `vllm_xpu_kernels 0.1.13.2` wheels **SIGSEGV in the
  forward on PVC for ANY model** (even dense opt-125m, both FLASH_ATTN and TRITON_ATTN, with/without
  prefix-cache) — a prebuilt-wheel↔Xe-HPC mismatch, not MoE/checkpoint specific.
- **Basic ops work** (torch-XPU matmul, `silu_and_mul`, kernel import) — only full model forward crashes.
- **Built vllm_xpu_kernels from source AOT-targeted for PVC** (CMake supports `pvc`). Fixed the
  `mem_info.cpp` Level-Zero compile error with the project's `patches/mem_info.cpp.aurora-ze-fallback`.
  **Build succeeded** (`0.1.dev1+g13013c599`, job 8758898). Source tree:
  `build-vllm-xpu/build-src/vllm-xpu-kernels`.
- **But source-built kernels still SIGSEGV** (job 8758912) at engine-core init: **oneAPI ABI split** —
  kernels built with Aurora `icpx` 2025.3 vs torch's bundled oneAPI 2026 runtime. Loading the 2025.3
  module to satisfy the kernels breaks torch (`libsycl urDeviceWaitExp`); not loading it, the kernels
  crash. See `VLLM_INVESTIGATION.md §3c`.

### THE key next step (start here)
The two oneAPI versions must match. Options:
1. **Build torch-XPU from source against Aurora oneAPI 2025.3** (project's original stack route), so
   torch + source-built kernels share one ABI. Heavy but correct.
2. Find/install a **torch-XPU wheel built against oneAPI 2025.3** (matching Aurora icpx), then rebuild
   kernels against it.
3. **[TRY FIRST — cheapest, most promising]** Install the **oneAPI 2026 DPC++ compiler via pip** to
   match torch's bundled 2026 runtime, then rebuild kernels WITHOUT the 2025.3 module → one ABI.
   - Env currently has only runtime `dpcpp-cpp-rt 2026.0.0` (no compiler). Verified: no `icpx` in env.
   - Try: `pip install dpcpp-cpp-rt==2026.0.0` already present; the compiler pkg is
     `intel-dpcpp-cpp-compiler` (or `dpcpp_cpp`) — search PyPI/intel channel. If a 2026 `icpx` installs,
     set `CMAKE_CXX_COMPILER=icpx` from that path, do **not** `module load oneapi`, keep `env/lib`
     first on `LD_LIBRARY_PATH`, rebuild `build_kernels_pvc.pbs`.
   - If no pip compiler: fall back to option 1 (build torch-XPU from source with Aurora oneAPI 2025.3).

### New files this update
- `vllm_bench2.py` now: plain-prompt fallback (no chat template), `VLLM_BENCH_SIMPLE_SCHED` env.
- `build_kernels_pvc.pbs` — source build (AOT pvc, `-j32`, full log to `.pipfull`, mem_info patch applied).
- `kernel_probe.pbs` — proves basic ops work on PVC.
- `models/opt-125m` — tiny model for XPU-path isolation.
- Env patch #3 note: `oracle/unquantized.py` prefers Triton MoE on XPU (still crashes; keep for BF16).

### vLLM job ledger (append)
| 8758589 | BF16 TP=4 Triton-MoE | load OK → SIGSEGV forward |
| 8758626/29/42 | opt-125m TP=1 (triton/flash/simple) | SIGSEGV forward (dense, no MoE) |
| 8758636 | kernel probe | matmul+silu_and_mul OK |
| 8758898 | **source kernel build (pvc)** | **BUILD OK** after mem_info patch |
| 8758912 | MXFP4 TP=2 source kernels | SIGSEGV at engine init (oneAPI ABI split) |

---

**(historical) Last updated:** 2026-08-15 ~15:58 UTC.
**Author:** Kilo (Claude Opus 4.8)
**Read this first** to resume without redoing work. Companion docs in this dir:
`VLLM_INVESTIGATION.md` (root cause), `RESULTS.md` (llama.cpp numbers), `PLAN.md` (E1–E9),
`E9_parity_audit.md`.

---

## TL;DR state

1. **Codebase migrated** MOFA→MatSciAI: **DONE** (0 refs in code/text). Do not redo.
2. **Models downloaded** to `gpt-oss-120b/models/` and `inkling/models/`: **DONE**
   - gpt-oss MXFP4 GGUF (63GB), gpt-oss HF MXFP4 (183GB), gpt-oss **BF16** (218GB, 73 shards, no quant), Inkling UD-IQ1_S GGUF (270GB).
3. **llama.cpp SYCL** built + best recipes benched: **DONE** (see RESULTS.md).
   - gpt-oss F4_hbm gen **41.8 tok/s**; P14_tp2 **34.0**; Inkling MO1 **6.1**, PG10 **6.46**.
4. **vLLM-XPU runtime** built from wheels (vllm 0.27.1 + torch 2.13+xpu): **DONE, imports+loads model**.
5. **ROOT CAUSE of vLLM failure/slowness FOUND** (see below): MXFP4 MoE kernel is Xe2/Xe3-only and
   PVC misreports as Xe2 → SIGSEGV. PVC has **no native FP4/FP8** (only INT4/INT8), so MXFP4 has no
   correct kernel on this GPU.
6. **In progress:** getting vLLM to *run* on **BF16** gpt-oss (no MXFP4) to then measure/debug perf.
   Last job **8758532** (BF16 TP=8 Triton-MoE, mem_util 0.80) — check its result first on resume.

---

## Environments (all under MatSciAI)

| Env | Path | Purpose |
|-----|------|---------|
| aurora-llm | `/lus/flare/projects/MatSciAI/xiaoliyan/software/conda/envs/aurora-llm` | downloads, cmake/ninja, llama.cpp build |
| **vllm-xpu** | `/lus/flare/projects/MatSciAI/xiaoliyan/software/conda/envs/vllm-xpu` | **vLLM 0.27.1 + torch 2.13+xpu + vllm_xpu_kernels 0.1.13.2 + triton 3.7.2+xpu** |
| miniforge3 base | `/lus/flare/projects/MatSciAI/xiaoliyan/miniforge3` | conda root; `source .../etc/profile.d/conda.sh` |

Install cmd used (reproducible):
```
pip install "vllm==0.27.1" --extra-index-url=https://wheels.vllm.ai/xpu/ --extra-index-url=https://download.pytorch.org/whl/xpu
pip install "vllm_xpu_kernels==0.1.13.2" --extra-index-url=https://wheels.vllm.ai/xpu/
```
torch 2.13+xpu bundles oneAPI 2026 runtime → **do NOT `module load frameworks` or `oneapi`**
(loading oneapi 2025.3 breaks torch: `libsycl.so.9 undefined symbol urDeviceWaitExp`).

---

## ROOT CAUSE (why vLLM crashes / is slow vs llama.cpp)

**Hardware fact (user-confirmed + Intel BigDL-LLM docs):** Max 1550 (PVC, Xe-HPC) has **no native
FP4/FP8** tensor units — only **INT4 (maybe INT8)**. gpt-oss-120b ships only MXFP4 (or BF16/FP16);
no INT4/INT8 checkpoint exists.

**vLLM 0.27.1 code path:** both MXFP4 and BF16 unquantized MoE route to `XpuFusedMoe` via
`vllm/model_executor/layers/fused_moe/experts/xpu_moe.py`. `XPUExperts.__init__` (and
`XPUExpertsMxFp4.__init__`) gate on:
```python
is_xe2_or_xe3 = torch.ops._xpu_C.is_xe2_arch() or torch.ops._xpu_C.is_xe3_arch()
if not is_xe2_or_xe3: raise NotImplementedError("...only supported on Intel Xe2/Xe3 GPUs")
```
**Measured on PVC (job 8758399):** `is_xe2_arch() = True` (WRONG — PVC is Xe-HPC). The false positive
lets the Xe2-only MXFP4/MoE SYCL kernels run on PVC → **SIGSEGV** in the profiling forward.

**Consequence:** on PVC, vLLM MoE is either
- old stack: forced REF (dequant+matmul) → correct but ~1.2 tok/s (25–35× slower than llama.cpp), or
- new stack (0.27.1): fused Xe2 kernel → **crashes**.
The `VLLM_XPU_FUSED_MOE_USE_REF=1` env is **not recognized** in 0.27.1 (warns "Unknown env var").

llama.cpp is fast because its MXFP4 GGUF kernels use INT-friendly SYCL paths that actually run on PVC.

---

## Bring-up fixes already applied (env patches — persist in the env, re-apply if env rebuilt)

1. `vllm/v1/worker/xpu_worker.py` (~line 104): skip warmup `all_reduce` when `world_size==1`
   (bundled oneccl-2022 ATL can't init single-rank).
2. `vllm/platforms/xpu.py` `get_mem_info_wrapper` (~line 95): fall back to `torch.xpu.mem_get_info`
   when custom `getMemoryInfo` returns free<=0 (fixed "Free memory 0.0 GiB" profiler bug).
3. `vllm/model_executor/layers/fused_moe/oracle/unquantized.py` (~line 95): on XPU prefer
   `[TRITON, BATCHED_TRITON, XPU]` (env `VLLM_XPU_FORCE_TRITON_MOE=1`, default on) so BF16 MoE avoids
   the crashing Xe2 XPU kernel. **NOTE: Triton MoE also SIGSEGV'd at TP=8 (job 8758510) — see open items.**

Runtime env that made vLLM init work (in `vllm_run.pbs`):
- Launch under `mpiexec -n TP` (PALS PMI) + `--distributed-executor-backend external_launcher`.
- `CCL_PROCESS_LAUNCHER=pmix`, `CCL_ATL_TRANSPORT=mpi`, `CCL_ZE_IPC_EXCHANGE=sockets`, `FI_PROVIDER=cxi`.
- `ZES_ENABLE_SYSMAN=1`, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`.
- `attention_backend="TRITON_ATTN"` (LLM kwarg; env var is ignored). FLASH_ATTN default garbles/segfaults gpt-oss.
- PMI→torch rank mapping done in `vllm_bench2.py` (`PALS_RANKID→RANK` etc.).

---

## Harness files (this dir: `build-vllm-xpu/opus4.8-new_campaign/`)

| File | Purpose |
|------|---------|
| `vllm_bench2.py` | vLLM 0.27.1 bench: TTFT/prefill/decode/e2e + quality; `--attention-backend`, `--cpu-offload-gb`, `--kv-cache-memory-gib`, `--distributed-executor-backend`; rank-0-only PERF_JSON; PMI env mapping |
| `vllm_run.pbs` | PBS driver. `-v MODEL,TP,MML,MNS,MEMUTIL,KVGIB,ATTN,OFFLOADGB,TAG` |
| `xpu_mem_probe.pbs` | proves mem_get_info works (68GB free) vs vLLM's 0.0 bug |
| `arch_probe.pbs` | proves `is_xe2_arch()=True` on PVC (root cause) |
| `VLLM_INVESTIGATION.md` | full write-up |

Run example (BF16, avoids MXFP4):
```
BF16=/lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/gpt-oss-120b/models/openai-gpt-oss-120b-bf16
qsub -q debug-scaling -v MODEL=$BF16,TP=8,MML=4096,MNS=2,MEMUTIL=0.80,KVGIB=8,ATTN=TRITON_ATTN,TAG=bf16_tp8 \
  -N vg-bf16 -o build-vllm-xpu/opus4.8-new_campaign/logs/<name>.out \
  build-vllm-xpu/opus4.8-new_campaign/vllm_run.pbs
```

---

## Job ledger (vLLM)

| Job | Config | Result |
|-----|--------|--------|
| 8758319 | MXFP4 TP=2 4k | CCL fixed; "Free memory 0.0" |
| 8758342 | MXFP4 TP=2 4k, mem patch | weights load OK; **SIGSEGV @ XPUExpertsMxFp4** |
| 8758364 | MXFP4 TP=2, TRITON_ATTN kwarg | Triton attn OK; **SIGSEGV @ MXFP4 MoE** |
| 8758399 | arch probe | **is_xe2_arch()=True on PVC** (root cause) |
| 8758405 | MXFP4 TP=2 + cpu_offload 20G | SIGSEGV @ MoE (offload doesn't help) |
| 8758469 | **BF16** TP=8 4k | uses XPU Unquantized MoE → **SIGSEGV** |
| 8758510 | BF16 TP=8 + Triton-MoE patch | uses TRITON MoE → **SIGSEGV** (ranks 3/7), rest SIGKILL |
| 8758524 | BF16 TP=4 | load OK (54GiB/tile); **SIGKILL 9 = host OOM after load** |
| 8758532 | BF16 TP=8 mem_util0.80 Triton-MoE | **SIGKILL 9 (host OOM) ~101s in**, before load finished; used TRITON MoE |

**All vLLM runs to date FAIL.** MXFP4 → SIGSEGV (Xe2 kernel on PVC). BF16 → SIGSEGV (Triton MoE
at TP=8, job 8758510) or SIGKILL host-OOM (TP=4 job 8758524; TP=8 job 8758532). No successful
generation yet. See OPEN ITEMS for the two independent blockers still to clear:
(a) MoE kernel PVC-incompatibility, (b) host-RAM OOM during BF16 load.

---

## OPEN ITEMS / next steps (priority order) — START HERE NEXT SESSION

Two independent blockers remain. Both must be cleared for a working vLLM run.

**Blocker A — MoE kernel PVC-incompatibility (SIGSEGV):**
- XPU native MoE (`XPUExperts`) = Xe2-only → segfaults on PVC (jobs 8758342/8758469).
- Triton MoE (our oracle patch) = segfaulted at TP=8 (8758510) but that run *also* had host-OOM
  cascade, so Triton-MoE-on-PVC is **not yet cleanly proven bad**. Re-test Triton MoE at a config
  that does NOT host-OOM (see Blocker B) to isolate.
- If Triton MoE truly segfaults: try (i) `VLLM_XPU_FORCE_TRITON_MOE=0` and patch `is_xe2_arch`→False
  so `XPUExperts` raises NotImplementedError, then wire a naive torch experts fallback in
  `oracle/unquantized.py` for XPU; or (ii) check triton 3.7.2+xpu JIT on PVC with a 1-layer probe
  (project history: needed Triton 3.6, not 3.8).

**Blocker B — host-RAM OOM during BF16 load (SIGKILL 9):**
- TP=4 loaded 54GiB/tile then SIGKILL (8758524); TP=8 SIGKILL ~101s into load (8758532).
- 8 ranks on ONE node each stage BF16 shards in host RAM → exceeds node RAM. Mitigations to try:
  - `MEMUTIL` lower doesn't help host RAM. Instead reduce **host** pressure:
    set `VLLM_XPU...`/loader streaming, or `--load-format` streaming, or set
    `SAFETENSORS`/mmap load (avoid page-cache prefetch: patch `_prefetch_all_checkpoints` off or
    set `num_prefetch_threads`/disable).
  - Try **TP=4 with `cpu_offload_gb`** small + `MEMUTIL 0.9` (fewer ranks = less host staging).
  - Consider FP16 mirror `twhitworth/gpt-oss-120b-fp16` (not downloaded) — same size, no change to OOM.
  - Best lever: **reduce concurrent host staging** — investigate a vLLM env to serialize per-rank
    weight load, or load with fewer ranks then rely on TP sharding.

**Then (once a run completes):**
- Record TTFT / prefill tok/s / decode tok/s at **4096** and **max ctx**, for **pure-GPU** (TP=4/8)
  and **CPU-offload** (`OFFLOADGB=...`). Fill `RESULTS.md` vLLM section + `VLLM_INVESTIGATION.md §4`.
- Profile the vLLM↔llama.cpp gap (PLAN.md E1) once BF16 generates.
- Inkling under vLLM: check if `architectures` in its HF config is supported by vLLM 0.27.1 (GGUF-only
  UD-IQ1_S is not an HF checkpoint; would need HF weights). Likely llama.cpp-only for now.

## Constraints / gotchas
- **Per-user debug queue limit = 1 running/queued job.** Serialize submits. Kill strays with `qdel`.
- **Do NOT `module load frameworks`/`oneapi`** in the vllm-xpu env (breaks torch SYCL).
- Poll pattern: `qstat -f <JID> | grep job_state`; when gone, `qstat -xf <JID> | grep Exit_status`.
- Logs mix stale content across reruns; filter by fresh timestamp (`awk '/HH:MM/{p=1}p'`).
- Debug-scaling queue OK; each run ≈ queue + 2–4 min load. BF16 load ≈ 100s.
- `PYTHONFAULTHANDLER` dumps huge C-stacks; filter with `grep -avE "bytecodes|call.c|ceval|_Py|Objects/|Python/|Modules/"`.

## Model paths
- gpt-oss MXFP4 GGUF: `.../gpt-oss-120b/models/openai-gpt-oss-120b-mxfp4.gguf` (symlink)
- gpt-oss HF MXFP4: `.../gpt-oss-120b/models/openai-gpt-oss-120b` (config has quantization_config=mxfp4)
- **gpt-oss BF16 (use this for vLLM):** `.../gpt-oss-120b/models/openai-gpt-oss-120b-bf16` (quant=None)
- Inkling GGUF: `.../inkling/models/unsloth-Inkling-GGUF/UD-IQ1_S/inkling-UD-IQ1_S-00001-of-00007.gguf`
- (Optional FP16 mirror not downloaded: `twhitworth/gpt-oss-120b-fp16` via `download_gptoss_bf16_fp16.sh fp16`)

## UPDATE 3 (2026-08-15 23:10) — Path A exhausted; Path B (oneAPI 2026) chosen

- Path A (older vllm + oneAPI 2025.x): **dead end for prebuilt.** vllm XPU wheels only exist for
  recent versions (0.27.1). vllm 0.11.0/0.11.2 have **no XPU wheel** — pip pulled the PyPI **CUDA**
  build (`vllm._C` needs libcudart.so.12 → "Device string must not be empty"). Older XPU vllm would
  require building vllm from source (VLLM_TARGET_DEVICE=xpu), hours. Env `vllm-xpu2` (torch 2.8 + ipex
  2.8.10 + vllm 0.11.0-CUDA) is not usable; can delete.
- **Decision: Path B.** Keep working stack = **vllm 0.27.1 + torch 2.13+xpu (env `vllm-xpu`)**. Its
  only blocker is the forward SIGSEGV, root-caused to the oneAPI ABI split (kernels need 2026 icpx to
  match torch's bundled 2026 runtime; Aurora only has 2025.3 icpx).
- **User installing oneAPI 2026.** Once available:
  1. In `build_kernels_pvc.pbs`: replace `module load oneapi/release/2025.3.1` with the 2026 module
     (or `source <2026>/setvars.sh`); keep env/lib first on LD_LIBRARY_PATH; keep mem_info patch +
     AOT pvc. Rebuild vllm_xpu_kernels from source (`build-src/vllm-xpu-kernels`, HEAD 13013c5).
  2. Run `vllm_run.pbs` (env vllm-xpu) MXFP4 TP=2 4096 → expect no SIGSEGV.
  3. Then push tok/s >30: TP sweep (2/4/8), fused vs REF MoE, max_num_seqs batching, enable XPU graph
     (`VLLM_XPU_ENABLE_XPU_GRAPH=1`).
- Exact need: `icpx` **2026.0.0** (matches `intel-sycl-rt==2026.0.0` in torch 2.13 pins).
