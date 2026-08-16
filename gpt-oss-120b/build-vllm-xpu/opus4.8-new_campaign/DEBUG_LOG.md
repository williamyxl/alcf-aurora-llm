# vLLM-XPU on Aurora PVC — timestamped debug log (problem → solution → result)

Goal: get a working vLLM inference for gpt-oss-120b MXFP4 on Aurora Max 1550 (PVC / Xe-HPC),
then push decode >30 tok/s. Each entry: symptom, root cause, fix tried, outcome, job id.

Hardware/software fixed facts:
- Max 1550 = PVC / Xe-HPC. **No native FP4/FP8** (INT4/INT8 only). (user + Intel BigDL-LLM docs)
- torch 2.13.0+xpu (vLLM 0.27.1) hard-pins **oneAPI 2026.0** runtime.
- Aurora modules provide oneAPI **2025.3** only. User installed oneAPI **2026.1** at
  `/lus/flare/projects/MatSciAI/xiaoliyan/software/intel/oneapi`.
- llama.cpp SYCL reference (works): gpt-oss MXFP4 decode ~34–42 tok/s (see RESULTS.md).
- vLLM quality-OK REF path historically ~1.2 tok/s (25–35× slower) → motivation.

Env: `vllm-xpu` = vLLM 0.27.1 + torch 2.13.0+xpu + triton 3.7.2+xpu (+ source-built kernels).
Queue: **debug-scaling only** (leave `debug` for the other agent's MACE jobs). Per-user max_run=1.

---

## P1 — conda MPI activation crashes under `set -u`
- **When:** 2026-08-15 ~06:57
- **Symptom:** `mpivars.activate.sh: line 16: SETVARS_CALL: unbound variable`; job exit 1 in <25s.
- **Cause:** env `activate.d` script references unset var; PBS script used `set -u`.
- **Fix:** drop `set -u`; `export SETVARS_CALL=0`, `I_MPI_ROOT=`.
- **Result:** PASS — activation clean.

## P2 — oneCCL ATL init fails (even TP=1)
- **When:** 2026-08-15 ~07:09–07:20
- **Symptom:** `oneCCL: atl_ofi_comm.cpp:848 init_transport: failed to initialize ATL`; also
  `MPI_Init_thread ... failed` / `write_line error fd=9` when wrapped in `mpiexec -n 1`.
- **Cause:** bundled `oneccl-2022.0.0` ATL transport needs a PMI/KVS; vLLM does a warmup
  `all_reduce` even at world_size=1. `CCL_ATL_TRANSPORT=ofi` (default) can't init on Aurora without
  PMI; `mpi` transport needs a real MPI/PMI context that vLLM's spawned EngineCore lacked.
- **Fixes tried:**
  1. `CCL_ATL_TRANSPORT=mpi` + `mpiexec -n1` → PMI handshake failed (child proc). ❌
  2. `CCL_ATL_TRANSPORT=ofi` + `FI_PROVIDER=tcp` + launcher=none → ATL still wants local_idx from PMI. ❌
  3. **Launch under PALS `mpiexec -n TP` + `--distributed-executor-backend external_launcher`**, map
     `PALS_RANKID→RANK/LOCAL_RANK/WORLD_SIZE`, `CCL_PROCESS_LAUNCHER=pmix`, `CCL_ATL_TRANSPORT=mpi`,
     `FI_PROVIDER=cxi`. ✅ (for TP≥2)
  4. For TP=1: **patch** `vllm/v1/worker/xpu_worker.py` to skip the warmup `all_reduce` when
     `world_size==1`. ✅
- **Result:** PASS — oneCCL initializes; `CCL_ATL_TRANSPORT: mpi`, both ranks init xccl (job 8758319).

## P3 — vLLM XPU memory profiler reads 0 free
- **When:** 2026-08-15 ~13:29
- **Symptom:** `ValueError: Free memory on device xpu:0 (0.0/60.79 GiB) ... less than gpu_memory_utilization`.
- **Cause:** `vllm/platforms/xpu.py get_mem_info_wrapper` calls
  `torch.ops._C_cache_ops.getMemoryInfo(device)` which returns **free=0** on PVC. (probe 8758330 showed
  `torch.xpu.mem_get_info` returns correct 68 GB free per tile.)
- **Fix:** `ZES_ENABLE_SYSMAN=1` + **patch** `get_mem_info_wrapper` to fall back to
  `torch.xpu.mem_get_info` when the custom op returns free<=0.
- **Result:** PASS — memory profiling passes, weights load (job 8758342: "Model loading took 31.11 GiB").

## P4 — wrong attention backend (FLASH_ATTN) for gpt-oss
- **When:** 2026-08-15 ~13:55–14:16
- **Symptom:** `xpu.py:205 Using Flash Attention backend` despite `VLLM_ATTENTION_BACKEND=TRITON_ATTN`;
  gpt-oss docs say FLASH garbles/segfaults on XPU.
- **Cause:** in vLLM 0.27.1, attention backend is an `LLM(attention_backend=...)` **kwarg**, not the env var.
- **Fix:** pass `attention_backend="TRITON_ATTN"` kwarg (0.27.x); use env for 0.11.x.
- **Result:** PARTIAL — `xpu.py:169 Using Triton backend` confirmed (job 8758364), but forward still SIGSEGV
  (see P6). Rules out attention-backend-selection as the crash.

## P5 — prebuilt vllm_xpu_kernels wheel SIGSEGV in forward on PVC (any model)
- **When:** 2026-08-15 14:00–17:16
- **Symptom:** after weights load, **SIGSEGV in the profiling forward** for gpt-oss MXFP4 (8758342/64),
  gpt-oss BF16 (8758469/510), and even **dense opt-125m** (8758626 TRITON, 8758629 FLASH, 8758642
  no-prefix-cache). Basic ops OK (probe 8758636: matmul, silu_and_mul, import).
- **Cause:** prebuilt `vllm_xpu_kernels 0.1.13.2` wheel is validated for Xe2/Xe3 client GPUs; its
  AOT binaries don't include a working PVC path → crashes in the full forward. Also `is_xe2_arch()`
  returns **True** on PVC (source intentionally groups `intel_gpu_pvc` into xe2), so kernels mis-target.
- **Fix direction:** build vllm_xpu_kernels **from source AOT-targeted for PVC**.
- **Result:** confirmed prebuilt wheels unusable on PVC → move to source build.

## P6 — source build blocked: Level-Zero header mismatch
- **When:** 2026-08-15 21:05
- **Symptom:** `csrc/utils/mem_info.cpp: error: unknown type 'ze_device_usablemem_size_ext_properties_t'`,
  `ZE_STRUCTURE_TYPE_DEVICE_USABLEMEM_SIZE_EXT_PROPERTIES`, `currUsableMemSize`.
- **Cause:** Aurora Level-Zero headers lack the usable-mem ext properties the kernel source assumes.
- **Fix:** apply project's `patches/mem_info.cpp.aurora-ze-fallback` (guards with `#if defined(...)`,
  falls back to `getTotalMemory`). Also needed: `pip install setuptools-scm cmake ninja` in env,
  reduce `-j` from 208→32 (host OOM risk).
- **Result:** PASS — build compiled + installed (job 8758898, PIP_RC=0) BUT verify segfaulted (→ P7).

## P7 — oneAPI ABI split: kernels built with 2025.3 vs torch's 2026 runtime
- **When:** 2026-08-15 21:18 / 23:36
- **Symptom:** source-built kernels (2025.3 icpx) SIGSEGV at import/engine-init;
  `libsycl.so.9: undefined symbol: urDeviceWaitExp, version LIBUR_LOADER_0.12` when the 2025.3 module
  was loaded (breaks torch's bundled 2026 libsycl).
- **Cause:** torch 2.13+xpu bundles oneAPI **2026** runtime; Aurora `icpx` is **2025.3**. Two oneAPI
  versions can't share one ABI. Loading 2025.3 module breaks torch; not loading it, the 2025.3-built
  kernels crash.
- **Fix:** **user installed oneAPI 2026.1**; rebuild kernels with 2026.1 `icpx` to match torch's 2026 runtime.
- **Result:** unblocked P7 (see P8–P10 for the 2026 build chain).

## P8 — 2026 build: CMPLR_ROOT empty → `/bin/icpx` not found
- **When:** 2026-08-15 23:42
- **Symptom:** `CMAKE_CXX_COMPILER: /bin/icpx is not a full path`.
- **Cause:** kernels' `cmake/toolchain.cmake` uses `$CMPLR_ROOT/bin/icpx`; `setvars.sh` set it in a
  subshell but pip's build subprocess lost it. Also Aurora's 2025.3 icpx was winning on PATH.
- **Fix:** `export CMPLR_ROOT=$ONEAPI2026/compiler/latest`; prepend its bin to PATH; `module unload
  oneapi`; set absolute `CC/CXX/CMAKE_*_COMPILER`.
- **Result:** PASS — `icpx=2026.1.0` confirmed in build log.

## P9 — 2026 build: CMake 4.x + Ninja RPATH relink error
- **When:** 2026-08-15 23:47
- **Symptom:** `CMake Error at cmake/utils.cmake:566 (add_library): The install of grouped_gemm_xe_2
  requires changing an RPATH ... not supported with the Ninja generator`.
- **Cause:** cmake 4.4.2 is stricter about build→install RPATH relink with Ninja.
- **Fix:** add `set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)` in CMakeLists.txt (after `CMAKE_INSTALL_RPATH`).
- **Result:** PASS — configure completes, proceeds to compile.

## P10 — 2026 build: SYCL header needs modern libstdc++ (`<version>`)
- **When:** 2026-08-15 23:57
- **Symptom:** `sycl/bit_cast.hpp:26:10: fatal error: 'version' file not found`.
- **Cause:** 2026 icpx uses system g++ 7.5 (no `<version>` C++ header) as host toolchain.
- **Fix:** `--gcc-toolchain=<Aurora GCC 13.4>` in SYCL flags (via `GCC_TOOLCHAIN` env + CMakeLists
  `list(APPEND SYCL_FLAGS ...)`); add GCC13.4 `lib64` to LD_LIBRARY_PATH; GCC path:
  `/opt/aurora/26.26.0/spack/unified/1.1.1/install/linux-x86_64/gcc-13.4.0-hgnyg4p`.
- **Result:** **PASS (BIG)** — full build succeeded (job 8759067, ~1h, PIP_RC=0). Verify:
  `kernels import OK, dev Max 1550`; `silu_and_mul OK`. **ABI split resolved.**

## P11 — MXFP4 MoE still SIGSEGV in forward (post-ABI-fix)
- **When:** 2026-08-16 01:07 (job 8759119), 01:30 (8759144)
- **Symptom:** kernels import fine, model loads (31 GiB, `XPUExpertsMxFp4`), then **SIGSEGV in forward**.
- **Hypothesis A:** cutlass grouped-GEMM takes the xe2 branch on PVC (`grouped_gemm_interface.cpp:34
  is_xe2_arch()||is_xe3_arch()` → `cutlass_grouped_gemm_xe2`), which crashes on Xe-HPC.
- **Fix tried:** `VLLM_XPU_FORCE_XE_DEFAULT_KERNEL=1` (C++ `getEnv`, routes to
  `cutlass_grouped_gemm_xe_default`). Env is read at kernel level (vLLM logs "Unknown vLLM env" — expected).
- **Result (8759144):** still SIGSEGV. Did not isolate the crash yet → P12.

## P12 — crash is NOT MoE-specific (isolation)
- **When:** 2026-08-16 01:36 (job 8759158 dense opt-125m), 01:43 (op probe 8759165)
- **Test:** dense **opt-125m** (no MoE) with the 2026-built kernels + TRITON_ATTN.
- **Result:** dense opt-125m **still SIGSEGV in forward** → the crash is in a path common to all models,
  i.e. **attention** or a fused op invoked with real forward shapes, NOT the MXFP4 MoE GEMM.
- **Op probe (8759165):** `rms_norm` OK, `silu_and_mul` OK. `_C` only exposes {rms_norm, silu_and_mul};
  attention/KV ops live in `_vllm_fa2_C` / `_moe_C` / triton — not yet probed.
- **Status:** OPEN. Next: probe the attention path (FA2 kernel `_vllm_fa2_C`, or the Triton attention
  JIT) with realistic shapes; try `VLLM_ATTENTION_BACKEND` variants and `enforce_eager` off/on; capture
  the faulting kernel via `ZE_DEBUG`/`unitrace` or a single-layer forward.

## P13 — crash is the ATTENTION kernel (Triton AND SYCL FA2), both xe2-targeted, crash on PVC
- **When:** 2026-08-16 01:51–02:20
- **Tests:**
  - opt-125m custom_ops disabled (8759173) → still SIGSEGV ⇒ not vllm `_C` rms/act ops.
  - **bare Triton kernel** probe (8759186/92, `triton_probe.py`) → **SIGSEGV in
    `triton/backends/intel/driver.py:364 init_devices`** (device enumeration / SYCL-L0 interop),
    even for a trivial add kernel. `TRITON_INTEL_DEVICE_EXTENSIONS` did not help.
  - opt-125m FLASH_ATTN = SYCL FA2 (my PVC build) (8759202) → **SIGSEGV** too.
- **Cause:** two independent attention problems on PVC:
  1. **TRITON_ATTN**: triton 3.7.2+xpu `init_devices` segfaults on Aurora L0 (matches project history:
     needed triton 3.6; 3.7/3.8 break). vllm `_C` custom ops (SYCL) are fine; only Triton JIT device
     init crashes.
  2. **FLASH_ATTN**: vllm_xpu_kernels SYCL FA2 (`csrc/xpu/attn/attn_interface.cpp:33`) is **xe2-only**
     (`is_xe2_arch()||is_xe3_arch()` → `cutlass_chunk_prefill_xe2`), **no xe_default fallback** (unlike
     grouped GEMM which has `VLLM_XPU_FORCE_XE_DEFAULT_KERNEL`). The xe2 attention kernel crashes on Xe-HPC.
- **Status:** OPEN. Both stock attention backends are broken on PVC.
- **Options next:**
  a. **Fix Triton**: install triton **3.6** (project's known-good) compatible with torch 2.13, or patch
     `driver.py init_devices` to not segfault; then TRITON_ATTN works and MoE can use `xe_default`.
  b. **Add xe_default attention**: patch `attn_interface.cpp` to route PVC to a non-xe2 path (the repo
     may have `xe_default` attn like it does for grouped GEMM; if not, this is a kernel dev task).
  c. Accept vLLM-XPU is Xe2/Xe3-only for attention on this wheel line and report llama.cpp as the
     production path (already 34–42 tok/s).

---

## P14 — pinpointed: triton `sycl::get_native<level_zero>(device)` SIGSEGV on PVC
- **When:** 2026-08-16 02:41–02:56
- **Method:** instrumented `triton/backends/intel/driver.c init_devices` with fprintf markers (JIT-recompiles).
- **Result (job 8759234, FLAT):** `queue ok → zeInit ok → get_context ok → get_devices n=6 → SEGV`
  in the loop calling `sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_devices[i])`.
  COMPOSITE (8759237): `n=3 → SEGV` at same call. So hierarchy doesn't matter.
- **Cause:** triton 3.7.2's JIT-compiled `driver.c` and torch's in-process SYCL runtime resolve to
  **different libsycl instances / incompatible SYCL runtime**, so extracting the native L0 handle from a
  SYCL device crashes. `find_sycl` picks `icpx`-or-`intel-sycl-rt`; `INJECT_PYTORCH=True` (link torch's
  SYCL) did not fix it (cached module or still mismatched). `sycl_queue` int pointer itself is valid
  (probe 8759214), and `zeInit`/`get_context`/`get_devices` all succeed — only `get_native` faults.
- **Interpretation:** triton 3.7.2+xpu is not compatible with this torch 2.13 + Aurora L0 combo. Matches
  project history (needed triton **3.6**; 3.7/3.8 broke). Fixing needs a compatible triton, not a flag.
- **Status:** OPEN. Attention has no working path on PVC with this stack (Triton broken; SYCL FA2 xe2-only).

## Current state (2026-08-16 02:21)
- Attention is the wall: TRITON (triton 3.7.2 driver init SIGSEGV) and FLASH (xe2-only SYCL FA2) both
  crash on PVC. MoE grouped-GEMM has a `xe_default` escape; attention does not.

## (earlier) Current state (2026-08-16 01:44)
- ✅ Single-ABI source-built PVC kernels (oneAPI 2026.1 + GCC 13.4): import + basic ops work.
- ✅ oneCCL, memory profiler, attention selection, engine init all working; model loads.
- ❌ Forward pass SIGSEGV, now isolated to a **non-MoE common path (attention / fused op)**, affects
  even dense opt-125m.

## Next steps (priority)
1. Probe attention: build a minimal forward that calls the FA2/paged-attention op (`_vllm_fa2_C`) with
   gpt-oss-like shapes; or run vLLM with different `VLLM_ATTENTION_BACKEND` + `enforce_eager` combos.
2. If Triton attention JIT is the crasher: it mirrors the project's history (needed Triton 3.6, 3.8
   broke). Try alternate attention backend or a Triton version pin.
3. If FA2 SYCL kernel is the crasher: check its xe2/xe_default arch branch (same pattern as grouped GEMM)
   and whether `VLLM_XPU_FORCE_XE_DEFAULT_KERNEL` / an analogous knob covers attention.
4. Once forward runs: quality-gate gpt-oss MXFP4, record TTFT/prefill/decode at 4096 + max ctx, then
   push >30 tok/s (TP sweep, xe_default vs xe2, batching, XPU graph).

## Key files
- Build: `build_kernels_pvc.pbs` (oneAPI 2026.1 + GCC13.4 + mem_info patch + RPATH + AOT pvc).
- Run: `vllm_run.pbs` (PALS mpiexec + external_launcher, TRITON_ATTN, GCC13.4 lib, FORCE_XE_DEFAULT).
- Bench: `vllm_bench2.py`. Probes: `kernel_probe.pbs`, `arch_probe.pbs`, `fwd_probe.pbs`, `xpu_mem_probe.pbs`.
- Source: `build-src/vllm-xpu-kernels` (HEAD 13013c5 + patches).
- Env patches (in `vllm-xpu` site-packages): xpu_worker warmup-skip, xpu.py mem fallback,
  oracle/unquantized.py Triton-MoE-first.

## P15 — frameworks env (vllm 0.15 + torch 2.10 + triton 3.6 + ipex 2.10): IPEX Marlin path, new failure
- **When:** 2026-08-16 03:24–03:32
- **Discovery:** Aurora `frameworks/2025.3.1` conda env
  (`/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1`) has a **complete known-good XPU stack**:
  vllm 0.15.0, torch 2.10.0a0, triton 3.6.0, ipex 2.10.10. No build needed.
- **Key difference:** vllm 0.15 gpt-oss MXFP4 uses `mxfp4.py:167 Using ipex marlin backend on XPU`
  (IPEX Marlin), NOT the cutlass xe2 kernel. Attention = "Flash Attention backend". vllm also logs
  "Triton not installed or not compatible" and disables triton GPU funcs → avoids the triton crash.
- **Result (job 8759277, TP=2):** **model loads fine (33 GiB, 87s), NO segfault**. Then during forward
  the **IPEX Marlin kernel JIT-compiles** and throws
  `terminate: sycl::exception what(): No device of requested type available`. Different failure class
  (SYCL device selector), not a segfault.
- **Interpretation:** the IPEX Marlin MoE JIT kernel can't select the XPU at forward time. Likely a
  device-selector/env mismatch (`ONEAPI_DEVICE_SELECTOR=level_zero:gpu` + FLAT hierarchy vs what the
  JIT's default SYCL selector expects), or worker-process device context.
- **Status:** OPEN but PROMISING — this path avoids both prior crashes; needs the SYCL device selector fixed.
- **Next:** adjust device env for the IPEX JIT (try unset/relax `ONEAPI_DEVICE_SELECTOR`, ensure
  `SYCL_DEVICE_FILTER`/`ONEAPI_DEVICE_SELECTOR=level_zero:*`, set `-fsycl` device flags, or run TP=1 to
  remove worker/device-mask interplay). Also fix `vllm_bench2.py` for vllm 0.15 metrics API.

## P16 — IPEX Marlin "No device" persists across launch/selector variants
- **When:** 2026-08-16 03:35–03:58
- **Tests (frameworks env, vllm 0.15):**
  - TP=1 (8759285): loads then **XPU OOM** (gpt-oss MXFP4 62GB > one 64GB tile) — expected, no "No device".
  - TP=2 vllm mp executor, no mpiexec (8759291): loads both workers (33GiB) → IPEX Marlin forward
    `sycl::exception: No device of requested type available`.
  - TP=2 mp, `ONEAPI_DEVICE_SELECTOR` unset (8759298): same "No device".
- **Cause (working theory):** IPEX Marlin MoE JIT-compiles a SYCL kernel at forward time; that kernel's
  SYCL runtime/default-selector can't acquire the XPU inside the vllm worker process on Aurora. This is
  an IPEX-on-Aurora launch/runtime-context issue, likely requiring ALCF's specific vllm launch wrappers
  / SYCL env (not just conda activate + generic launch).
- **Status:** OPEN. frameworks stack avoids the segfaults (triton disabled, IPEX Marlin MoE, model loads)
  but the IPEX JIT device selection fails under generic launch.
- **Next:** find ALCF's documented `frameworks` vLLM serving launch (env/wrapper) for gpt-oss on Aurora;
  or set the IPEX/SYCL env the JIT needs (candidates: `IPEX_*`, `SYCL_PI_LEVEL_ZERO_*`,
  `ONEAPI_DEVICE_SELECTOR=level_zero:0,1` per-rank, disable IPEX marlin JIT and use a REF MoE).

## Status summary (2026-08-16 03:58)
Two independent working stacks explored on PVC:
- **self-built (vllm 0.27.1 + torch 2.13 + oneAPI2026 kernels):** ABI fixed, kernels+MoE ops work, but
  ATTENTION crashes (triton 3.7.2 get_native SIGSEGV; SYCL FA2 xe2-only).
- **frameworks (vllm 0.15 + torch 2.10 + triton 3.6 + ipex 2.10):** no segfault, IPEX Marlin MoE, model
  loads at TP=2; but IPEX Marlin JIT throws "No device" in the forward under generic launch.
Neither yet produces tokens. llama.cpp remains the working >30 tok/s path (34–42 tok/s).

## P17 — SOLVED: vLLM generates on PVC via frameworks module env (full env, not just conda activate)
- **When:** 2026-08-16 04:09 (job 8759303)
- **Fix:** run under the **full `module load frameworks` environment** (not just `conda activate` the env).
  The module sets `CCL_CONFIGURATION=cpu_gpu_dpcpp`, `CCL_ROOT`, `CMPLR_ROOT`, SYCL/oneAPI runtime on
  LD_LIBRARY_PATH, FI_CXI_* etc. that the **IPEX Marlin JIT kernel needs to select the XPU**. With only
  conda activate, the JIT threw "No device" (P16). With full module env → works.
- **Result:** gpt-oss-120b MXFP4 TP=2 4096, **coherent output** (quality_ok=True), decode:
  cold e2e 15.7 / warm 26.95 / **warm2 28.83 tok/s (e2e incl prefill)**; engine logger showed
  gen throughput ~17 tok/s mid-run. IPEX Marlin MoE + Flash Attention, REF MoE env set.
- **Stack:** frameworks/2025.3.1 = vllm 0.15.0 + torch 2.10 + triton 3.6 + ipex 2.10.10.
- **Note:** `vllm_bench2.py` TTFT/decode metrics are null on vllm 0.15 (RequestOutput.metrics API differs);
  e2e tok/s is valid. Need to fix metrics for true prefill/decode split.
- **THE working recipe.** Next: fix metrics, then push decode >30 (TP sweep, batching, xe tuning).

## P18 — TARGET MET: TP sweep, decode >30 tok/s
- **When:** 2026-08-16 04:14–04:42 (fixed two-call metrics in vllm_bench2.py)
- **gpt-oss-120b MXFP4, frameworks vLLM 0.15, MML=4096, MNS=1, warm2 decode:**
  | TP | decode tok/s (warm2) | cold decode | prefill tok/s | TTFT (ms) |
  |----|----------------------|-------------|---------------|-----------|
  | 2  | 29.6                 | 30.1        | 1288          | 68        |
  | **4** | **31.9**          | **33.4**    | **1671**      | **52**    |
  | 8  | 29.3                 | 30.8        | 1863          | 47        |
- **Best decode = TP=4 (31.9 warm / 33.4 cold tok/s), quality OK. >30 target MET.**
- TP=8 decode regresses (comm overhead) but has best prefill/TTFT. TP=4 is the decode sweet spot.
- All quality_ok=True (coherent MOF answers).

## Clarifications (2026-08-16 05:09)

### Why an earlier campaign (Cursor Grok-4.5) said frameworks couldn't run gpt-oss
Different **frameworks module version**:
- OLD `frameworks/2025.2.0` (Aurora 25.190.0, Oct 2025): **vllm 0.10.1rc2.dev189** — predates solid
  `GptOssForCausalLM` + XPU MXFP4 → no gpt-oss causal path (their conclusion was correct then).
- CURRENT `frameworks/2025.3.1` (Aurora 26.26.0, Feb 18 2026): **vllm 0.15.0 + torch 2.10 + triton 3.6
  + ipex 2.10** with `GptOssForCausalLM`, MXFP4, and **IPEX Marlin MoE** → works (31.9 tok/s).
The module was upgraded; that is the entire discrepancy.

### Why oneAPI 2026 "didn't help" (it did — for the other stack)
oneAPI 2026.1 fixed exactly its target: the **ABI split in the self-built stack** (vllm 0.27.1 + torch
2.13, which pins oneAPI 2026 runtime). After it, PVC kernels built + imported + ran basic ops (the
`urDeviceWaitExp` crash was gone; DEBUG_LOG P7–P10). It could not fix that stack's **second, unrelated**
blocker — attention (triton 3.7.2 `get_native` SIGSEGV; SYCL FA2 xe2-only, DEBUG_LOG P11–P14). Those are
Triton/kernel bugs, not ABI. The path that reached tokens (frameworks vllm 0.15) uses oneAPI **2025.3**
+ IPEX Marlin — a separate self-consistent stack that never needed 2026. Two stacks, two problems.

## P19 — Single-tile CPU-offload (llama.cpp F4_hbm analogue): NOT SUPPORTED on vLLM XPU
- **When:** 2026-08-16 04:56 (job 8759377)
- **Config:** TP=1, `--cpu-offload-gb 35`, ZE_AFFINITY_MASK=0, NUMA membind=2.
- **Result:** `AttributeError: '_OpNamespace' '_C' object has no attribute
  'get_cuda_view_from_cpu_tensor'` at engine init. vLLM's weight `cpu_offload_gb` uses a **CUDA-only UVA
  op** (`get_cuda_view_from_cpu_tensor`, model_executor/models/utils.py:666) not implemented in XPU `_C`.
- **Conclusion:** vLLM XPU has **no weight CPU-offload** (and no `-ncmoe`-style CPU-expert offload like
  llama.cpp). `swap_space` exists but is KV-cache CPU swap under pressure, not weight offload. So the
  llama.cpp F4_hbm single-tile+CPU-HBM recipe has **no vLLM equivalent on PVC**. Minimum single-node
  vLLM config is TP=2 (2 tiles hold the ~62 GB MXFP4 model).

## P20 — Max context (131072) works with negligible decode penalty
- **When:** 2026-08-16 05:06 (TP=4 job 8759384), TP=2 pending (8759404)
- **TP=4 MML=131072, KVGIB=40:** GPU KV cache = 1,942,976 tokens; decode **31.4 warm / 32.8 cold**,
  prefill 1656, TTFT 53 ms, quality OK. Essentially same as 4096 (gpt-oss sliding_window=128 keeps KV
  cost low). Model load 16.6 GiB/tile.

## P21 — llama.cpp F4_hbm (1-tile MoE→CPU + HBM) context sweep
- **When:** 2026-08-16 05:41 / 05:48
- **Recipe:** F4_hbm — `ZE_AFFINITY_MASK=0` (1 tile), `-ncmoe 99` (MoE experts on CPU), `-sm none`,
  `numactl --physcpubind=1-51,105-155 --membind=2` (HBM NUMA node 2), `-t 32`, `-fa on`, `--no-mmap`.
  Runner: `llama_f4hbm_ctx.pbs -v CTX=<n>`.
- **Results (gpt-oss-120b MXFP4 GGUF, N_PREDICT=128, quality OK):**
  | CTX | TTFT (ms) | prefill tok/s | decode tok/s | load (s) | job |
  |-----|-----------|---------------|--------------|----------|-----|
  | 4096 | 410.7 | 41.4 | **41.56** | 61.5 | 8759430 |
  | 131072 (max) | 371.6 | 45.8 | **38.07** | 50.7 | 8759441 |
- **Note:** decode drops ~8% at 128K context (larger KV) but stays ~38 tok/s. This is the TRUE
  single-tile + CPU-MoE-offload deployment (vLLM cannot do this on XPU — see P19).
