# vLLM-XPU on Aurora PVC — root-cause investigation (2026-08-15)

**Author:** Kilo (Claude Opus 4.8)
**Goal:** Rebuild a working vLLM runtime locally and investigate why vLLM is so slow vs llama.cpp.
**Result:** Built a modern vLLM-XPU stack from wheels; fixed 3 Aurora bring-up blockers; then hit and
**root-caused** the core failure: the MXFP4 MoE kernel is fundamentally wrong for PVC hardware.

---

## 1. Runtime built (from wheels, no from-source torch)

Fresh conda env `software/conda/envs/vllm-xpu` (py3.12):

| Component | Version | Source |
|-----------|---------|--------|
| vllm | 0.27.1 | wheels.vllm.ai/xpu |
| torch | 2.13.0+xpu | download.pytorch.org/whl/xpu (bundles oneAPI 2026 rt) |
| triton | 3.7.2+xpu | pulled by torch |
| vllm_xpu_kernels | 0.1.13.2 | wheels.vllm.ai/xpu |

This is **much newer** than the project's 2026-07 self-built stack and needs **no `module load
frameworks`** (torch carries its own oneAPI 2026 runtime).

## 2. Aurora bring-up blockers fixed (in order hit)

| # | Symptom | Root cause | Fix |
|---|---------|-----------|-----|
| B1 | `SETVARS_CALL: unbound variable` | env activate.d script under `set -u` | export SETVARS_CALL / drop `set -u` |
| B2 | `oneCCL atl_ofi_comm init_transport ... failed to initialize ATL` (even TP=1) | bundled `oneccl-2022` ATL can't init without a PMI/KVS; vLLM does a warmup `all_reduce` even at world_size=1 | launch under Aurora PALS `mpiexec -n TP` + `CCL_ATL_TRANSPORT=mpi` + `external_launcher`; map `PALS_RANKID→RANK`; patch `xpu_worker.py` to skip the single-rank warmup collective |
| B3 | `Free memory on device xpu:0 (0.0/60.79 GiB)` | vLLM XPU memory profiler calls `torch.ops._C_cache_ops.getMemoryInfo` which returns **free=0** on PVC | `ZES_ENABLE_SYSMAN=1` + patch `platforms/xpu.py get_mem_info_wrapper` to fall back to `torch.xpu.mem_get_info` (returns correct 68 GB) |
| B4 | wrong attention backend | env `VLLM_ATTENTION_BACKEND` ignored; gpt-oss defaults to FLASH_ATTN (garbles/segfaults on XPU) | pass `LLM(attention_backend="TRITON_ATTN")` kwarg |

After B1–B4, vLLM **loads gpt-oss-120b successfully** (weights 31 GiB/tile, `Using Triton backend`,
`Using XPUExpertsMxFp4`) and reaches the profiling forward pass.

## 3. ROOT CAUSE of the crash (and of the whole vLLM-XPU MoE story)

The profiling forward **SIGSEGVs** inside the MXFP4 MoE. Source path:

`vllm/model_executor/layers/fused_moe/experts/xpu_moe.py` → `XPUExpertsMxFp4.__init__`:

```python
is_xe2_or_xe3 = torch.ops._xpu_C.is_xe2_arch() or torch.ops._xpu_C.is_xe3_arch()
if not is_xe2_or_xe3:
    raise NotImplementedError("XPUExperts is only supported on Intel Xe2/Xe3 GPUs")
```

**Measured on PVC (Max 1550), job 8758399:**
```
dev = Intel(R) Data Center GPU Max 1550
is_xe2_arch() = True      <-- WRONG. PVC is Xe-HPC, not Xe2.
is_xe3_arch() = False
```

The arch probe **false-positives** as Xe2, so the guard passes and vLLM runs **Xe2 MXFP4 MoE
kernels on PVC** → SIGSEGV (the kernels are compiled/tuned for a different ISA).

**Confirmed by hardware reality (user + Intel docs):** Max 1550 (PVC) has **no native FP4/FP8
tensor support** — only INT4 (and possibly INT8). So an MXFP4 fused MoE kernel has no correct
hardware path on this GPU regardless of the arch flag. The Xe2/Xe3 kernels exist because MXFP4 is
validated on Arc B-series (Xe2) / Panther-Lake (Xe3), **not** on Xe-HPC PVC.

Ref: Intel — Accelerating LLMs on Intel GPUs (BigDL-LLM): PVC supports INT4/INT8, not FP4/FP8.

### Why this also explains "vLLM slow vs llama.cpp"
- The project's older stack avoided the crash only by forcing a **reference (unfused) MoE**
  (`VLLM_XPU_FUSED_MOE_USE_REF=1`) — dequant + `torch.matmul` — which is correct but ~1.2 tok/s
  (25–35× slower than llama.cpp's INT-friendly MXFP4 GGUF kernels).
- In vLLM 0.27.1, `VLLM_XPU_FUSED_MOE_USE_REF` is **no longer recognized** (warned "Unknown vLLM
  environment variable"); the only MoE path is the fused `XPUExpertsMxFp4`, which crashes on PVC.
- So on PVC, MXFP4 in vLLM is either **slow (old REF)** or **broken (new fused)** — both because the
  quant format doesn't match the hardware's native INT4/INT8 units.

## 3b. UPDATE (2026-08-15 PM): prebuilt 0.27.1 wheels SIGSEGV on PVC for ANY model

Isolation experiments proved the failure is **not** MoE- or checkpoint-specific:

| Test | Model | Attn | MoE | Result |
|------|-------|------|-----|--------|
| 8758589 | gpt-oss BF16 TP=4 | TRITON | Triton (patched) | load OK → **SIGSEGV** in forward |
| 8758626 | **opt-125m** (dense, no MoE) TP=1 | TRITON | n/a | **SIGSEGV** in forward |
| 8758629 | opt-125m TP=1 | FLASH | n/a | **SIGSEGV** in forward |
| 8758642 | opt-125m TP=1, no prefix-cache/chunked-prefill | FLASH | n/a | **SIGSEGV** in forward |
| 8758636 | kernel probe | — | — | torch-XPU matmul OK; `_C`/`_moe_C` import OK; `silu_and_mul` OK |

**Conclusion:** basic vllm_xpu_kernels ops and torch-XPU compute work on PVC, but the **full model
forward (attention + KV) SIGSEGVs on PVC regardless of model, attention backend, or MoE path.** The
prebuilt `vllm 0.27.1` / `vllm_xpu_kernels 0.1.13.2` / `torch 2.13+xpu` wheels are validated on
**Xe2/Xe3 client GPUs** and are **not compatible with Xe-HPC PVC (Max 1550)**. `is_xe2_arch()` also
false-positives on PVC, so the library mis-targets kernels. This is a **prebuilt-wheel/hardware
mismatch**, on top of the FP4-not-native fact.

**Decision (user):** keep **MXFP4** (dtype casting is NOT the 30× cause — BF16 was ~3 tok/s). FP16
path abandoned. A working vLLM on PVC needs either the **project's older self-built stack** (which ran
MXFP4 REF ~1.2 tok/s) or **vllm_xpu_kernels built from source for Xe-HPC/PVC**. Prebuilt wheels are a
dead end on this hardware.

## 3c. UPDATE 2 (2026-08-15 late): source-built PVC kernels — compile fixed, still SIGSEGV

Pursued building `vllm_xpu_kernels` from source AOT-targeted for PVC (the CMake default
`AOT_DEVICES` includes `pvc`; `is_xe2_arch()` intentionally groups `intel_gpu_pvc`, so PVC *is* a
supported target using the `csrc/xpu/attn/xe_2/` kernels).

- **Compile blocker (fixed):** `csrc/utils/mem_info.cpp` uses
  `ze_device_usablemem_size_ext_properties_t` / `currUsableMemSize`, absent from Aurora's Level-Zero
  headers. Applied the project's existing `patches/mem_info.cpp.aurora-ze-fallback` (guards with
  `#if defined(...)`, falls back to `getTotalMemory`). **Build then SUCCEEDED**:
  `vllm-xpu-kernels-0.1.dev1+g13013c599` installed (job 8758898, PIP_RC=0).
- **Still SIGSEGV at runtime** (job 8758912, MXFP4 TP=2): the source-built `.so` was linked with the
  oneAPI **module** (icpx 2025.3) loaded at build time, but runtime uses torch's bundled oneAPI 2026
  (module unloaded). The resulting libsycl/libur ABI mismatch (same `urDeviceWaitExp` family) crashes
  during engine-core init, **before** weight load. Prebuilt-wheel path crashed later (in the forward);
  source path crashes earlier (link/ABI).

**Net:** two oneAPI toolchains are in play and incompatible:
- torch 2.13+xpu bundles **oneAPI 2026** runtime (needed for torch to import).
- Aurora's `icpx` compiler is **oneAPI 2025.3** (only way to build SYCL kernels here).
Kernels built with 2025.3 don't load cleanly against torch's 2026 runtime, and loading the 2025.3
module to satisfy them breaks torch. This toolchain split is the practical wall for a from-scratch
vLLM-XPU on Aurora with the current wheels.

**Open path (next session):** build torch-XPU **from source with the same oneAPI 2025.3** as the
kernels (the project's original `build_vllm_xpu_*` intent), so torch + kernels share one ABI — OR find
a torch-XPU wheel built against oneAPI 2025.3. This is the project's documented self-built-stack route;
the prebuilt-wheel shortcut cannot reconcile the two oneAPI versions on PVC.

## 4. The fix direction (per hardware reality)

**Do not use MXFP4 on PVC.** Options, best-first for a *working+faster* vLLM:
1. **FP16/BF16 gpt-oss checkpoint** (no `quantization_config`) → vLLM runs a standard dequantized
   MoE, avoiding the FP4 kernel entirely. Downloading `unsloth/gpt-oss-120b-BF16`. FP16 fits in HBM
   across enough tiles; this is the path the user directed ("stick with FP16, debug vLLM perf").
2. INT4/INT8 checkpoint matched to PVC's native units — **none exists for gpt-oss-120b** today.
3. llama.cpp SYCL remains the fast production path (34–42 tok/s, see `RESULTS.md`).

## 5. Patches applied to the env (record for reproducibility)

- `vllm/v1/worker/xpu_worker.py`: skip warmup `all_reduce` when `world_size == 1`.
- `vllm/platforms/xpu.py` `get_mem_info_wrapper`: fall back to `torch.xpu.mem_get_info` when the
  custom `getMemoryInfo` op returns free<=0.

Both are Aurora-portability shims, not perf changes.

## 6. Harness

- `vllm_bench2.py` — TTFT/prefill/decode/e2e + quality gate; PMI env mapping; external_launcher; rank-0-only PERF_JSON; `--attention-backend`, `--cpu-offload-gb`, `--kv-cache-memory-gib`.
- `vllm_run.pbs` — PALS `mpiexec -n TP`, CCL=mpi, TRITON_ATTN, sysman, no oneapi module.
- Probes: `xpu_mem_probe.pbs` (mem-info bug), `arch_probe.pbs` (is_xe2 false-positive).

## 7. Job ledger

| Job | Config | Outcome |
|-----|--------|---------|
| 8758319 | TP=2 4k, first CCL success | `Free memory 0.0` (B3) |
| 8758342 | TP=2 4k, mem patch | weights loaded; SIGSEGV at MoE |
| 8758357 | TP=2 4k, TRITON_ATTN env | still FLASH_ATTN (B4); SIGSEGV |
| 8758364 | TP=2 4k, attn kwarg | `Using Triton backend`; SIGSEGV at XPUExpertsMxFp4 |
| 8758399 | arch probe | **is_xe2_arch()=True on PVC (root cause)** |
| 8758405 | TP=2 4k + cpu_offload 20G | SIGSEGV at XPUExpertsMxFp4 (offload doesn't avoid FP4 kernel) |

## 8. Next

Run vLLM on **BF16** gpt-oss (downloading), TP sized to fit FP16 in HBM, TRITON_ATTN, and record
TTFT/prefill/decode at 4096 + max ctx; then compare to llama.cpp and profile the remaining gap.
