> **Planning document, written before execution.** The minimal-compile plan was carried out: see `docs/ENGINES.md` and `common/vllm/setup_overlay.sh`. Statements about what was 'expected' predate the measurements in the per-model READMEs.

# Plan: build the latest vLLM for XPU on Aurora (PVC / Max 1550)

Written 2026-11-15. Plan only: nothing in this file has been executed. Resources: `debug-scaling`, `-A MatSciAI`.
Network downloads are limited to what the build needs (pip wheels, git source). No model weights.

## 1. Why build (what the installed module can't do)

`frameworks/2026.1.0` ships vllm 0.26.1 (Aug 3 snapshot), torch 2.13, triton-xpu 3.7.2, vllm-xpu-kernels 0.1.11.2. It sits on read-only `/opt/aurora`.

| Gap found in testing | Root cause (verified in source) | Fixed by building |
|---|---|---|
| Nemotron-3-Ultra INT4: `Unsupported TopK value` | `vllm-xpu-kernels` `csrc/moe/remap_hidden_states.cpp` (`DISPATCH_TOPK_LAUNCH`) and `moe_gather.cpp` (`CASE_TOPK`) accept top-k in {1,2,3,4,6,7,8,10,12,16}. Ultra uses 22. `moe_align_sum_kernels.cpp:1368` accepts {1,2,3,4,6,8,9}. | Patch the tables to add 22 and rebuild `vllm-xpu-kernels` (the wheel can't be patched). |
| Nemotron-3-Ultra grouped top-k slow path | `csrc/moe/grouped_topk.cpp:43`: `SG_P2_ENABLE_SINGLE_NEMOTRON 0` (512 experts, top-22). The source warns "set to 1 and rebuild". | Flip to 1 in the rebuild. |
| Beam 1.0, Muse Spark: `model_type` unknown | `transformers` 5.14.1 in the module has no `beam` / `muse_spark`. Transformers 5.22.0 and 5.23.0 have both. vLLM added `models/beam` and `models/muse_spark` between v0.32.0 and v0.33.0rc3. | New vllm plus transformers >= 5.22. |
| Inkling: no XPU path | vLLM `models/inkling/nvidia` uses cutlass/FA4/Lamport (NVIDIA-only). `models/inkling/amd` exists but depends on ROCm-side code (see section 6). | Not fixed by building alone. |

## 2. Version selection

| Source | Version | torch | triton | vllm-xpu-kernels | Notes |
|---|---|---|---|---|---|
| vLLM `main` (2026-11-15) | `9fbe012` | 2.15.0+xpu | 3.9.0+xpu | 0.1.17 | newest, moving target |
| vLLM `v0.33.0rc3` (2026-11-02) | latest tag | 2.14.0 | 3.8.0+xpu | 0.1.15.4 | first tag with Beam and Muse Spark |
| vLLM `v0.32.0` (2026-10-18) | last stable | 2.14.0 | 3.8.0+xpu | 0.1.15.4 | no Beam / Muse |
| xpu-kernels `main` | `93f275f` (2026-11-13) | 2.15.0 | n/a | 0.1.18 dev | repo README requires torch 2.15.0+xpu |

**Recommendation: two-stage.**
- **Track A (low risk, do first):** vLLM `v0.33.0rc3` with torch 2.14.x+xpu, triton 3.8.0+xpu and the **pre-built** `vllm_xpu_kernels==0.1.15.4` wheel. Build only the vLLM wheel. This tests Beam / Muse Spark / Nemotron-H on a known-consistent set.
- **Track B (needed for the Nemotron top-k fix):** vLLM `main`, torch 2.15.0+xpu, triton 3.9.0+xpu, and **`vllm-xpu-kernels` built from source** with the top-k patch. Heavier, because the kernels build needs the oneAPI 2026.1 DPC++ compiler.

Pin by commit hash (not branch) once chosen, and record it in a `pins.env` as the earlier build did.

## 3. Environment

Facts checked on this machine:
- `oneapi/release/2026.1.0` provides `icpx` 2026.1.0.20260617. The xpu-kernels README asks for "Intel OMIX 0.4 including the oneAPI 2026.1 DPC++ compiler", so the version lines up.
- Driver stack: `intel_gpu_umd_aicoe` 2026.06.19 (igc 2.11.43). vLLM docs recommend compute runtime >= 26.18; I have not confirmed the runtime version here.
- cmake 3.28.3 (system) and 3.31.10 (in the frameworks env) both meet the >= 3.26 requirement. ninja 1.11.1 exists in the frameworks env.
- gcc 14.3.0 is available via module.
- Compute nodes: 104 cores, ~1 TB RAM, 495 GB `/tmp`. Login node reaches PyPI, the PyTorch XPU index, GitHub and `wheels.vllm.ai` through the proxy.
- Not installed: Rust, uv. Python 3.12 is required (the kernels wheel is cp312/abi3).

Layout (all under the project, not `/opt`):
```
fw2026/vllm-build/
  env/              # fresh conda or venv, python 3.12
  src/vllm          # git checkout @ pinned commit
  src/vllm-xpu-kernels
  wheels/           # built wheels (for reuse, as in the old build)
  logs/  pins.env
```
Use a separate env. Do not touch `frameworks/2026.1.0`.

Open question: the torch XPU wheels bundle their own oneAPI runtime. The Aug investigation found that this bundled runtime hit oneCCL transport failures on Aurora (it needed PALS + `CCL_ATL_TRANSPORT=mpi` plus patches). Building against the module's oneAPI instead may avoid that, but torch must match its ABI. Resolve this in the Phase 1 smoke test before building anything heavy.

## 4. Phases

Run builds on a compute node in `debug-scaling` (no heavy compiles on login nodes). Wheel downloads and `git clone` can happen on the login node. Builds will likely exceed one 59-minute debug-scaling job, so every build step must be resumable: reuse the ninja build dir, ccache, and keep wheels in `wheels/`.

**Phase 0 — prepare (login node, no GPU).**
1. Create the env (python 3.12). Install `torch==2.14.x+xpu` (Track A) from `https://download.pytorch.org/whl/xpu`.
2. Pre-download wheel artifacts into `wheels/` so compute jobs need no network.
3. `git clone` vLLM at the pinned commit; record hashes.

**Phase 1 — smoke test the stack (1 compute node, <10 min).**
1. `import torch; torch.xpu.device_count() == 12`.
2. Run a tiny tensor op on tile 0, and a 2-rank `all_reduce` under PALS to check oneCCL.
3. Check the Aug failure list: Triton `get_native` crash, `is_xe2_arch()` false positive on PVC, free-memory reporting (needs `ZES_ENABLE_SYSMAN=1`).
4. If this fails, stop and decide between the bundled runtime and the module's oneAPI before building.

**Phase 2 — Track A: vLLM v0.33.0rc3 + prebuilt kernels (no kernel compile).**
1. `pip install vllm_xpu_kernels==0.1.15.4`, then build vLLM: `VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e .` (or `pip wheel`).
2. Rust: the vLLM build runs `setuptools-rust` for a `vllm-rs` frontend (`rust/`, toolchain 1.95). `setup.py` has `VLLM_REQUIRE_RUST_FRONTEND` and `VLLM_USE_PRECOMPILED_RUST` switches. Check whether the build works without Rust for XPU; if not, install rustup (`static.rust-lang.org` is reachable) into the project dir.
3. Re-run opt-125m, then gpt-oss MXFP4 (XPU graph on) as regression checks against the current numbers (60.0 single, 11860 on 6×TP2).
4. Test Beam and Muse Spark with the model-class path (see section 6).

**Phase 3 — Track B: patched `vllm-xpu-kernels` (the Nemotron fix).**
1. Check out `vllm-xpu-kernels` `main`; build with the oneAPI 2026.1 module.
2. Patches (one commit, one diff file under `patches/`):
   - `csrc/moe/remap_hidden_states.cpp`: add `TopK == 22` to `DISPATCH_TOPK_LAUNCH`.
   - `csrc/moe/moe_gather.cpp`: add `CASE_TOPK(22, …)`.
   - `csrc/moe/moe_align_sum_kernels.cpp`: add 22 to the `switch (topk)` near line 1368 (it lists 1,2,3,4,6,8,9 only, so the MoE sum path may have the same limit).
   - `csrc/moe/grouped_topk.cpp`: `SG_P2_ENABLE_SINGLE_NEMOTRON 1`.
3. AOT targets: restrict to PVC to cut compile time: `export VLLM_XPU_AOT_DEVICES="pvc"` and `VLLM_XPU_XE2_AOT_DEVICES="pvc"`, and `VLLM_XPU_ENABLE_XE3P=OFF`. The default list also builds `bmg`, `bmg-g21-a0`, `bmg-g31-a0` and `cri`, which are not needed here.
4. Build a wheel (`pip wheel --no-build-isolation`), install into the env, and re-run the Nemotron-3-Ultra INT4 probe (`fw2026/ultra_int4_vllm.pbs`, already written; weights already on disk).
5. A top-k 22 kernel template may be correct but untuned. Compare the result against a CPU or reference implementation on a small input before trusting the throughput.

**Phase 4 — Track B vLLM: `main` + torch 2.15 + patched kernels.** Same steps as Phase 2 with the Track B pins. Then run the full regression and new-model matrix.

## 5. Known risks (from the earlier self-built attempt, 2026-07 to 08)

The earlier from-source stack (`build-vllm-xpu`, vllm 0.27.1 / torch 2.13 / triton 3.7.2) did not run on PVC. Read `gpt-oss-120b/build-vllm-xpu/opus4.8-new_campaign/VLLM_INVESTIGATION.md` and `DEBUG_LOG.md` P11-P18 before starting. Specific traps:
- **Triton `sycl::get_native<level_zero>(device)` SIGSEGV** with triton 3.7.2 on PVC. The module's own triton-xpu 3.7.2 works today (the fw2026 runs use it), so the same version behaves differently in the module's env. The difference is probably in how the module's torch/oneAPI is built. Test triton in Phase 1.
- **`is_xe2_arch()` returns true for PVC** (confirmed again in current `csrc/utils.h`: `is_xe2_arch` includes `intel_gpu_pvc`). This is current upstream behavior, so Xe2 kernels are chosen on PVC; the module's 0.1.11.2 build runs MXFP4 gpt-oss correctly, so this is not automatically fatal. Compare kernel-selection behavior between the module's kernels and the new ones.
- **oneCCL transport init failure** with the bundled runtime (see section 3).
- **Free-memory profiler returns 0** on PVC; needs `ZES_ENABLE_SYSMAN=1` and may need the fallback patch.
- **Moving target:** vLLM main changes daily. Pin commits; keep the wheels so a result can be reproduced.
- **Build time/resources:** the xpu-kernels AOT build with device lists is the long step. Plan on more than one debug-scaling job; use ccache.
- **Queue limit:** one running + one queued `debug-scaling` job per user, shared across sessions. Builds must checkpoint (keep the build dir).
- **Version coupling:** torch / triton / vllm-xpu-kernels must match each other (vLLM `requirements/xpu.txt` pins them). Don't mix Track A and Track B wheels.

## 6. What the build does and does not unlock

| Model | After Track A | After Track B | Remaining blocker |
|---|---|---|---|
| Nemotron-3-Ultra INT4 | no change (top-k 22 still unsupported) | **expected to run** | verify numerics/perf |
| Beam 1.0 BF16 (934 GB local) | config loads (transformers >= 5.22) | same | `models/beam/nvidia` uses `cutedsl` + allreduce ops; the `amd` path is chosen only when `current_platform.is_rocm()`. XPU falls to the NVIDIA branch and will likely fail on CUDA-only ops. Needs an XPU branch in `models/beam/__init__.py` (dispatch is a 3-line `if`) and a replacement for the cutedsl attention ops. |
| Muse Spark Open BF16 (1.5 TB local) | config loads | same | same pattern: `platform = "amd" if is_rocm() else "nvidia"`; NVIDIA code uses flashinfer/cutedsl; `amd` code imports `aiter` (ROCm). Multimodal. |
| Inkling | needs safetensors (only GGUF local) | same | `nvidia/` (cutlass FA4) or `amd/` (aiter) only |
| Laguna, gpt-oss, Nemotron-4 | unchanged | unchanged | none |

So: Track B is expected to fix Nemotron-3-Ultra. Beam, Muse Spark and Inkling become loadable at the config level but their model code has no XPU branch, so they need porting work beyond a build. Treat that as a separate step with its own decision.

## 7. Success criteria

1. Phase 1: torch sees 12 tiles, `all_reduce` works, a Triton kernel runs, no segfault.
2. Phase 2: opt-125m and gpt-oss MXFP4 reproduce the current fw2026 numbers within noise; Beam and Muse config load succeeds (even if model init later fails, record the first error).
3. Phase 3/4: Nemotron-3-Ultra INT4 TP=8 completes a chat generation (quality OK) and has a measured single-stream and concurrency number.
4. Every result goes into `RESULTS_FW2026.md` with pinned commits and wheel hashes.

## 8. Decisions needed

1. Track A first (low risk) or go straight to Track B (needed for Nemotron-3-Ultra)? Recommendation: Phase 0-1 for both, then Track B, since Track A alone doesn't fix the known blocker.
2. Approve installing a Rust toolchain into the project dir if the vLLM build requires it.
3. Approve the kernel patch for top-k 22, and the AOT restriction to PVC.
4. Beam / Muse / Inkling XPU porting: out of scope for the build, or do you want a follow-up plan?
