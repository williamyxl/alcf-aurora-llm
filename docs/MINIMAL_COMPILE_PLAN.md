> **Planning document, written before execution.** The minimal-compile plan was carried out: see `docs/ENGINES.md` and `common/vllm/setup_overlay.sh`. Statements about what was 'expected' predate the measurements in the per-model READMEs.

# Minimal-compile plan: model compatibility on Aurora PVC

Written 2026-11-15. Supersedes the "build newest vLLM" idea in `BUILD_VLLM_XPU_PLAN.md`.
Account `MatSciAI`, queue `debug-scaling` only. Downloads approved earlier cover only the models in the earlier table.

## 0. Result: the newest vLLM is not needed

Two findings change the plan.

1. **The installed vLLM (0.26.1) contains no compiled code of its own for XPU** apart from two tiny files (`fs_io_C`, `spinloop`). Everything compiled sits in `vllm_xpu_kernels` (`_moe_C`, `_xpu_C`, attention libs). So "compile vLLM" is the wrong target.
2. **Only one thing truly needs compiling: `_moe_C.abi3.so` (21 MB) from `vllm-xpu-kernels` tag `v0.1.11.1`** (commit `a692986`, the exact base of the installed `0.1.11.2.dev0+ga692986`). The other models need Python-only changes.

| Compatibility gap | Fix | Compile? |
|---|---|---|
| Muse Spark Open: unknown `model_type` | `transformers` 5.23.0 into a user overlay (11 MB pip `--target`). vLLM then resolves it through its **Transformers backend** (`TransformersMultiModalMoEForCausalLM`). Verified. | **No** |
| Beam 1.0: unknown `model_type` | same overlay, plus a 25-line Python shim for two `None` rope entries (see section 3). Resolves to `TransformersMoEForCausalLM`. Verified. | **No** |
| Nemotron-3-Ultra: `Unsupported TopK value` (top-k 22) | patch top-k tables in 3 C++ files, rebuild only `_moe_C` | **Yes, one extension** |
| Inkling | model code is NVIDIA/ROCm only; local weights are GGUF only | not fixable by building |

## 1. What was verified in this session (nothing here needs a GPU)

- Overlay: `fw2026/overlay/py` holds `transformers 5.23.0`, `tokenizers 0.23.3`, `huggingface_hub 2.4.0` (installed with `--no-deps`; the module already has every other dependency). Use with `PYTHONPATH=fw2026/overlay/py`. Module `transformers` is 5.14.1 and read-only; the overlay shadows it without touching `/opt/aurora`.
- vLLM config resolution with the overlay:
  - Muse: `OK arch=MuseSparkForConditionalGeneration -> TransformersMultiModalMoEForCausalLM`
  - Beam without the shim: `TypeError: argument of type 'NoneType' is not iterable` in `vllm/transformers_utils/config.py:484` (`patch_legacy_rope_type`)
  - Beam with `fw2026/overlay/shim/sitecustomize.py`: `OK arch=BeamForCausalLM -> TransformersMoEForCausalLM`
- Why the shim is needed: Beam's `config.json` has `rope_parameters = {sliding_attention: {...}, full_attention: null}` (the full-attention layers use no RoPE). vLLM 0.26.1 crashes on the `null` entry in two places (`patch_legacy_rope_type`, then `_get_and_verify_max_len`). vLLM 0.30.0 and later already contain the `is not None` guard for the first one; the shim reproduces that fix instead of upgrading vLLM.
- The three top-k files compile cleanly against the module's own torch 2.13 headers and icpx 2026.1.0 (syntax-only, `-fsycl`). So the `_moe_C` rebuild toolchain works here.

## 2. The one compile: patched `_moe_C`

**Source:** `vllm-xpu-kernels` tag `v0.1.11.1` (commit `a692986`, 2026-07-17). It matches the installed kernels, so the rest of the installed package (`_xpu_C`, `_C`, attention and grouped-gemm libs) stays untouched and ABI-compatible. Rebuilding from `main` would bring torch 2.15 requirements and is unnecessary.

**Patches** (all small table edits; the templates are generic in `TopK`):

| File (at v0.1.11.1) | Current top-k accepted | Edit |
|---|---|---|
| `csrc/moe/remap_hidden_states.cpp` (`DISPATCH_TOPK_LAUNCH`, ~line 519) | 1,2,4,6,7,8,10; else throws `Unsupported TopK value` | add `TopK == 22` branch. This is the exact error seen with Nemotron-3-Ultra. |
| `csrc/moe/moe_gather.cpp` (`CASE_TOPK`, ~line 134) | 1,2,4,6,7,8,10; else `not support TOPK=` | add `CASE_TOPK(22, ElemsPerItem)` |
| `csrc/moe/moe_align_sum_kernels.cpp` (`switch (topk)`, ~line 1325) | 1,2,3,4,6,8,9; default branch falls back to `at::sum_out` | no edit needed. Top-k 22 takes the safe default (slower but correct). Optionally add `case 22`. |
| `csrc/moe/grouped_topk.cpp` | `TORCH_CHECK(topk <= 32)` | no edit needed at this tag |

Check before building: the router for Nemotron-3-Ultra (512 experts, top-22) goes through `topk_softmax` or `grouped_topk`. Both appear to allow 22 at this tag, but confirm by running the probe, since the remap/gather kernels are the ones that threw.

**Build recipe** (CMake, only the `_moe_C` target; the rest of the repo is not built):
- Toolchain: `module load oneapi/release/2026.1.0` (icpx 2026.1.0.20260617, the same compiler that built the shipped `_moe_C`: its strings show "Intel(R) oneAPI DPC++/C++ Compiler 2026.1.0"). gcc 14.3.0 (the shipped lib was built with Spack GCC 14.3.0).
- Python/torch: the module's own env (`aurora_frameworks-2026.1.0`, torch 2.13, cxx11abi=1), so no new torch is needed and no ABI mismatch.
- Source list: `file(GLOB csrc/moe/*.cpp)` (11 files), `define_gpu_extension_target(_moe_C ... USE_SABI 3)`.
- Link needs: `libtorch*`, `libc10_xpu`, `libsycl.so.9`, `libze_loader`, all present. The shipped `_moe_C` links exactly these.
- AOT: the shipped lib embeds `spir64_gen` (AOT) plus `spir64` (SPIR-V fallback). Set `VLLM_XPU_AOT_DEVICES="pvc"` to compile only for PVC, which also cuts build time. PVC ocloc is at `/usr/bin/ocloc`.
- To avoid pulling oneDNN and SYCL-TLA (the full CMake does `find_package(oneDNN REQUIRED)` and fetches sycl-tla, which are not needed for `_moe_C`): compile `_moe_C` with a minimal standalone CMake/ninja file or a `torch.utils.cpp_extension.SyclExtension` build over `csrc/moe/*.cpp` with the same flags. This is the part to prototype first. If the full CMake is unavoidable, set `BUILD_SYCL_TLA_KERNELS=OFF` and the other `*_KERNELS_ENABLED=OFF` options (they exist at this tag) so only `MOE_KERNELS_ENABLED` stays on, and provide oneDNN or stub `find_package(oneDNN)`.
- Install: copy the new `_moe_C.abi3.so` into a user overlay directory that shadows `vllm_xpu_kernels/_moe_C.abi3.so` (for example `fw2026/overlay/kernels/vllm_xpu_kernels/`, placed ahead on `PYTHONPATH` with a copy of the package's Python files), keeping the module untouched. Alternatively `LD_PRELOAD`-free: copy the whole `vllm_xpu_kernels` package (about 800 MB, mostly `libattn_kernels_xe_2.so` at 692 MB) into the overlay and replace one file.

Expected cost: 11 small C++ files and one extension, versus the many-hour full vLLM plus kernel library builds in the earlier plan. A first attempt should fit one `debug-scaling` job. Not yet measured.

## 3. Python-only changes (no compile)

`fw2026/overlay/` layout:
```
overlay/py/            transformers 5.23.0, tokenizers 0.23.3, huggingface_hub 2.4.0
overlay/shim/sitecustomize.py   Beam rope-None shim (auto-loaded by every Python process via PYTHONPATH)
```
Launch environment for Beam / Muse runs: `PYTHONPATH=$PWD/fw2026/overlay/shim:$PWD/fw2026/overlay/py:$PYTHONPATH`, then the usual `module load frameworks/2026.1.0`. Check that spawned vLLM worker processes inherit it; `sitecustomize` is imported in each interpreter start, so they should.

Why a lower vLLM version doesn't help here: moving to a different vLLM would mean new torch / triton / kernels pins (0.28-0.30 want torch 2.13 and kernels 0.1.13.2 to 0.1.14.1, 0.31+ want torch 2.14), so more compiling, not less. The installed 0.26.1 plus one rebuilt extension is the smallest delta.

## 4. Can the models actually run after these fixes? (remaining risks, not yet tested)

| Model | Config ok | Likely next blocker |
|---|---|---|
| Nemotron-3-Ultra INT4 w4a16 | already past config | patched `_moe_C` needed; then verify numerics and speed. Nemotron-H and relu2 MoE already build. |
| Muse Spark (Transformers backend) | yes | **Memory.** BF16 is 1.5 TB and does not fit one node (768 GB HBM). The NVFP4 (533 GB), `amd` Quark-MXFP4 (509 GB) and a FP8 option would fit, but PVC has no native FP4 and XPU FP8 MoE is slow. This is a download decision. The Transformers backend also needs the MoE and multimodal paths to work on XPU, untested. |
| Beam (Transformers backend) | yes (with shim) | **Memory.** BF16 is 934 GB locally; does not fit one node. FP8 (503 GB) would fit but is likely slow on PVC (Laguna FP8 was 5x slower). NVFP4 (294 GB) has no hardware path. Two nodes for BF16 would need multi-node vLLM (not tested here). |
| Inkling | no (GGUF only) | needs safetensors plus an XPU port of attention/short-conv; out of scope |

Honest summary: the compatibility (config / model class) fixes are cheap and already verified up to config resolution. Actually running Beam and Muse is limited by node memory, not by compilation. Testing them on the local BF16 weights needs at least 2 nodes (Beam) or about 3 nodes (Muse), or a quantized checkpoint.

## 5. Steps

**Step 1 (login node, ~10 min, no GPU):** prototype the standalone `_moe_C` build of `v0.1.11.1` with the 2-line patch (syntax already compiles). Confirm it links against the module's torch.
**Step 2 (`debug-scaling`, short job):** full compile of `_moe_C` with `VLLM_XPU_AOT_DEVICES=pvc`. Save the `.so` and the diff under `fw2026/overlay/kernels/patches/`.
**Step 3 (`debug-scaling`):** re-run the Nemotron-3-Ultra INT4 probe (`fw2026/ultra_int4_vllm.pbs`, weights on disk) with the overlay kernel. Also run a gpt-oss and Laguna regression to confirm the replaced `_moe_C` did not change their numbers (top-k 4 and 10 take the same code paths).
**Step 4 (`debug-scaling`, optional):** Beam and Muse smoke tests on 1 node. They will almost certainly fail on memory with the BF16 weights; the useful output is the first error beyond config (model class build, MoE on XPU). A reduced-layer config with `load_format=dummy` is a cheap way to test the architecture on one node without the full weights.
**Step 5:** record everything in `RESULTS_FW2026.md`.

## 6. Decisions needed

1. Go ahead with Steps 1-3 (the Nemotron-3-Ultra fix)? It is the only real compile.
2. For Beam and Muse, do you want (a) a dummy-weight architecture test on one node (no download), (b) a quantized checkpoint download (Beam FP8 503 GB; Muse MXFP4 509 GB, both "fit but slow or unsupported on PVC hardware"), or (c) leave them at config-level support?
3. Is multi-node vLLM for the local BF16 Beam (2 nodes) in scope? It is untested on this machine and the debug-scaling queue limit applies.
