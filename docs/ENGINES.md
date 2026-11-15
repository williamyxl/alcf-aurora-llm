# Engines, versions and add-ons

Every recipe runs on one of three engine setups. All are reproducible from this repo; none needs root or a write to `/opt/aurora`.

## 1. vLLM on the Aurora frameworks module (default)

| | `FW=2026` (default) | `FW=2025` (legacy) |
|---|---|---|
| module | `frameworks/2026.1.0` | `frameworks` from `/opt/aurora/26.26.0/frameworks/modulefiles` (2025.3.1) |
| vLLM / torch / triton | 0.26.1 / 2.13 / triton-xpu 3.7.2 | 0.15 / 2.10 / 3.6 |
| vllm-xpu-kernels | 0.1.11.2 (base `a692986`, tag `v0.1.11.1`) | bundled |
| transformers | 5.14.1 | older |
| used for | everything except the legacy gpt-oss API recipe | `gpt-oss-120b/vllm/serve_api_legacy_fw2025.pbs` |

Selected by `FW=` in `common/env.sh`; recipes call `vllm_env`.

### Two settings matter more than anything else
1. **XPU graph.** Set `XPUGRAPH=1` and do not pass `--enforce-eager`. It is *off* in a default run. It took gpt-oss-120b single-stream from 26.3 to 60.0 tok/s and Laguna S 2.1 from 13.6 to 36.6 tok/s (concurrency 1352 to 2342 tok/s). Capture warmup scales with layer count and is too slow for the 96-layer Nemotron-4-340B: the 30-minute distributed-init timeout fires. Use eager for the very large models.
2. **Data-parallel layout.** For a model that fits on 2 tiles, six TP=2 engines beat three TP=4 engines (gpt-oss MXFP4: 11860 vs 6849 tok/s).

### Add-ons on top of the read-only module (`common/vllm/setup_overlay.sh`)
These live in `data/overlay/` and are switched on per run, so the module itself is never modified.

| Switch | What it adds | Needed for |
|---|---|---|
| `OVERLAY_KERNELS=1` | a shadow of `vllm_xpu_kernels` whose `_moe_C` was rebuilt with top-k = 22 (`common/vllm/kernels/topk22.patch`) | Nemotron-3-Ultra (512 experts, top-22). Stock kernel stops with `Unsupported TopK value`. |
| `OVERLAY_TF=1` | transformers 5.23.0 + tokenizers 0.23.3 + huggingface_hub 2.4.0, and `shim_sitecustomize.py` (tolerates `None` RoPE entries) | Beam / Muse Spark: lets vLLM recognise `model_type` `beam` / `muse_spark`. This gets the config and model class to resolve; it does **not** make them serve (see their READMEs). |

The kernel patch is two table edits (`csrc/moe/remap_hidden_states.cpp`, `csrc/moe/moe_gather.cpp`). The rebuild compiles only the 11 files of `_moe_C` against the module's own torch 2.13 and `icpx` 2026.1, PVC-only, in about ten minutes on one node. Only `_moe_C` is replaced, so the rest of the package stays ABI-matched.

## 2. llama.cpp SYCL (two prebuilt trees)

Built with `common/llamacpp/build.sh` (oneAPI 2025.3.1, `-DGGML_SYCL=ON -DGGML_SYCL_DEVICE_ARCH=pvc`).

| `LLAMA=` | llama.cpp | architectures |
|---|---|---|
| `gptoss` | master `9d57ce4` (2026-08-15) | gpt-oss |
| `inkling` | PR 25731 at `368b24c35` (2026-08-12) | inkling, nemotron-h (Ultra), nemotron (4-340B), laguna |

The binaries keep the RUNPATH of the directory they were built in, so `common/env.sh` sets `LD_LIBRARY_PATH` to the build's `bin/`. Rebuild with `common/llamacpp/build.sh` if you move the tree. The `inkling` tree is also the engine for Nemotron.

The llama.cpp numbers in this repo were measured on these unchanged builds. They do not exercise `frameworks/2026.1.0`.

## 3. Not used any more
The earlier self-built Torch 2.10 + IPEX + vLLM stack (July 2026) never ran vLLM correctly on PVC; its history is in `gpt-oss-120b/vllm/WHY_SELFBUILT_FAILED.md`. The module-based setup above replaced it. The LoRA scripts in `*/training/` depended on that stack and are kept as reference only.

## Hardware facts that shape every recipe
- One Aurora node: 6 GPUs x 2 tiles = 12 tiles, 64 GB HBM each (768 GB), 2 sockets. Set `ZE_FLAT_DEVICE_HIERARCHY=FLAT` and pin tiles with `ZE_AFFINITY_MASK`.
- PVC has no native FP4, and the XPU FP8 MoE path is both slow and (for Nemotron-4) wrong. Working precisions here: MXFP4 (gpt-oss only), BF16, INT4 weight-only.
- Compute nodes need `proxy.alcf.anl.gov:3128` for any download; do downloads on a login node.
- Queues used: `debug` and `debug-scaling` (1 h cap, one running plus one queued job per user per queue), `capacity` for long serving jobs. Always `-l filesystems=flare`, account `MatSciAI` (change in `common/env.local.sh`).
