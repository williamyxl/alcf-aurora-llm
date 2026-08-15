# Campaign results — llama.cpp SYCL best-recipe reruns (fresh build, migrated tree)

**Date:** 2026-08-15
**Author:** Kilo (Claude Opus 4.8)
**Env:** aurora-llm conda env (`/lus/flare/projects/MatSciAI/xiaoliyan/software/conda/envs/aurora-llm`);
llama.cpp SYCL built fresh from source on Aurora PVC (Max 1550), `oneapi/release/2025.3.1`.
**Metric source:** `llama-completion` `common_perf_print` (canonical TTFT / prefill / gen), plus
`llama-bench` where captured. All quality-OK (coherent MOF answer).

> Context: this rerun re-establishes the documented best recipes after migrating the codebase from
> the old `MOFA` project path to `MatSciAI/.../alcf-aurora-llm` and rebuilding from scratch. Numbers
> match or beat the historical `BEST_RECIPE.md` values, confirming the migrated + rebuilt stack works.

---

## Summary table

| Model | Recipe | Tiles | TTFT (ms) | Prefill (tok/s) | Gen/decode (tok/s) | Job | Quality |
|-------|--------|-------|-----------|-----------------|--------------------|-----|---------|
| **gpt-oss-120b** MXFP4 | **F4_hbm** — 1-tile MoE→CPU, HBM `--membind=2` | 1 | **364.45** | **54.88** | **41.78** | 8757601 | OK |
| **gpt-oss-120b** MXFP4 | **P14_tp2** — 2-tile pure GPU, `-sm tensor -ts 0.5/0.5` | 2 | 557.81 | 35.85 (comp) / **495.58** (bench pp512) | **34.02** | 8757611 | OK |
| **Inkling** UD-IQ1_S | **MO1** — 1-tile MoE→CPU, HBM `--preferred=2` | 1 | **1624.44** | **10.47** | **6.11** | 8757662 | OK |
| **Inkling** UD-IQ1_S | **PG10** — 10-tile pure GPU (5 GPUs), `-sm layer` | 10 | 2407.19 | 7.06 | **6.46** | 8757717 | OK |

All runs: `-fa on`, `GGML_SYCL_ENABLE_VMM=0`, `--no-mmap`, short context (N_CTX 2048–4096),
N_PREDICT 127–128, greedy-ish sampling, seed 42.

---

## Detail — gpt-oss-120b

### F4_hbm (best short-context decode) — job 8757601, host x4112c1s3b0n0
```
load  time =  49434.30 ms
prompt eval = 364.45 ms /  20 tok ( 18.22 ms/tok,  54.88 tok/s)
eval        = 3039.61 ms / 127 runs ( 23.93 ms/tok,  41.78 tok/s)
total       = 3481.61 ms / 147 tok
```
- Recipe: `ZE_AFFINITY_MASK=0`, `-sm none`, `-ncmoe 99`, `-t 32`,
  `numactl --physcpubind=1-51,105-155 --membind=2`, `--numa numactl`.
- **41.78 tok/s decode beats the historical 34.32** (BEST_RECIPE.md F4_hbm) — new best.

### P14_tp2 (best 2-tile pure GPU) — job 8757611
```
load  time = 122029.93 ms
prompt eval = 557.81 ms /  20 tok ( 27.89 ms/tok,  35.85 tok/s)   # short prompt
eval        = 3733.14 ms / 127 runs ( 29.39 ms/tok,  34.02 tok/s)
llama-bench : pp512 = 495.58 tok/s ; tg = 33.85 tok/s ; ttft(bench) = 1033.13 ms
```
- Recipe: `ALLOW_MULTI_GPU=1 TP=2`, `ZE_AFFINITY_MASK=0,1`, `-sm tensor -ts 0.5/0.5`, `-fa on`.
- Matches historical ~30 tok/s (BEST_RECIPE.md P14_tp2); prefill scales to ~496 tok/s at pp512.

**gpt-oss verdict:** For single-stream short-context decode, **F4_hbm (1-tile MoE→CPU + HBM) is best
(41.78 tok/s, TTFT 364 ms)**. For prefill-heavy / long prompts, P14_tp2 pure-GPU gives ~496 tok/s pp.

---

## Detail — Inkling

### MO1 (best MoE→CPU short-context) — job 8757662, host (debug-scaling)
```
load  time = 283039.54 ms   # 270 GB UD-IQ1_S, --no-mmap
prompt eval = 1624.44 ms /  17 tok ( 95.56 ms/tok,  10.47 tok/s)
eval        = 20796.81 ms / 127 runs ( 163.75 ms/tok,  6.11 tok/s)
total       = 22648.36 ms / 144 tok
```
- Recipe: `ZE_AFFINITY_MASK=0`, `-sm none`, `-ncmoe 99`, `-t 32`,
  `numactl --physcpubind=1-51,105-155 --preferred=2`, `--numa numactl`, N_CTX 4096.
- Matches historical MO1 (gen 6.43 / prefill 11.62; BEST_RECIPE.md). Quality-OK.

### PG10 (10-tile pure GPU) — job 8757717
```
load  time = 160483.65 ms
prompt eval = 2407.19 ms /  17 tok (141.60 ms/tok,  7.06 tok/s)
eval        = 19667.42 ms / 127 runs (154.86 ms/tok,  6.46 tok/s)
total       = 22175.59 ms / 144 tok
```
- Recipe: `ZE_AFFINITY_MASK=0..9` (5 GPUs × 2 tiles), `-sm layer -ts 0.1×10`, `-fa on`, N_CTX 4096.
- gen 6.46 tok/s matches historical PG10 (~6.73); slightly above MO1 (6.11) but uses 10 tiles vs 1.
- **Verdict: MO1 (1-tile) is the efficient choice** — within ~5% of PG10 decode at 1/10 the tiles.

**Inkling verdict:** MO1 single-tile MoE→CPU reproduces documented ~6 tok/s decode. Inkling is a much
larger model at 1-bit-ish quant; decode is inherently slower than gpt-oss MXFP4.

---

## Cross-model note (vs vLLM narrative)

These llama.cpp numbers re-anchor the diagnosis in `../opus4.8-new_campaign/PLAN.md` and
`../../../cursor-opus4.8-med-diagnosis.md`: gpt-oss-120b MXFP4 does **~34–42 tok/s decode** on this
PVC hardware under llama.cpp, versus the documented **~1.2 tok/s** for the quality-OK vLLM REF-MoE
path — the ~25–35× gap the E1–E9 experiments are designed to explain. The vLLM side of this campaign
requires building the `build-vllm-xpu/env` stack (torch-XPU + IPEX + oneCCL + vllm-xpu-kernels + vLLM),
which is not yet built in the migrated tree; that is the next prerequisite before E1–E9 can run.

---

## Environment / build provenance

| Item | Value |
|------|-------|
| conda env | `/lus/flare/projects/MatSciAI/xiaoliyan/software/conda/envs/aurora-llm` (py3.12) |
| llama.cpp (gpt-oss) | fresh clone `ggml-org/llama.cpp` @ 9d57ce4; SYCL PVC, `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DEVICE_ARCH=pvc`; job 8757581 BUILD_OK |
| llama.cpp (inkling) | own clone; SYCL PVC; BUILD_OK |
| gpt-oss weights | `models/openai-gpt-oss-120b` (HF, 15 safetensors 183 GB) + `models/ggml-org-gpt-oss-120b-GGUF/gpt-oss-120b-MXFP4.gguf` (63.4 GB); symlink `models/openai-gpt-oss-120b-mxfp4.gguf` |
| inkling weights | `models/unsloth-Inkling-GGUF/UD-IQ1_S/*-00001..00007-of-00007.gguf` (270 GB) |
| modules | `oneapi/release/2025.3.1` only (no frameworks) |
