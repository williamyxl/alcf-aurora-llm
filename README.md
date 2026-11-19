# LLM serving recipes for ALCF Aurora (Intel Max 1550 / PVC)

Reproducible recipes for running large language models on one Aurora node (12 GPU tiles, 768 GB HBM) with **vLLM** and **llama.cpp (SYCL)**.
One folder per model; each has `llamacpp/` and `vllm/` recipes, a README with measured numbers, and a `download` script.

## Results at a glance

Measured 2026-07 to 2026-11 on the `debug` / `debug-scaling` queues. "Conc" = aggregate generation tok/s under concurrent load. **Single-stream and conc come from different runs and settings; read each model README before comparing.**

| Model | Best vLLM recipe | vLLM result | Best llama.cpp recipe | llama.cpp result |
|---|---|---|---|---|
| [gpt-oss-120b](gpt-oss-120b/) MXFP4 | 6 x TP2 data parallel; 1 x TP4 + XPU graph | **11860 tok/s conc** (92.7 req/s); **60.0 tok/s** single | 1 tile, MoE on CPU | **42.4 tok/s** single |
| [Laguna S 2.1](laguna-s-2.1/) BF16 | TP8 + XPU graph | **2342 tok/s conc** (18.3 req/s); **36.6 tok/s** single | not tried | |
| [Nemotron-3-Ultra](nemotron-3-ultra/) 550B | INT4, TP8, patched top-k=22 kernel | 115 tok/s at 16-way; 7.8 single | 3 x 4 tiles full-XPU | 15.7 tok/s at 8-way; 7.1 single (1 tile) |
| [Nemotron 3.5 Super VL](nemotron-3-5-super-vl/) 8B | INT4, TP8, patched top-k=22 kernel | 7.8 tok/s single, 115 tok/s at 16-way, quality OK | 3 x 4 tiles full-XPU | 12.9 / 13.9 / 15.7 tok/s at c=1/4/8; 7.1 single (1 tile) |
| [Nemotron-4-340B](nemotron-4-340b/) dense | BF16 = 1 TB, **does not fit one node** (768 GB HBM); FP8 runs but **output is garbage** | not usable | 2 x 6 tiles | 6.6 tok/s at 8-way; 2.4 single |
| [Inkling](inkling/) ~975B | not supported (NVIDIA-only model code, GGUF-only weights) | n/a | 12 tiles pure GPU | 17.1 tok/s at 128 seq; 6.6 single |
| [Beam 1.0](beam-1.0/) 501B | **blocked** (model build, memory) | n/a | no arch | |
| [Muse Spark Open](muse-spark-open/) 816B | **blocked** (memory; 1.5 TB BF16) | n/a | no arch | |

What produced the gains (details in [`docs/ENGINES.md`](docs/ENGINES.md)):
- **XPU graph** (`XPUGRAPH=1`, no `--enforce-eager`): 2.3x single-stream on gpt-oss, 2.7x on Laguna. It was off in every earlier benchmark.
- **More, smaller engines**: 6 x TP2 beats 3 x TP4 by 73% on gpt-oss.
- **A rebuilt `_moe_C` kernel** (top-k 22) is what lets vLLM run Nemotron-3-Ultra.
- **Avoid FP8 on PVC**: slower (Laguna, ~5x) or wrong (Nemotron-4).

## August vs November 2026

Same model paired where both periods measured it. Single-stream = one request at a time; high concurrency = aggregate generation tok/s under load. Layouts differ for some concurrency pairs (see notes). N/A = not measured with that metric. Chart: `docs/figures/then_vs_now.png`. Table and chart come from the same data in `common/tools/plot_then_vs_now.py` (regenerate both with `python3 common/tools/plot_then_vs_now.py --readme`).

<!-- then-vs-now:start -->
| Model — engine, layout | Prefill tok/s (Aug / Nov) | Generation tok/s (Aug / Nov) | TTFT ms (Aug / Nov) | High-concurrency gen tok/s (Aug / Nov) |
|---|---:|---:|---:|---:|
| gpt-oss-120b MXFP4 — llama.cpp, 1 tile MoE->CPU | 41.4 / 48.6 | 41.6 / 42.4 | 411 / N/A | 417 / 194 |
| gpt-oss-120b MXFP4 — vLLM, 1 x TP4 eager | N/A / 1,977 | N/A / 26.3 | N/A / 44 | N/A / N/A |
| gpt-oss-120b MXFP4 — vLLM, 1 x TP4 XPU graph | N/A / 2,887 | N/A / 60.0 | N/A / 30 | N/A / N/A |
| gpt-oss-120b MXFP4 — vLLM, node-level data parallel | N/A / N/A | N/A / N/A | N/A / N/A | 4,565 / 11,860 |
| Laguna S 2.1 BF16 — vLLM, TP8 XPU graph | N/A / 1,069 | N/A / 36.6 | N/A / 58 | N/A / 2,342 |
| Nemotron-3-Ultra UD-IQ2_M — llama.cpp, 1 tile MoE->CPU | 11.9 / N/A | 7.1 / 7.8 | 2,670 / N/A | N/A / N/A |
| Nemotron-3-Ultra UD-IQ2_M — llama.cpp, 3 x 4 tiles DP | N/A / N/A | N/A / N/A | N/A / N/A | 15.6 / 15.7 |
| Nemotron-3-Ultra INT4 — vLLM, TP8 eager | N/A / 194 | N/A / 7.8 | N/A / 185 | N/A / 115 |
| Nemotron-4-340B i1-Q4_K_M — llama.cpp, 6 tiles | 11.0 / N/A | 2.3 / 2.3 | 2,390 / 2,375 | 9.9 / 6.6 |
| Inkling UD-IQ1_S — llama.cpp, 12 tiles | 8.7 / 6.2 | 6.7 / 6.6 | 2,300 / N/A | N/A / 17.1 |
| Nemotron-4-340B FP8-Dynamic — vLLM, TP8 eager | N/A / N/A | N/A / N/A | N/A / N/A | N/A / N/A |
<!-- then-vs-now:end -->

Per-row sources are in the `DATA` list of that script. Notes:
- **Same layout in both periods (a true pair):** Nemotron-3-Ultra llama.cpp 3×4-tile data-parallel, concurrency 15.6 → 15.7.
- **Different layout or engine:** gpt-oss-120b node data-parallel (Aug 3×TP4 → Nov 6×TP2); gpt-oss llama.cpp (2 tiles both, different batch sizes); Nemotron-4-340B llama.cpp (3×4 → 2×6 tiles).
- **Not reported:** Nemotron-4-340B FP8 on vLLM. The run completed, but its output is invalid, so every metric is N/A.

## Quick start
```bash
git clone https://github.com/williamyxl/alcf-aurora-llm.git && cd alcf-aurora-llm
cp common/env.local.sh.example common/env.local.sh     # set PBS_ACCOUNT and DATA (where weights/builds live); see below
bash gpt-oss-120b/download.sh                          # login node: fetches weights into $DATA/store, links $DATA/models/...
qsub -v REPO=$PWD common/vllm/smoke_opt125m.pbs        # ~10 min stack check (needs the opt-125m weights, see below)
qsub -v REPO=$PWD gpt-oss-120b/vllm/dp_6xtp2.pbs       # the headline recipe
```
Every recipe is a small PBS script that sources `common/env.sh` and calls a shared launcher. Submit from the repo root with `-v REPO=$PWD`. The `#PBS` headers name account `MatSciAI`; use a different one with `qsub -A <account>` (command-line options override the header), and change queue with `qsub -q <queue>`. Results go to `logs/` (gitignored). vLLM tests print a `SERVE_JSON=` / `PERF_JSON=` / `CONC_JSON=` line.

## Layout
```
common/
  env.sh                site config: account, data dirs, module setup, engine switches (edit env.local.sh, not this)
  lib.sh                bash helpers
  vllm/                 run_bench.sh (single+concurrency), run_dp.sh (data parallel), serve_api.sh (OpenAI endpoint + LB),
                        serve_conc.py, bench_single.py, bench_conc.py, setup_overlay.sh (transformers 5.23 + top-k22 kernel), kernels/topk22.patch
  llamacpp/             build.sh (SYCL build), oneshot.sh (single-stream / batched-bench), serve_bench.sh (llama-server + load test)
  serve/                lb.py (round-robin LB), bench_client.py, client_example.py, uan_tunnel.sh, download_tiktoken_cache.sh
  tools/                hf_download.sh (resumable), plot_gptoss_serving.py
<model>/                gpt-oss-120b  laguna-s-2.1  nemotron-3-ultra  nemotron-4-340b  inkling  beam-1.0  muse-spark-open
  README.md             what runs, measured numbers, caveats
  download*.sh          checkpoint download (login node)
  llamacpp/*.pbs        llama.cpp recipes + MEASUREMENTS.md (imported campaign notes)
  vllm/*.pbs            vLLM recipes
  training/             LoRA smoke tests from July 2026 (reference only; env no longer exists)
docs/                   ENGINES.md (versions, switches, hardware), plans, RESULTS_RAW.md (raw session log)
data/   (gitignored)    store/ weights  models/<model>/<variant> links  llamacpp/ builds  overlay/  -- relocate with DATA=
logs/   (gitignored)
```

## Data and engines (not in git)
Weights (about 4.5 TB here), the two llama.cpp builds and the vLLM overlays live under `$DATA` (default `./data`; point it at project storage with `DATA=` in `common/env.local.sh`). Recipes only ever refer to `$MODELS/<model>/<variant>`, so relocating data means re-linking, not editing scripts.

Rebuild what is not in git:
- **vLLM**: nothing to build; it comes from the Aurora module. For the add-ons: `common/vllm/setup_overlay.sh py` (login node) and `qsub -v REPO=$PWD common/vllm/setup_overlay.pbs` (compiles the kernel on a compute node).
- **llama.cpp**: `qsub -v REPO=$PWD,NAME=gptoss,REF=9d57ce4 common/llamacpp/build.pbs` and `NAME=inkling,PR=25731,REF=368b24c35` (see `docs/ENGINES.md`).
- **Weights**: each model's `download*.sh`. Nothing downloads on its own.
- `opt-125m` (the smoke test model) is a plain Hugging Face download: `common/tools/hf_download.sh facebook/opt-125m $DATA/store/opt-125m`, then `ln -s ../../store/opt-125m $DATA/models/opt-125m/fp16-hf` after `mkdir -p $DATA/models/opt-125m`.

## Known limits
- Only gpt-oss-120b reaches thousands of tok/s. The others are single-node, memory-bound or CPU-bound; numbers are in each README.
- llama.cpp results come from builds made before `frameworks/2026.1.0`; they were rerun unchanged and match the earlier campaign.
- Each configuration was measured once (no error bars).
- Beam, Muse Spark and Inkling cannot be served by vLLM here; upstream vLLM/llama.cpp `main` have newer support not present in the installed module.
- Nemotron-4-340B BF16 (~1 TB) does not fit one Aurora node's 768 GB HBM; downloading it would not enable vLLM without multi-node, which is not set up. Only FP8 (garbage output) and GGUF (llama.cpp only) were tested locally.

Remote: https://github.com/williamyxl/alcf-aurora-llm
