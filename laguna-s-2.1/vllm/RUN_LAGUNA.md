# Laguna S 2.1: vLLM on Aurora, working eager recipe

Verified 2026-11-20/21 (jobs 9109893 standalone and 9110540 via serve_conc_eager.pbs, frameworks/2026.1.0, vLLM 0.26.1, triton_xpu 3.7.2): BF16, TP=8, eager.
`ok=true`, quality OK, load 746 s, 14.3 tok/s single stream, 1350 tok/s at 128-way. These match the eager reference in the README.

## Run
```bash
cd /lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm
qsub -v REPO=$PWD laguna-s-2.1/vllm/serve_conc_eager.pbs                       # debug queue
qsub -q debug-scaling -v REPO=$PWD laguna-s-2.1/vllm/serve_conc_eager.pbs      # or debug-scaling
```
Output goes to `logs/laguna_eager.out`. Look for `SERVE_JSON=... "ok": true`, `"quality_ok": true`, and `rc=0` in the PBS output.
Model load takes about 12.5 minutes, so the 59-minute debug walltime is enough.

Equivalent standalone commands (no repo scripts):
```bash
module use /soft/modulefiles && module load frameworks/2026.1.0
export PYTHONNOUSERSITE=1; unset PYTHONUSERBASE; export PYTHONPATH=
export HF_HUB_OFFLINE=1 TMPDIR=/tmp ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7
$FWROOT/bin/python common/vllm/serve_conc.py --model data/models/laguna-s-2.1/bf16-hf \
  --tp 8 --max-model-len 8192 --mns 128 --kv-gib 12 --eager
```
`standalone_eager_check.pbs` is a self-contained copy of this. Root-cause write-up: `ROOT_CAUSE_triton_shadowing.md`.

## The one thing that matters: ignore the user site-packages
`module load frameworks/2026.1.0` sets `PYTHONUSERBASE=~/.local/aurora/frameworks/2026.1.0`.
If that directory has a pip-installed `triton` (for example 3.8.0, installed for sglang), it shadows the framework's `triton_xpu 3.7.2`:
- The pip triton has only `amd/` and `nvidia/` backends, so there is no `intel/` backend and Triton sees 0 active drivers.
- vLLM prints `Triton is installed but 0 active driver(s) found`.
- Engine init then fails in `profile_run` with `0 active drivers ([]). There should only be one.`, or with `Cannot find a working triton installation` or `No module named 'triton.backends.intel'`.

`common/env.sh` `vllm_env` now sets `PYTHONNOUSERSITE=1` and unsets `PYTHONUSERBASE`, so every recipe that sources it is protected.
Any new launcher must do the same, including ones that start `vllm serve` directly.
The same fix likely applies to the other models' recipes that failed with triton errors.

## Things that do NOT help (and can break it)
- `PYTHONPATH` games and `dummy_torchcodec/`. Its `vllm/__init__.py` shadows the real `vllm` package (`No module named 'vllm.entrypoints'`), and its `importing.py` copy is broken. It is not needed, since `import torchcodec` only prints a harmless warning.
- `sitecustomize.py` import hooks (NameError at import).
- `TRITON_BACKENDS_IN_TREE`, `TRITON_INTEL_GPU_ARCH`, `VLLM_DISABLE_TRITON`: no effect on this failure.
- Skipping `vllm_env` / the `module load`: gives `libpti_view.so.0: cannot open shared object file`.
- Launching without `--enforce-eager` while not using the XPU-graph recipe: this runs the inductor/compile path, not eager.

## Diagnose a failing run
```bash
module load frameworks/2026.1.0
python -c "import triton; print(triton.__version__, triton.__file__)"                        # may show the user-site 3.8.0
PYTHONNOUSERSITE=1 python -c "import triton; print(triton.__version__, triton.__file__)"     # must be 3.7.2 under /opt/aurora/...
```
Run these on a compute node (`qsub -I -q debug`); on a login node they can hang.

## Not verified
- `vllm serve` (OpenAI API) via `serve_api.pbs`. This recipe uses the offline engine (same scheduler). The HTTP server has not been tested after the fix.
- XPU-graph mode (`serve_conc_graph.pbs`, `serve_api.pbs`). The README reports 36.6 tok/s with it, but it has not been re-run after the env change. TASK2 saw `RuntimeError: cancelled` on long prompts in graph mode.
- FP8 (`fp8_not_recommended.pbs`).
