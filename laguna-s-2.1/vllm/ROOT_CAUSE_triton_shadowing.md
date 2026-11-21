# Laguna S 2.1 vLLM failure: root cause (2026-11-20)

## Root cause
`module load frameworks/2026.1.0` sets `PYTHONUSERBASE=/home/xiaoliyan/.local/aurora/frameworks/2026.1.0`.
That user site holds a pip-installed `triton 3.8.0` (installed 2026-11-18, for sglang work). It has only
`amd/` and `nvidia/` backends, with no `intel/`. It shadows the framework's `triton_xpu 3.7.2`, which does have `intel/`.

Python's user site comes before the framework site-packages, so every process imports the wrong triton:
- `triton.backends` finds 0 active drivers.
- vLLM prints `Triton is installed but 0 active driver(s) found`.
- Inductor/XPU compile in `profile_run` fails with `0 active drivers ([]). There should only be one.`
- Other runs surfaced it as `Cannot find a working triton installation` or `No module named 'triton.backends.intel'`.

Evidence:
- Worker traceback (`TASK2/logs/serve_9106439_laguna_eager_r0.out`) points at `~/.local/aurora/.../triton/runtime/driver.py`.
- Job 9109873, probe A (user site on): `ModuleNotFoundError: No module named 'triton.backends.intel'`.
- Job 9109873, probe B (`PYTHONNOUSERSITE=1`): triton 3.7.2, backends amd/intel/nvidia, active driver `intel`, `HAS_TRITON=True`.

## Fix
Set `PYTHONNOUSERSITE=1` and unset `PYTHONUSERBASE`. Do not rely on `PYTHONPATH` tricks. Verified in job 9109893 with `serve_conc.py`:
`ok=true`, quality OK, load 746 s, 14.3 tok/s single stream, 1350 tok/s at 128-way.
This matches the eager reference in `serve_conc_eager.pbs`.
The pristine framework `vllm` was used, with no `dummy_torchcodec`, no triton env vars and no overlay.
(`import torchcodec` only prints a harmless load-traceback warning; vLLM runs fine.)
Repro: `qsub laguna-s-2.1/vllm/standalone_eager_check.pbs` or `qsub -v REPO=$PWD laguna-s-2.1/vllm/serve_conc_eager.pbs`.

## Why earlier attempts failed or misled
1. `common/env.sh` `vllm_env` never sets `PYTHONNOUSERSITE`. `TRITON_BACKENDS_IN_TREE`, `TRITON_INTEL_GPU_ARCH` and `VLLM_DISABLE_TRITON` are unrelated and do nothing.
2. `PYTHONPATH="...:${FW}/lib/python3.12/site-packages"` uses the relative path `2026/lib/...`, which is nonexistent. It is a no-op.
3. `dummy_torchcodec/` makes things worse:
   - `vllm/__init__.py` uses `extend_path`, so the repo `vllm/` shadows the real package. That gives `No module named 'vllm.entrypoints'` and `vllm has no attribute __version__`.
   - `vllm/triton_utils/importing.py` is a broken copy: `find_spec` and `version` are not imported. It does not contain the fix RUN_LAGUNA.md describes.
   - `sitecustomize.py` (the `.bak`) raises `NameError: importing_module` (jobs 9106899/900/920).
   - RUN_LAGUNA.md's claimed bug in the stock `importing.py` is not real. The stock file has no `root`/`backend_name`.
4. TASK2 `serve_local_llm.sh` (rovi-anl-tasks, read only):
   - Last runs (9107772 to 9108001) died with `libpti_view.so.0: cannot open shared object file`, because `vllm_env` was skipped and the framework lib paths were never set up.
   - `$EAGER_FLAG` is never defined, so the `laguna_eager` recipe ran with `enforce_eager=False` (see `serve_9106439` log). That is inductor/graph mode, not eager.
   - The `LAGUNA_DEBUG_REPORT` conclusion that Laguna/MoE is incompatible with Triton on XPU is wrong. It is an environment shadowing problem, and gpt-oss only worked because it avoids the inductor path.
5. The first run in `laguna-eager.o9106464` hit the walltime. Load takes about 12.5 min, so the 59-minute walltime is fine, but the bench plus first-run compile time is not trivial.

## Recommended changes (not applied outside this repo)
- `common/env.sh` `vllm_env`: add `export PYTHONNOUSERSITE=1; unset PYTHONUSERBASE`, and remove the bogus `PYTHONPATH` line and the `dummy_torchcodec` dependency.
- Delete or rename `dummy_torchcodec/vllm` and `sitecustomize*`.
- Optionally `pip uninstall` the user-site `triton 3.8.0` and `torchcodec 0.17.0` from `~/.local/aurora/frameworks/2026.1.0`, or move that dir aside.
- TASK2 `serve_local_llm.sh` needs the same environment, including `vllm_env`, and must define `EAGER_FLAG=--enforce-eager`. TASK2 is read-only for me.
