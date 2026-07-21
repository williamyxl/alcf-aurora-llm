# SUCCESS_INFER — Phase 5 PASS

**Date:** 2026-07-18T00:58:28+00:00  
**Job:** `8680184.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`  
**Host:** `x4303c1s3b0n0`  
**Log:** `build-vllm-xpu/logs/test_gptoss.out` (run starting `host=x4303c1s3b0n0`)

## Verdict

gpt-oss-120b inference on the self-built Torch-XPU + IPEX + oneCCL + vLLM stack produced a **coherent** MOF isotherm reply (not all-`!` / token-id-0 garbage) plus parseable `METRICS_JSON`.

## Runtime recipe (required)

| Setting | Value |
|---------|--------|
| Modules | `oneapi/release/2025.3.1` only (no `frameworks`) |
| Env | `build-vllm-xpu/env` |
| `ONEAPI_DEVICE_SELECTOR` | `level_zero:gpu` |
| `TRITON_INTEL_DEVICE_EXTENSIONS` | `cl_intel_subgroup_matrix_multiply_accumulate cl_intel_subgroup_matrix_multiply_accumulate_tensor_float32 cl_intel_subgroup_2d_block_io cl_intel_bfloat16_conversions` |
| `VLLM_XPU_FUSED_MOE_USE_REF` | `1` (fused XPU MXFP4 MoE alone → all `!`) |
| Attention | `attention_backend="TRITON_ATTN"` |
| TP | **8** (historical PASS). **Current best practice: TP=2** — see [`BEST_PRACTICE.md`](BEST_PRACTICE.md) |
| Other | `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `TORCHDYNAMO_DISABLE=1`, `TORCH_COMPILE_DISABLE=1`, `CCL_WORKER_COUNT=1`, unset `SYCL_CACHE_PERSISTENT`, durable `$WORKDIR/.cache/{triton,sycl}_xpu_gptoss` (aligned with `bench_perf_persist.pbs`) |
| Triton patch | `triton/backends/intel/driver.c` — OpenCL twin-device probe wrapped in try/catch (see `build-vllm-xpu/patches/triton_intel_driver_opencl_optional.txt`) |

Do **not** expose OpenCL in `ONEAPI_DEVICE_SELECTOR` (`*:gpu` / dual selector) — vLLM multiprocess SEGVs after smoke.

## Quality sample

Warmup / timed reply (head): Type I IUPAC isotherm for microporous Cu-BTC CO2 uptake; plateau from limited micropore volume; follow-up via temperature / high-P experiment.

`token_ids_head`: `[200005, 35644, 200008, 2167, 1309, 316, 6052, 3407, 5571, 11, 1641, 220, 1179, 6391, 11, 93194]`

## METRICS_JSON

```json
{"ttft_s":343.3135431089995,"prefill_tok_s":0.5009997521285995,"decode_tok_s":"n/a","n_prompt_tokens":172,"n_output_tokens":128,"finish_reason":"length","text_preview":"analysisWe need to answer three points, under 200 words, numbered. Provide IUPAC isotherm type: Type I (a) typical for microporous materials with strong adsorption at low pressure and plateau. Reason:","token_ids_head":[200005,35644,200008,2167,1309,316,6052,3407,5571,11,1641,220,1179,6391,11,93194],"wall_s":343.3135431089995,"e2e_tok_s":0.37283702483988795}
```

Notes: the reported “343 s TTFT” equals `wall_s` (e2e wall, not engine first-token — `ttft_source` would be `fallback_wall` after S1). Dominated by Triton JIT (`kernel_unified_attention`); e2e ~0.37 tok/s after cold compile. Decode tok/s n/a (no per-token timestamps in this path).

## Scripts

- `infer_chat.pbs` — PBS wrapper with recipe above  
- `one_chat.py` — TP=8 chat + metrics  
- `triton_xpu_smoke.py` — preflight Triton JIT gate  

## Related docs

- Project README: `../README.md`
- Versions / patches: `VERSIONS.md`, `patches/README.md`
- Phase log: `PHASE_STATUS.md`
- Training gate (after this): `SUCCESS_TRAIN.md`
