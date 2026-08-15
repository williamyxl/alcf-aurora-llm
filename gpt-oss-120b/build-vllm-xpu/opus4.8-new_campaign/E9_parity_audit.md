# E9 — Parity / measurement audit (no GPU needed) — DONE

**Goal:** Before trusting the ~25–35× llama.cpp-vs-vLLM gap, confirm the two engines are compared
like-for-like: same prompt class, warm, **decode-only** metric, same tiles/context, coherent output.

**Inputs:** this campaign's `RESULTS.md` (llama.cpp reruns 2026-08-15) + documented vLLM REF numbers
in `../PATHS_TO_EXPECTED_PERF.md` / `../BEST_PRACTICE.md`.

## Metric definitions reconciled

| Quantity | llama.cpp (`common_perf_print`) | vLLM (`bench_perf.py`) | Comparable? |
|----------|----------------------------------|------------------------|-------------|
| TTFT | `prompt eval time` (wall to finish prompt) | engine `first_token_latency` (`ttft_source=engine`) | Yes — both are first-token, not e2e wall |
| Prefill tok/s | `prompt eval` tok/s | `n_prompt / ttft` | Yes |
| Decode tok/s | `eval` tok/s = `(runs)/(eval time)` | `(n_out-1)/(t_last-t_first)` | Yes — both exclude prompt |
| e2e tok/s | `total` tok/s | `n_out / wall` | Yes (secondary) |

Both harnesses report **decode separately from prefill**, so the headline decode comparison is
apples-to-apples (not e2e-including-prefill vs pure-gen).

## Like-for-like check

| Axis | llama.cpp rerun | vLLM REF (documented) | Match |
|------|-----------------|------------------------|-------|
| Model / quant | gpt-oss-120b MXFP4 GGUF | gpt-oss-120b MXFP4 HF | same weights/quant class |
| Prompt | MOF question, ~17–20 tok | MOF question, short | same class |
| Warm? | steady-state eval (post prompt) | warm2 generate | yes |
| Metric | decode tok/s (`eval`) | warm2 `decode_tok_s` (engine ts) | yes |
| Quality | coherent MOF answer (PASS) | REF PASS (fused = `!!!` FAIL) | quality-gated both |
| Tiles | F4_hbm=1 tile / P14_tp2=2 tiles | TP=2 | comparable (2-tile) |
| Context | 2048–4096 | 4096 | same short-ctx regime |

## Verdict

The gap is **real and like-for-like**:

| Engine (gpt-oss, 2-tile-class, decode, quality-OK) | decode tok/s |
|----------------------------------------------------|-------------:|
| llama.cpp P14_tp2 (2-tile GPU) | **34.0** |
| llama.cpp F4_hbm (1-tile MoE→CPU) | **41.8** |
| vLLM REF MoE TP=2 (documented) | **~1.2** |

→ **~28× (P14_tp2) to ~35× (F4_hbm)** decode gap, not a measurement artifact. This confirms the
premise of E1–E8 and the diagnosis in `PLAN.md` / `../../../cursor-opus4.8-med-diagnosis.md`.

**E9 status: PASS (gap validated).** The remaining discriminators (E1 profile, E2 REF expert-count
audit, E3 batch sweep, E4 bandwidth, E5 TP=1, E6 graph capture, E7 NUMA, E8 fused fix) require the
vLLM-XPU stack and are staged as ready-to-run scripts (`exp_vllm_bench.pbs`), blocked only on the
`build-vllm-xpu/env` build.
