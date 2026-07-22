# Failed performance attempts (2026-07-21 ledger)

Quality gate: any `quality_ok=false` / all-`!` / token-id-0 is a **failed** recipe for production ranking.

| Campaign | Docs | Best speed (ignore quality) | Quality | Disposition |
|----------|------|----------------------------|---------|-------------|
| Fused MXFP4 TP=2/4/8 | [`FUSED_MOE_QUALITY.md`](FUSED_MOE_QUALITY.md) | ~5.2 decode @ TP=2 | **FAIL** all TP | Discard; fix kernels |
| Unquant BF16/FP16 TP=2/4/8 | [`HALFPREC_TP248.md`](HALFPREC_TP248.md) | ~3.0 decode @ BF16 TP=4 | **FAIL** (TP2 OOM) | Discard; casting ≠ bottleneck |
| mxfp4_fp8 (hist) | `PERF.md` / S3 | ~1.47 e2e @ TP=8 | **FAIL** | Discard |
| `enforce_eager=false` | S4 | ≈ REF | OK | No speed win |
| TP=12 | S5 | — | N/A | Invalid heads |

**Still quality-OK (slow):** REF MXFP4 + **TP=2** ≈ 1.22 decode — [`BEST_PRACTICE.md`](../BEST_PRACTICE.md).

**Hypothesis killed:** “MXFP4→BF16 upcast every forward is why we’re ~1 tok/s.” Half-prec resident weights still ≪13 and quality-broken.

**Do not use `module load frameworks`** for this project (no usable causal / gpt-oss inference path).

**Checkpoint policy:** stay on **MXFP4** (`models/openai-gpt-oss-120b`); do not retry BF16/FP16.

**Next:** llama.cpp SYCL from scratch — [`../../build-llamacpp-sycl/README.md`](../../build-llamacpp-sycl/README.md) · [`BETTER_SOLUTIONS.md`](BETTER_SOLUTIONS.md).
