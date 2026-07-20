# A5b — SUCCESS_PERF notes

Wrote / updated:

1. **`build-vllm-xpu/SUCCESS_PERF.md`** — S2–S5 closure: best quality recipe still Phase 5 PASS (TP=8, REF MoE, TRITON_ATTN, eager, 4096, bf16); warm2 ≈0.37 tok/s; fused/fp8 discarded; TP=12 invalid; P6 pending; includes baseline `PERF_JSON` excerpt (job 8680399).
2. **`build-vllm-xpu/PERF.md`** — light pointer to SUCCESS_PERF + closure one-liner at top / team artifacts.
3. **`README.md`** — short Performance pointer (link only; Phase 5 PASS untouched).
4. **`.gitignore`** + **`FILES.md`** — allowlist `!build-vllm-xpu/SUCCESS_PERF.md`.

**Verdict one-liner:** Best quality-passing recipe remains ~0.37 warm e2e tok/s — not a speed breakthrough.
