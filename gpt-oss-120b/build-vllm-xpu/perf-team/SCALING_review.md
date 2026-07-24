# SCALING review — TP=2/4/8 harness

**Verdict: OK** — no blockers. PASS env intact (L0, REF MoE, TRITON_ATTN, eager, oneapi-only, no frameworks).
TP=2/4/8 all divide 64 attn / 8 KV heads; walltime 00:59:59 ≤1h.
Jobs match doc: TP4=8680707 (debug), TP2=8680711 (debug-scaling); TP8 baseline warm2≈0.372 (8680399).
Util 0.85 (TP2) / 0.82 (TP4) is intentional for per-tile weight memory — not a recipe drift.
