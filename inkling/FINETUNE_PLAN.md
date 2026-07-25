# Inkling finetuning plan (Aurora)

**Status:** Phase 0–2 **PASS** (job `8701235`: 4-node FSDP2 LoRA smoke, 8 steps, `SUCCESS_TRAIN.md`). Adapter weight export still flaky under FSDP/DTensor (tokenizer saved; peft weights need gather-to-CPU).  
**Created:** 2026-07-24  
**Queues:** `debug` / `debug-scaling` only.  
**Reuse:** gpt-oss Phase 6 LoRA (`workdir/llm/gpt-oss-120b/` Torch-XPU + IPEX + peft/trl).

Inference **frozen** (`BEST_RECIPE.md`). Trainable weights: `thinkingmachines/Inkling` → `models/inkling-hf/` (**~1.894 TB** BF16 safetensors).

---

## Goal

Working **LoRA/SFT smoke** for Inkling (~975B MoE / ~41B active) on Aurora XPU, then small domain SFT (MOF / MatSci) with node cost fixed by the sizing below.

**Non-goals (v1):** full-parameter FT; train-from-GGUF; multi-node max-ctx train; replacing llama.cpp serve.

---

## Node sizing (authoritative)

Estimates from **GPU HBM/node**, **FP16/BF16 checkpoint size**, and **training-data disk** (data does not set GPU `select=`).

### Inputs

| Quantity | Value | Source |
|----------|------:|--------|
| GPU HBM / Aurora node | **768 GB** (6×PVC × 128 GB) | ALCF |
| Flat tiles / node | 12 × ~64 GB (same **768 GB** pool) | `ZE_FLAT_DEVICE_HIERARCHY=FLAT` |
| HF BF16 checkpoint | **1894 GB** (108 shards) | `models/inkling-hf/` measured |
| GGUF UD-IQ1_S | ~252 GB | infer only — ignore for train nodes |
| Host DDR / CPU-HBM | extra | dataloader / offload only — **not** in floor |

Text MoE: `hidden=6144`, `layers=66`, `n_routed_experts=256`, `num_experts_per_tok=6`, `n_shared_experts=2`.

### Weight-only floor

\[
N_{\mathrm{weights}} = \left\lceil \frac{1894}{768} \right\rceil = \left\lceil 2.47 \right\rceil = \mathbf{3\ nodes}
\]

**&lt;3 nodes cannot hold BF16 weights** (zero activations).

### Mode multipliers → node estimate

| Mode | GPU contents | × weights | **Est. `select=`** |
|------|----------------|----------:|-------------------:|
| Load / light `device_map` | weights (+ light KV) | 1.0–1.2× | **3–4** |
| **LoRA smoke** (short seq, grad ckpt) | + acts + LoRA grads/opt | 1.5–2.0× | **4–6** floor; **6–8** comfortable |
| LoRA longer \(B\times S\) | acts dominate | 2–3× | **8–16** |
| Full FT + Adam | params+grads+\(m,v\) | 8–16× | **≥20** / typically **32–64** |

### Training data (disk only)

| Corpus | Disk (~4 B/tok) | Effect on `select=` |
|--------|----------------:|---------------------|
| Smoke 8–64 rows | ≪1 GB | none |
| Small SFT 10 M–100 M tok | 0.04–0.4 GB | none |
| Medium ~1 B tok | ~4 GB | host cache only |
| Large ~10 B tok | ~40 GB | host streaming |

Budget **≤10 GB** under `inkling/data/` first. Raise nodes only when **seq/batch/acts** grow — not when JSONL grows.

### Committed schedule (from estimates)

| Phase | Default `select=` | Escalate on OOM | Queue |
|-------|------------------:|-----------------|-------|
| 0 download | 0 | — | — |
| 1 probe / load | **4** | **8 → 16** | `debug-scaling` (≥2); `debug` only if ≤2 |
| 2 LoRA smoke | **4** (prefer **8** if 4 tight) | **8 → 16** | `debug-scaling` |
| 3 domain SFT short | **8–16** | **16 → 32** | `debug-scaling` |
| 3 longer seq / larger LoRA | **16–32** | as needed | `debug-scaling` |
| 4 GGUF eval | **1** (PG8–10) | — | `debug` |
| 4 HF+adapter eval | **≥4** | same as load | `debug-scaling` |

**Do not** plan on 1-node BF16 load. A 1-node probe may run once as a negative control; treat OOM as **PASS for sizing**, then jump to **4**.

---

## Hard facts

| Fact | Implication |
|------|-------------|
| HF BF16 **1.894 TB** on disk | Phase 0 **PASS** |
| GGUF stays for llama.cpp | Infer-only |
| Floor **3** / LoRA **≥4** | Phase ladder **4→8→16** |
| gpt-oss LoRA + TRL `loss_type=nll` | Port for Phase 2 |
| Serve recipe frozen | Eval can reuse PG8–10 |

---

## Stack

| Option | Role |
|--------|------|
| **A — Torch-XPU + peft/trl** | Primary — gpt-oss `build-vllm-xpu/env` |
| B — llama.cpp train | Out (IQ1_S is infer quant) |
| C — CUDA / Tinker | Out (Aurora PVC) |

Train: HF BF16 → LoRA → adapter. Serve: existing GGUF; optional merge→GGUF later.

---

## Phases

### Phase 0 — Trainable weights — **PASS**

- [x] `download_inkling_hf.{sh,pbs}`
- [x] `models/inkling-hf/` (~1.894 TB, 108 shards + `config.json`)
- [x] `PHASE0_WEIGHTS.md`

**Nodes:** 0.

---

### Phase 1 — Arch + module discovery

**Deliverables**
- [x] `probe_inkling_modules.py` + `.pbs`
- [x] Loader: **`AutoModelForImageTextToText`** (not CausalLM)
- [x] `suggested_lora_targets`: **`q_proj`, `v_proj`** (`InklingAttention`)
- [ ] Optional: PBS `8694492` XPU load attempt (1 node; expect OOM)

**Exit gate:** LoRA targets known — **met**.

**Nodes:** module scan does not need 4; weight shard is Phase 2.

---

### Phase 2 — LoRA smoke (mirror gpt-oss Phase 6)

**Deliverables**
- [x] `train_lora_smoke.pbs` + `lora_one_epoch.py`
- [ ] Submit / harvest: LoRA `r=8`, targets q/v; TRL **`loss_type="nll"`**; 8 MOF rows; FSDP FULL_SHARD
- [ ] Adapter → `checkpoints/lora-smoke/adapter/` + `SUCCESS_TRAIN.md`

**Exit gate:** one epoch; `ok:true`.

**Nodes:** **`select=4`**, **48 ranks** (12 tiles/node) so ~1894 GB / 48 ≈ **39 GB/rank** (+ acts). OOM → **8 → 16**.

---

### Phase 3 — Scale SFT (small domain)

**Deliverables**
- [ ] MatSci/MOF JSONL ≤**10 GB** in `data/`; Inkling tokenizer
- [ ] FSDP2 / ZeRO-3; document `select=N`
- [ ] `max_seq_len` 512 → 2k (8k only if acts fit)
- [ ] `CYCLE_LOG_TRAIN.md`

**Exit gate:** ≥1 multi-node job + holdout quality.

**Nodes:** **8–16** short; **16–32** longer seq. Dataset GB on flare only.

---

### Phase 4 — Eval + optional merge

**Deliverables**
- [ ] HF+adapter vs base; optional merge→GGUF
- [ ] Train pointer in `BEST_RECIPE.md` (do not overwrite infer recipes)

**Nodes:** GGUF **1**; HF+adapter **≥4**.

---

## Agentic workflow

| Step | Action |
|------|--------|
| 1 | Phase 0 done — keep `PHASE0_WEIGHTS.md` |
| 2 | Harvest Phase 1; on fail/OOM → **`select=4`** then **8→16** (`debug-scaling`) |
| 3 | Phase 2 LoRA at **4+** (prefer proven Phase-1 `select=`) |
| 4 | `SUCCESS_TRAIN.md` → Phase 3 data (disk-sized) |
| 5 | Queues **debug / debug-scaling** only; no frameworks module |
| 6 | Stop wake on `SUCCESS_TRAIN.md` or user stop |

**Wake:** 10 m FT harvest (armed).

---

## File layout

```
inkling/
  FINETUNE_PLAN.md
  download_inkling_hf.{sh,pbs}
  probe_inkling_modules.{py,pbs}
  train_lora_smoke.pbs          # Phase 2
  lora_one_epoch.py
  models/inkling-hf/            # ~1.9TB (gitignored)
  data/                         # ≤10GB first (gitignored)
  checkpoints/lora-smoke/
```

---

## Open decisions

1. LoRA module names for `inkling_mm_model` (Phase 1).
2. Env fork if transformers needs newer pins than gpt-oss env.
3. ~~Min `select=` for LoRA~~ → **closed: 4 floor / 6–8 comfortable** (sizing table).

---

## Success definition

- Phase 0–2 + `SUCCESS_TRAIN.md`
- Actual `select=` recorded vs this schedule
- Go/no-go for Phase 3
- Infer `BEST_RECIPE.md` intact

---

## References

- gpt-oss: `../gpt-oss-120b/train_lora_smoke.pbs`, `lora_one_epoch.py`
- Inkling infer: `build-llamacpp-sycl/BEST_RECIPE.md`
- Aurora: **768 GB GPU HBM/node**; checkpoint **1894 GB** → weight floor **3**, LoRA start **4**
