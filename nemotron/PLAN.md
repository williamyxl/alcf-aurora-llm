# Nemotron 3 Ultra — vLLM inference deployment on Aurora (PVC / Max 1550)

**Model:** NVIDIA Nemotron-3-Ultra-550B-A55B (catalog name "Nemotron 3 Ultra").
**Goal:** Deploy an OpenAI-compatible vLLM inference endpoint on Aurora, reusing the proven
gpt-oss-120b serving kit (`../gpt-oss-120b/build-vllm-xpu/opus4.8-new_campaign/serve/`).
**Status:** BLOCKED at model build by a vLLM-0.15 XPU MoE-kernel gap (non-gated relu² MoE unsupported
on PVC). Config/registration/INT4 all verified working via the `nemotron_h_shim.py`. **See
`FINDINGS.md` for the full result, evidence (job 8764932), and options.**

---

## 1. Model facts (from HF config.json)

| Property | Value |
|----------|-------|
| HF arch | `NemotronHForCausalLM` (`model_type: nemotron_h`) — **hybrid Mamba2 + attention + MoE** |
| Params | 550B total / ~55B active (MoE) |
| Layers | 108 total = 48 `mamba`, 48 `moe`, 12 `attention` |
| MoE | 512 routed experts, top-22 + 1 shared; latent-MoE (`moe_latent_size=2048`) |
| Attention | `num_attention_heads=64`, `num_key_value_heads=2`, `head_dim=128` |
| Mamba2 | `n_groups=8`, `mamba_num_heads=256`, `ssm_state_size=128`, `conv_kernel=4` |
| Context | `max_position_embeddings=262144` |
| Vocab | 131072, `tie_word_embeddings: False` |
| MTP | `num_nextn_predict_layers=1` (multi-token predict; optional speculative) |

### Available checkpoints & sizes
| Repo | Quant | Size | Aurora fit |
|------|-------|------|-----------|
| `nvidia/...-BF16` | bf16 | **1121 GB** | Needs ≥2 nodes; not single-node |
| `nvidia/...-NVFP4` | NVFP4 | 352 GB | **PVC has no native FP4** (INT4/INT8 only) — not usable |
| **`RedHatAI/...quantized.w4a16`** | **INT4 (compressed-tensors, group=128)** | **293 GB** | ✅ **single node** — chosen |

**Decision: use the RedHatAI w4a16 (INT4) checkpoint.** INT4 is natively supported on PVC and the
`compressed-tensors` quant method is present in the frameworks vLLM. NVFP4 is ruled out because PVC has
no native FP4 (established in the gpt-oss campaign; see that `DEBUG_LOG.md`). BF16 is ruled out for a
single node (1.1 TB > ~768 GB HBM/node).

---

## 2. Software stack (reuse gpt-oss campaign's proven env — do NOT rebuild)

Aurora `frameworks/2025.3.1` module env:
- vllm **0.15.0+xpu**, torch 2.10.0a0 (xpu), triton 3.6.0, ipex 2.10.10, compressed-tensors 0.13.0.
- `NemotronHForCausalLM` **is registered** in this vLLM (`.../vllm/model_executor/models/nemotron_h.py`).
- Mamba2 kernels (`mamba_ssm`, `causal_conv1d`, `ssd_*`) are **pure Triton `@triton.jit`** — the same
  Triton-3.6-on-XPU path that works for gpt-oss. This is the main feasibility unlock, but must be
  verified empirically (Triton-on-XPU had per-kernel crashes in the gpt-oss campaign).
- Must run under the **full `module load frameworks`** env (sets SYCL/CCL that the IPEX/Triton JIT need
  to select the XPU at forward time) — the key gpt-oss finding.

FWPY / VLLM binaries:
```
module use /opt/aurora/26.26.0/frameworks/modulefiles ; module load frameworks
FWPY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
VLLM=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/vllm
```

---

## 3. Parallelism / node layout

Aurora node = 6 GPUs × 2 tiles = **12 tiles**, ~64 GB HBM each (~768 GB/node).

**TP validity for this model:**
- Mamba2 `n_groups=8` ⇒ TP must divide 8 ⇒ **TP ∈ {1,2,4,8}** (TP=12 invalid).
- Attn `kv_heads=2`: vLLM replicates KV heads when TP>kv_heads ⇒ any of {1,2,4,8} OK.
- 64 query heads divisible by {2,4,8} ⇒ OK.

**Chosen serving config: single engine, TP=8** (8 of 12 tiles).
- INT4 weights ~293 GB / 8 = ~37 GB/tile → fits 64 GB with room for KV + Mamba state cache.
- TP=8 is the largest valid single-engine TP; needed because one INT4 copy (~293 GB) does not fit in
  fewer tiles with headroom. (Unlike gpt-oss, we cannot run 3× data-parallel copies — one INT4 copy
  already needs ~5 tiles minimum; a single TP=8 engine is the clean fit.)
- The remaining 4 tiles are idle on that node (can't form a second valid engine of the same model).
  If throughput demands it later, run a 2nd single-node job (2nd engine) behind the LB across nodes.

Fallbacks if TP=8 OOMs or a Mamba/quant kernel misbehaves at TP=8:
- Lower `--max-model-len` / `--kv-cache-memory` first.
- TP=8 with `--max-num-seqs 1` for a minimal smoke.

---

## 4. Queues (per ALCF docs — test on debug / debug-scaling only)

| Queue | Nodes | Walltime | Use here |
|-------|-------|----------|----------|
| `debug` | 1–2 | ≤1 h | **single-node probe + smoke + serve test** (select=1) |
| `debug-scaling` | 2–256 | ≤1 h | multi-node tests only (min 2 nodes) |

All test jobs: `-A MatSciAI -l filesystems=flare -l place=scatter`.
(Production long-running hosting would use `capacity` (≤168 h) as in the gpt-oss `serve/` kit, but only
after debug-queue validation — not part of this test task.)

---

## 5. Deliverables in this directory

| File | Purpose |
|------|---------|
| `download_nemotron_w4a16.sh` / `.pbs` | Resumable HF download of the INT4 checkpoint (~293 GB) |
| `probe_nemotron.pbs` + `probe_nemotron.py` | **feasibility probe**: load model TP=8, 1 short gen; catches Mamba/quant/XPU issues fast |
| `serve/vllm_serve_node.pbs` | single-node `vllm serve` TP=8 + round-robin LB, writes `ENDPOINT.txt` |
| `serve/lb.py`, `serve/client_example.py` | LB + client (adapted from gpt-oss kit) |
| `serve/serve_test_debug.pbs` | debug-queue (1 h) serve + self-smoke, for validation |
| `smoke_bench.py` | 2-call decode/prefill/TTFT bench (from gpt-oss `vllm_bench2.py`) |

---

## 6. Risk verification results (see FINDINGS.md for detail)

1. **Config/registration on XPU** — RESOLVED via `nemotron_h_shim.py` (AutoConfig gap + read-only
   `layers_block_type` property + crashing inspection subprocess all fixed).
2. **compressed-tensors INT4 (w4a16) on XPU** — WORKS (`XPUwNa16LinearKernel` selected on all 8 workers).
3. **Non-gated relu² MoE on XPU** — **BLOCKER (unresolved, hard engine gap).** `FusedMoE` rejects
   `is_act_and_mul=False` on XPU, and both XPU MoE kernels (Marlin + Triton) are SiLU/gated-only. This
   stops model construction. Nemotron-3-Ultra cannot run its MoE on vLLM-0.15 XPU today.
4. **Mamba2 Triton, KV/Mamba cache sizing, MTP** — not reached (build fails at the MoE layer first).
   Mamba2 kernels are pure Triton (expected to work); to be re-verified once the MoE gap is closed.
