# llama.cpp SYCL campaign plan — MXFP4 → >30 tok/s; final Phase G = max ctx 131072 (2-tile vs 1-tile MoE→CPU)

**Owner track:** escape hatch from vLLM-XPU MoE (~1.2 tok/s quality-OK).  
**Hard stop:** **100 cycles** OR warm **decode/eval ≥ 30 tok/s** with **meaningful text**.  
**As of:** 2026-07-21

---

## Hard constraints (every cycle)

| Rule | Value |
|------|--------|
| Checkpoint | **MXFP4 only** — `models/openai-gpt-oss-120b` → `models/openai-gpt-oss-120b-mxfp4.gguf` |
| Hardware (Phase A–E) | **Exactly 2 tiles of one Max 1550** (same physical GPU) |
| Hardware (**Phase F**) | **One tile** — `FLAT` + `ZE_AFFINITY_MASK=0`; MoE on **CPU** + **local-DDR NUMA** (`numactl`) |
| Hardware (**Phase G**) | Max `N_CTX` (131072→64k→32k); G0=2-tile GPU; G1=1-tile MoE→CPU with **NUMA required** |
| Tile pin (default) | `ZE_FLAT_DEVICE_HIERARCHY=FLAT` + `ZE_AFFINITY_MASK=0,1` (GPU0 both tiles as 2 devices) |
| Tile pin (alt) | `COMPOSITE` + `ZE_AFFINITY_MASK=0` (implicit whole GPU0) — tried S2, OOM |
| Also set | `ZES_ENABLE_SYSMAN=1` |
| Forbidden (A–E) | BF16/FP16 unquant; TP=4/8; cross-GPU tiles; `module load frameworks`; **no MoE-CPU / partial `-ngl` until Phase F** — see [`PURE_GPU_BACKLOG.md`](PURE_GPU_BACKLOG.md) |
| Quality gate | Reject all-`!` / empty / token-id-0 / nonsensical garbage before ranking speed |
| Metric primary | llama.cpp **eval (decode) tok/s** after warmup; secondary: prompt tok/s, wall |

Source env snippet: [`same_gpu_2tiles.env.sh`](same_gpu_2tiles.env.sh) (Phase F/G1 overrides affinity to a single tile via `cycles/F*.env` / `G1*.env`).

Verify every job log contains:

```text
TILE_PIN_OK=1
# Phase A–E: sycl device count == 2 ; ZE_AFFINITY_MASK=0,1
# Phase F:   sycl device count == 1 ; ZE_AFFINITY_MASK=0 ; MoE override active (-ncmoe / -cmoe)
# Phase G:   N_CTX=131072 (or step-down) + FILL_CTX=1 ; same tile rules as G0/G1 arms
```

---

## Phases

### Phase A — Build SYCL binary ✅ DONE

| Item | Status |
|------|--------|
| Clone `ggml-org/llama.cpp@76f46ad` | OK |
| `GGML_SYCL=ON` + F16 + `DEVICE_ARCH=pvc` AOT | OK |
| Binaries | `build/bin/llama-cli`, `llama-bench`, `llama-server` |

Rebuild only if a cycle needs a code/kernel flag change. Script: `../build_llamacpp_sycl.pbs`.

### Phase B — MXFP4 GGUF (**prefer Hugging Face first**)

**Do not convert locally until HF has been checked.**

| Priority | Source | Notes |
|----------|--------|-------|
| **1 (preferred)** | [`ggml-org/gpt-oss-120b-GGUF`](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF) | Official llama.cpp MXFP4: `gpt-oss-120b-MXFP4.gguf` (~59 GiB / 63387346208 B). Also historical 3-part split names in older docs. |
| 2 | [`lmstudio-community/gpt-oss-120b-GGUF`](https://huggingface.co/lmstudio-community/gpt-oss-120b-GGUF) | Split MXFP4 (`*-00001/00002-of-00002.gguf`) |
| ❌ skip | unsloth Q2–Q8 / F16 GGUFs | Not our MXFP4 track |
| 3 (fallback) | Local convert from `models/openai-gpt-oss-120b` | Only if HF download fails |

Download helper: `download_gptoss_mxfp4_gguf.sh` / PBS `download_gptoss_mxfp4_gguf.pbs`.

**Already on disk (local convert, job 8681247):** `models/openai-gpt-oss-120b-mxfp4.gguf` (61G) — usable for smoke/perf; optional later switch to ggml-org file for bit-identical upstream pin.

### Phase C — Smoke until meaningful text (cycles S1…Sn)

**Goal:** coherent English (MOF / science prompt), not garbage.

Default smoke:

```bash
qsub smoke_llamacpp_sycl.pbs   # uses same-GPU 2-tile pin + MXFP4 GGUF
```

| Pass criteria | Fail → next smoke fix |
|---------------|------------------------|
| Non-empty text, not all-`!` | Check full GPU offload (`-ngl 99`); dump `GGML_SCHED_DEBUG=2` for CPU fallbacks |
| Answers prompt sensibly | Try `--jinja` + chat template; sampling `temp=1 top_p=1 top_k=0 min_p=0.01` (gpt-oss defaults) |
| MoE on device | If mxfp4 tensors stay on CPU → kernel/op gap; rebuild newer llama.cpp or patch SYCL ops |
| OOM | Confirm affinity is **same GPU** (not two GPUs); try implicit `ZE_AFFINITY_MASK=0` as cycle variant |

Log each smoke attempt in [`CYCLE_LOG.md`](CYCLE_LOG.md). **Do not enter Phase D until smoke PASS.**

### Phase D — Performance baseline (same 2 tiles)

```bash
qsub bench_llamacpp_sycl_perf.pbs
```

Protocol (fixed unless a cycle changes one knob):

- Prompt: fixed MOF one-paragraph prompt (same as smoke)
- `-n 128`, `-c 4096`, warmup then measure
- Prefer `llama-bench` **and** `llama-cli` timing lines
- Record: `eval_tok_s`, `prompt_tok_s`, `quality_ok`, job id, commit, env knobs

**Target:** `eval_tok_s ≥ 30`.

### Phase E — Iterate (cycles 1…100)

Each cycle = **one change** → rebuild if needed → smoke (if quality risk) → perf → update this plan + `CYCLE_LOG.md`.

#### If eval < 30 tok/s — ranked fix backlog

| Pri | Experiment | Why |
|-----|------------|-----|
| E1 | Confirm **zero** CPU offload of mxfp4 MoE (`GGML_SCHED_DEBUG`) | CPU MoE kills PVC speed |
| E2 | Implicit scaling: `ZE_AFFINITY_MASK=0` (still same GPU, both tiles) | Often better than layer-split |
| E3 | `-sm none` vs `-sm layer` on 2 visible tiles | Comm vs compute tradeoff |
| E4 | `GGML_SYCL_F16` already ON; try flash-attn / known SYCL flags from `docs/backend/SYCL.md` | Attn bottleneck |
| E5 | Bump llama.cpp tip (keep MXFP4); rebuild SYCL | Newer mxfp4/swiglu kernels |
| E6 | Batch/ubatch, `-ngl 99`, mmap off, threads pin | Host overhead |
| E7 | PVC env: large GRF / IGC opts from Intel SYCL guides | Kernel occupancy |
| E8 | Profile which op dominates (mul_mat_id mxfp4) | Directs patch vs config |

After each cycle: edit **Performance plan** section below with new number + next experiment. Re-run perf on **same 2 tiles**.

#### Stop conditions

1. `quality_ok` AND `eval_tok_s ≥ 30` → **SUCCESS**; freeze recipe in `BEST_RECIPE.md`
2. Cycle count == 100 without (1) → **STOP**; write failure analysis + recommend return to vLLM fused clamp / newer kernels

---

## Phase F — One GPU tile + MoE weights on CPU (NEW)

**Unlock condition:** Phase E pure-GPU 2-tile path has plateaued (~29.7–30.0 noise band). This phase is an intentional exception to the “no CPU offload / always 2 tiles” Phase E rule.

### Goal
Fit gpt-oss-120b MXFP4 on **one Max 1550 tile** (~64 GiB HBM) by keeping **MoE expert weights in host/CPU memory**, while attention / non-expert tensors stay on the GPU tile. Measure whether decode can beat or match the 2-tile pure-GPU recipe, and/or free the second tile for packing density.

### Hardware / env (every F cycle)

| Knob | Value |
|------|--------|
| Hierarchy | `ZE_FLAT_DEVICE_HIERARCHY=FLAT` |
| Affinity | `ZE_AFFINITY_MASK=0` (**one tile only** — GPU0 tile0) |
| Split | `-sm none` (single device; no tensor/layer split) |
| Offload | `-ngl 99` still (layers scheduled on GPU), **plus** MoE buffer override to CPU |
| **Host NUMA** | **Required for MoE→CPU** — see below (avoid remote / slow DDR) |
| VMM | `GGML_SYCL_ENABLE_VMM=0` (keep Aurora OOM fix) |
| Model | same MXFP4 GGUF |

### MoE→CPU mechanisms (llama.cpp CLI)

Prefer built-ins (in order):

| Flag | Meaning |
|------|---------|
| **`-ncmoe N` / `--n-cpu-moe N`** | First **N** layers’ MoE weights on CPU (preferred — works in **completion + llama-bench**) |
| **`-cmoe` / `--cpu-moe`** | All MoE experts on CPU (**completion only**; bench lacks this flag) |
| `-ot PATTERN=CPU` | Manual tensor override if finer control needed |

Example extras (F0 uses large N ≈ all layers):

```bash
EXTRA_ARGS='-fa on -ncmoe 99'   # F0: all MoE → CPU
EXTRA_ARGS='-fa on -ncmoe 16'   # F2: first 16 layers
```

### Host NUMA / CPU–GPU affinity (MoE weight placement)

**Authority:** [ALCF Aurora — MPI rank and thread binding](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/#mpi-rank-and-thread-binding-to-cores-and-gpus) and [Using the HBM on the Sapphire Rapids CPUs](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/#using-the-hbm-on-the-sapphire-rapids-cpus).

llama.cpp only chooses buffer type **CPU** vs **GPU**. Host **which** NUMA bank + which cores is via `numactl` (we use `--no-mmap` so first-touch applies).

#### Aurora node map (must match GPU tile mask)

| Socket | Usable HWTs (skip OS-reserved 0/52) | DDR NUMA | HBM NUMA | GPUs (COMPOSITE) / FLAT tiles |
|--------|-------------------------------------|----------|----------|--------------------------------|
| **0** | `1-51,105-155` | **0** (~512 GB) | **2** (64 GB) | GPUs **0,1,2** → FLAT tiles **0–5** |
| **1** | `53-103,157-207` | **1** (~512 GB) | **3** (64 GB) | GPUs **3,4,5** → FLAT tiles **6–11** |

Distances (from ALCF `numactl -H`): sock0 DDR↔local HBM = 13; sock0↔sock1 DDR = 21; sock0↔remote HBM = 23. **Slow path = remote socket.**

Our campaign: `ZE_AFFINITY_MASK=0` or `0,1` → **GPU0** → **always socket 0** (NUMA DDR=0, HBM=2).

| Control | How | Role |
|---------|-----|------|
| **`numactl` wrap** | `NUMACTL_ENABLE=1` + `NUMACTL_ARGS=…` | CPU + mem policy for MoE pages |
| **`--numa numactl`** | `LLAMA_NUMA=numactl` | llama.cpp uses the numactl CPU map |
| **Log** | Job prints `numactl -H` | Must show nodes **0–3** as above |

**Default MoE policy** (GPU0 / FLAT `0` or `0,1`) — socket-0 **DDR**, exclude reserved cores:

```bash
# cycles/* + numa_moe_host.env.sh
export NUMACTL_ENABLE=1
export NUMACTL_ARGS='--physcpubind=1-51,105-155 --membind=0'
export LLAMA_NUMA=numactl
# Harness: numactl $NUMACTL_ARGS … llama-completion|llama-bench … --numa numactl
```

**Optional variants** (ALCF-correct node IDs only):

| Variant | `NUMACTL_ARGS` | When |
|---------|----------------|------|
| Local DDR (default) | `--physcpubind=1-51,105-155 --membind=0` | Full MoE (~50–60 G) |
| CPU-HBM preferred | `--physcpubind=1-51,105-155 --preferred=2` | Faster BW; fall back to DDR |
| CPU-HBM bind | `--physcpubind=1-51,105-155 --membind=2` | Only if MoE fits in 64 G |
| Socket 1 | `--physcpubind=53-103,157-207 --membind=1` | **Only** if affinity is GPUs 3–5 / FLAT 6–11 |

Do **not** use HBM node ids `8–15` — that is **not** Aurora’s layout.

Snippet: [`numa_moe_host.env.sh`](numa_moe_host.env.sh). PBS: `bench_llamacpp_sycl_perf.pbs` wraps both completion and bench.

### Success / fail criteria

| Pass | Fail → next |
|------|-------------|
| Loads on 1 tile without OOM | Shrink ctx; try `-ncmoe` fewer/more layers; check HBM vs host RSS |
| Quality PASS (same MOF gate) | If garbage → CPU MoE path broken / wrong override |
| Record full metrics: TTFT, prefill_tps, gen_tps (+ bench) | Always append full row to `CYCLE_LOG.md` |
| NUMA policy logged (`numactl -H` + `NUMACTL_ARGS`) | If topology ≠ assumed node 0, retune `NUMACTL_ARGS` |
| Compare to best 2-tile (P11 / P14_tp2 ~29.9–30.0) | If much slower, still useful as **density** option (1 tile/model) |

### Planned F cycles

| Cycle | Change | Status |
|-------|--------|--------|
| **F0** | 1 tile + `-ncmoe 99` (no NUMA bind — baseline) | **DONE** gen=28.56 |
| **F1** | 1 tile + `-ncmoe 8` + **local-DDR NUMA** | planned / in flight |
| **F2** | 1 tile + `-ncmoe 16` + **local-DDR NUMA** | planned / in flight |
| **F3** | 1 tile + `-ncmoe 32` + **local-DDR NUMA** | planned |
| **F4** | Best F* (`-ncmoe 99`) + **local-DDR NUMA** (vs F0) | planned — `cycles/F4.env` |
| **F4h** | Same as F4 + **CPU-HBM preferred** | planned — `cycles/F4h.env` |
| **F5** | Best NUMA F* + IGC knobs (from P11) | planned |

Envs: `cycles/F0.env` … Scripts reuse `bench_llamacpp_sycl_perf.pbs` with `CYCLE=F0` (affinity from cycle env overrides default 0,1).

### Notes

- Expect **higher TTFT / lower gen_tps** than pure-GPU 2-tile if experts stream from host; win condition may be **fit + packing**, not peak tok/s.
- Do **not** confuse with Phase E parked “partial `-ngl`” — here layers stay GPU-scheduled; only MoE **weight buffers** are CPU.
- Still MXFP4-only; still no `module load frameworks`.
- **Never** leave MoE→CPU unbound on Aurora if avoidable — unbound often lands on default DDR, which may be remote relative to GPU0.

---

## Phase G — FINAL: max content length @ 131072 (2-tile GPU vs 1-tile MoE→CPU)

**Unlock:** After Phase F (or in parallel once F0 smoke-loads). This is the **last** campaign phase: stress the model’s native max context and compare the two deployment modes side-by-side.

### Model limit
From `models/openai-gpt-oss-120b/config.json`:

| Field | Value |
|-------|------:|
| `max_position_embeddings` | **131072** |
| `initial_context_length` | 4096 (YaRN base) |
| `rope_scaling.factor` | 32 |

### Goal
1. Set **`N_CTX=131072`** (max content length).
2. **Fill** the window (long prompt ≈ `N_CTX − N_PREDICT − margin`) + generate `N_PREDICT` tokens.
3. Run **two recipes** and record full metrics (TTFT, prefill_tps, gen_tps, + bench):

| Arm | Hardware | Offload | Host NUMA |
|-----|----------|---------|-----------|
| **G0** | 2 tiles (`FLAT 0,1`) pure GPU | none — P14_tp2 recipe (`-sm tensor -fa on -ts 0.5/0.5`) | Optional: local DDR for host threads (`NUMACTL` same as F) |
| **G1** | 1 tile (`FLAT 0`) | MoE→CPU (`-ncmoe 99`), attn on GPU | **Required** — same MoE NUMA policy as Phase F |

### Host NUMA (Phase G)

Same ALCF socket↔GPU map as Phase F. **G0/G1 use GPU0** (`FLAT 0,1` / `0`) → socket 0 only.

| Arm | Policy |
|-----|--------|
| **G1 / G1s / G1t** | **Mandatory** `--physcpubind=1-51,105-155 --membind=0` (or `--preferred=2` if F4h wins) |
| **G0 / G0s / G0t** | Same bind recommended (host threads next to GPU0) |

```bash
# G1* (MoE→CPU) — required (ALCF-correct)
export NUMACTL_ENABLE=1
export NUMACTL_ARGS='--physcpubind=1-51,105-155 --membind=0'
export LLAMA_NUMA=numactl
```

### Protocol / metrics

Every G cycle **must** append a full `CYCLE_LOG.md` row:

| Metric | Source |
|--------|--------|
| TTFT_ms | completion (`FILL_CTX=1` long prompt) |
| prefill_tps | completion prompt eval |
| gen_tps | completion eval |
| bench_* | llama-bench `-p $N_PP -n $N_PREDICT` (no `-c` on this build) |
| quality | MOF answer still coherent after long filler |
| NUMA | `NUMACTL_ARGS` + `numactl -H` excerpt in job log |

Harness: `FILL_CTX=1` → long prompt; on **debug/debug-scaling (≤59 m)** use `FILL_CTX_TOKENS=8192` while keeping `N_CTX=131072` for KV footprint.  
PBS: **`bench_llamacpp_sycl_phaseG.pbs`** — **debug / debug-scaling only** (no capacity). Full 128k prefill may not finish in 59 m → rely on step-downs G0s/G1s/G0t/G1t.

### Planned G cycles

| Cycle | N_CTX | Mode | NUMA | Status |
|-------|------:|------|------|--------|
| **G0** | **131072** | 2-tile pure GPU | optional local DDR | planned — `cycles/G0.env` |
| **G1** | **131072** | 1-tile + MoE→CPU | **local DDR required** | planned — `cycles/G1.env` |
| **G0s** | 65536 | 2-tile step-down | optional | planned |
| **G1s** | 65536 | 1-tile MoE→CPU step-down | **local DDR required** | planned |
| **G0t** | 32768 | 2-tile step-down | optional | planned |
| **G1t** | 32768 | 1-tile MoE→CPU step-down | **local DDR required** | planned |
| **G1h** | best G1 ctx | 1-tile MoE→CPU | CPU-HBM preferred (if F4h wins) | planned if needed |

### Submit

```bash
# debug / debug-scaling ONLY (max 59m) — do not use capacity
qsub -q debug-scaling -v CYCLE=G0 -N ll-G0 -o build-llamacpp-sycl/logs/perf_G0.out bench_llamacpp_sycl_phaseG.pbs
qsub -q debug         -v CYCLE=G1 -N ll-G1 -o build-llamacpp-sycl/logs/perf_G1.out bench_llamacpp_sycl_phaseG.pbs
# If OOM / walltime: G0s/G1s → G0t/G1t
```

### Pass / fail

| Pass | Action |
|------|--------|
| Loads + completes at target N_CTX | Record metrics; compare G0 vs G1 |
| OOM on KV / weights | Drop to next step-down (64k → 32k); note max workable ctx |
| Timeout mid-prefill | Same step-down; optionally lower `FILL_CTX_TOKENS` / `N_PP` |
| Quality FAIL after long fill | Flag; still keep speed numbers with note |
| MoE on wrong NUMA (no `numactl` in G1 log) | **FAIL policy** — resubmit with NUMA enabled |

### Notes

- Expect **much higher TTFT** and lower prefill_tps vs short-ctx Phase E/F; gen_tps may also drop as KV grows.
- 1-tile MoE→CPU frees GPU HBM for KV — may reach **higher N_CTX** than 2-tile pure GPU before OOM.
- Host NUMA binding does **not** replace GPU tile affinity; both are required for G1.
- Follow ALCF binding: GPU0 ↔ socket 0; never place MoE on sock1 when using `ZE_AFFINITY_MASK=0,1`.
- Still MXFP4-only; no `module load frameworks`.

---

## Phase H — MoE on Xeon Max HBM only (`--membind=2`) — **also on Inkling**

Cross-model twin of Inkling `MO_HBM` / `C_MO_HBM`. Force MoE host pages onto sock0 **HBM** (NUMA **2**), not DDR5.

| Cycle | Baseline | Change |
|-------|----------|--------|
| **F4_hbm** | F4h (`--preferred=2`) | `--membind=2` |
| **G1_hbm** | G1 (`--membind=0` DDR) | `--membind=2` |

```bash
numactl --physcpubind=1-51,105-155 --membind=2 \
  llama-completion ... -ngl 99 -sm none -fa on -ncmoe 99 -t 32 --numa numactl --no-mmap
```

gpt-oss MoE ~50–60 G may fit in 64 G HBM; Inkling UD-IQ1_S may OOM — still log metrics either way.

```bash
qsub -q debug -v CYCLE=F4_hbm -N ll-F4_hbm -o build-llamacpp-sycl/logs/perf_F4_hbm.out bench_llamacpp_sycl_perf.pbs
qsub -q debug -v CYCLE=G1_hbm -N ll-G1_hbm -o build-llamacpp-sycl/logs/perf_G1_hbm.out bench_llamacpp_sycl_phaseG.pbs
```

---

## Performance plan (living)

| Field | Current |
|-------|---------|
| Best short-ctx gen (2-tile GPU) | **30.01** (P14_tp2) |
| Best short-ctx gen (1-tile MoE+NUMA) | **33.78** (F4h HBM pref) / **33.28** (F4 DDR) |
| Best long-ctx gen (`N_CTX=131072`) | **32.66** (**G1** 1-tile MoE+NUMA) vs G0 28.61 |
| Best quality | PASS (MOF) on F* and G* |
| Recipe (2-tile short) | `FLAT 0,1` + VMM=0 + `-sm tensor` + `-fa on` + `-ts 0.5/0.5` |
| Recipe (1-tile MoE) | `FLAT 0` + `-ncmoe 99` + `numactl --physcpubind=1-51,105-155 --membind=0` (+ optional `--preferred=2`) |
| Campaign | **Phase F + G complete**; **Phase H** (HBM `--membind=2`) pending — also on Inkling |

Update this table every cycle.

---

## Job / script map

| Script | Role |
|--------|------|
| `same_gpu_2tiles.env.sh` | Default tile pin (2 tiles); cycle env may override to 1 tile for Phase F/G1 |
| `numa_moe_host.env.sh` | Default MoE host NUMA (local DDR + `--numa numactl`) |
| `make_fill_prompt.sh` | Phase G long-context filler prompt |
| `../build_llamacpp_sycl.pbs` | Rebuild SYCL |
| `../convert_gptoss_mxfp4_gguf.pbs` | HF MXFP4 → GGUF |
| `../smoke_llamacpp_sycl.pbs` | Quality smoke (2 tiles) |
| `../bench_llamacpp_sycl_perf.pbs` | Perf measure (Phases D–F; supports `FILL_CTX`) |
| `../bench_llamacpp_sycl_phaseG.pbs` | Phase G (`N_CTX` max); **debug/debug-scaling ≤59 m only** |
| `CYCLE_LOG.md` | Per-cycle ledger (**full** TTFT / prefill / gen columns) |
| `BEST_RECIPE.md` | Written on success |

---

## Cycle discipline

1. One independent variable per cycle.  
2. Phase A–E: MXFP4 + same-GPU **2 tiles**, no MoE-CPU.  
3. Phase F: MXFP4 + **1 tile** + MoE weights on CPU + **local-DDR NUMA** (except F0 unbound baseline).  
4. Phase G: max `N_CTX` · G0 2-tile vs G1 1-tile MoE→CPU (**G1 NUMA required**) · record full metrics.  
5. Phase H (**cross-model with Inkling**): MoE→CPU with **`--membind=2`** (Xeon Max HBM only, not DDR5) — short `F4_hbm` + long `G1_hbm`.  
6. Quality before speed ranking.  
7. Append **full** cycle row to `CYCLE_LOG.md` before submitting next job.  
8. Cap **100** cycles.  
9. **Login node:** no long-lived monitors (`nohup`, sleep loops). Use one-shot `harvest_once.sh` or agent `qstat` on demand.
