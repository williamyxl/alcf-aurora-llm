# vLLM high-concurrency serving recipe — gpt-oss-120b MXFP4 on Aurora (production)

The **high-concurrency best recipe** packaged as a hostable OpenAI-compatible service on one Aurora
node. Full deploy kit + operations in [`serve/`](serve/README.md).

## What it is
- **3× `vllm serve` (TP=4) across all 12 tiles** of one node (frameworks 0.15 + IPEX Marlin MXFP4),
  behind a stdlib round-robin load balancer on a single port → one OpenAI `/v1` endpoint.
- Runs on the **`capacity` queue** (max walltime **168h**), optionally **chained** for multi-week hosting.
- Measured **~4565 tok/s** aggregate output, **~36 req/s** at high concurrency (see `CONCURRENCY_RESULTS.md`).

## Why this shape
- gpt-oss heads=64 / kv_heads=8 → valid vLLM TP ∈ {2,4,8}; **TP=12 is invalid**. Use **data parallelism**
  (3× TP=4) to fill all 12 tiles.
- vLLM keeps MoE on-GPU (IPEX Marlin), so throughput scales with tiles — unlike llama.cpp MoE-offload
  which doesn't replicate (see `BEST_RECIPES.md`).
- Must run under the *full* `module load frameworks` env (sets SYCL/CCL the IPEX Marlin JIT needs).

## Deploy (see `serve/README.md` for detail)
```bash
cd /lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/gpt-oss-120b
# one week:
qsub build-vllm-xpu/opus4.8-new_campaign/serve/vllm_serve_node.pbs
# multi-week chain (afterany handoff, one node at a time):
build-vllm-xpu/opus4.8-new_campaign/serve/submit_chain.sh 2
```
Live endpoint → `serve/ENDPOINT.txt` (`lb_url=http://<node_ip>:8000/v1`). Refresh clients weekly at
each handoff (node changes). Reach it via SSH tunnel through a UAN (compute nodes have no public IP):
```bash
ssh -N -L 8000:<node_ip>:8000 you@aurora.alcf.anl.gov
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-oss-120b","messages":[{"role":"user","content":"Hi"}],"max_tokens":128}'
```

## Tuning knobs (`-v` on qsub)
| Var | Default | Effect |
|-----|---------|--------|
| `MNS` | 128 | max_num_seqs per engine = concurrency; raise for more in-flight requests |
| `KVGIB` | 20 | KV-cache pool per engine (GiB); raise with MNS / longer context |
| `MML` | 4096 | max context length (up to 131072; ~1–2% decode penalty at max) |
| `NREP`/`TPE` | 3 / 4 | engines × tiles-per-engine (must multiply to ≤12; TPE ∈ {2,4,8}) |
| `MEMUTIL` | 0.85 | gpu_memory_utilization |

## Operate
- Status: `qstat -u $USER`; LB health: `curl http://<node_ip>:8000/lb-health`
- Stop chain cleanly: `touch serve/STOP` (+ `qdel` queued successors)
- Logs: `serve/logs/{serve.out,engine_r*.out,lb.out}`

## Alternatives (see `BEST_RECIPES.md`)
- **Single user / low latency:** llama.cpp F4_hbm, 1 tile, 41.6 tok/s (`llama_f4hbm_ctx.pbs`).
- **Offline batch (no server):** `vllm_conc_fw.pbs` / `vllm_dp_node.pbs` measure aggregate throughput.

## Caveats
- Each engine loads its own model copy (~33 GiB HBM / 4 tiles). 3 engines fill the node.
- LB has no auth; node IP is intra-cluster only (or via your tunnel). Add a token/proxy if exposing.
- Endpoint changes on every (re)start — clients read `ENDPOINT.txt`.
