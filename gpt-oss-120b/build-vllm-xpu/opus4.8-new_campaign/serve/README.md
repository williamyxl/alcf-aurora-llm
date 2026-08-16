# Hosting gpt-oss-120b as a service on Aurora (capacity queue, up to 1 week)

Runs the **high-concurrency best recipe** (vLLM 3× TP=4 across all 12 tiles) as an OpenAI-compatible
service behind a load balancer, in one long PBS job.

## Architecture
```
client ──> LB :8000 (node IP, all interfaces)
              ├─ vllm serve  tiles 0-3  :8001
              ├─ vllm serve  tiles 4-7  :8002
              └─ vllm serve  tiles 8-11 :8003
```
One PBS job on the `capacity` queue (max walltime **168h**, ≤16 nodes, 2 running jobs/user). No
resubmission needed for a week-long job. `ENDPOINT.txt` is written with the live node IP:port.

## Files
- `vllm_serve_node.pbs` — the serving job (3× `vllm serve` + LB, writes `ENDPOINT.txt`).
- `lb.py` — stdlib async round-robin load balancer with health checks (`/lb-health`).
- `client_example.py` — OpenAI-compatible client; auto-reads `ENDPOINT.txt`.

## 1. Submit

**Single 1-week job:**
```bash
cd /lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/gpt-oss-120b
qsub build-vllm-xpu/opus4.8-new_campaign/serve/vllm_serve_node.pbs
# tune: -v MML=8192,MNS=192,KVGIB=30   (bigger context / more concurrency)
```

**Multi-week chain (recommended for continuous hosting):**
```bash
build-vllm-xpu/opus4.8-new_campaign/serve/submit_chain.sh 2   # 2 chained week-long jobs (~2 weeks)
```
Each successor uses `-W depend=afterany:<prev>`, so it starts when the predecessor **finishes** — one
node at a time, no extra cost, brief queue-wait gap at each weekly handoff. `capacity` allows 2 running
jobs/user, so also leave room if you run other jobs. **The node (endpoint) changes at each handoff** —
refresh clients from `ENDPOINT.txt` (fine for a weekly cadence). Successors honor `STOP` at startup, so
`touch serve/STOP` cleanly ends the chain (also `qdel` any queued successors to be sure).
Watch startup (model load ~1–2 min/engine): `tail -f build-vllm-xpu/opus4.8-new_campaign/serve/logs/serve.out`
When ready, `serve/ENDPOINT.txt` shows e.g. `lb_url=http://10.x.x.x:8000/v1`.

## 2. Reach the service

Aurora compute nodes have **no public IP** — reach them via a login node (UAN).

**From another Aurora node / login node (intra-cluster):** hit the node IP directly.
```bash
NODE_IP=$(grep '^ip=' .../serve/ENDPOINT.txt | cut -d= -f2)
curl http://$NODE_IP:8000/v1/models
```

**From your laptop (SSH jump through a UAN):**
```bash
NODE_IP=<ip from ENDPOINT.txt>
ssh -N -L 8000:$NODE_IP:8000 <you>@aurora.alcf.anl.gov
# then locally:
curl http://127.0.0.1:8000/v1/models
```

## 3. Call it (OpenAI API)
```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"gpt-oss-120b","messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'
# or:
python build-vllm-xpu/opus4.8-new_campaign/serve/client_example.py --n 64 --concurrency 64
```
Any OpenAI SDK works: `base_url=http://127.0.0.1:8000/v1`, `api_key="dummy"`, `model="gpt-oss-120b"`.

## 4. Capacity / expected performance
- ~4565 tok/s aggregate output at high concurrency (3× TP=4, mns=128); ~36 req/s. See `../CONCURRENCY_RESULTS.md`.
- Raise `MNS` and `KVGIB` for more in-flight requests (bounded by KV pool). Raise `MML` for longer context.

## 5. Operate
- **Status:** `qstat -u $USER`; LB health: `curl http://$NODE_IP:8000/lb-health`.
- **Stop gracefully:** `touch build-vllm-xpu/opus4.8-new_campaign/serve/STOP` (or `qdel <jobid>`).
- **Endpoint moves each (re)start** — clients should read `ENDPOINT.txt` (or keep an SSH tunnel that you
  repoint). For a fixed client endpoint, run a small reverse tunnel from a login node to the node IP.
- **Auto-renew past 1 week:** submit with `-v AUTO_RESUBMIT=1` to chain a successor near walltime.

## Notes / caveats
- Each engine loads its own model copy (~33 GiB HBM across its 4 tiles). 3 engines fill the node.
- `capacity` allows 2 running jobs/user — leave room if you also run other jobs.
- The LB has no auth; it binds the node IP which is only reachable intra-cluster or via your SSH tunnel.
  Do not expose beyond ALCF. Add a token/proxy if you need auth.
- Single-user low-latency instead? Use `../llama_f4hbm_ctx.pbs` (1 tile, 41.6 tok/s) — see `../BEST_RECIPES.md`.
