# gpt-oss-120b High-Concurrency API Serving on Aurora

OpenAI-compatible REST API (`/v1/chat/completions`, `/v1/models`) for **gpt-oss-120b**, served via
3× vLLM TP=4 engines + a round-robin load balancer on one Aurora node.

**Validated throughput:** ~4565 tok/s aggregate generation, ~36 req/s at 128-way concurrency.

---

## Architecture

```
laptop  ──SSH──►  UAN (aurora.alcf.anl.gov)  ──SSH tunnel──►  compute node
                  :8000 (uan_tunnel.sh)                        :8000 LB
                                                               ├─ vLLM engine 0  tiles 0-3
                                                               ├─ vLLM engine 1  tiles 4-7
                                                               └─ vLLM engine 2  tiles 8-11
```

Aurora compute nodes have **no public IP**. To reach the API from outside the cluster, you run a
two-hop SSH tunnel: laptop → UAN → compute node.

---

## Step 0 — prerequisites (one-time)

Pre-cache the `openai_harmony` tiktoken vocab on a login node (needed so the compute node can load
gpt-oss without internet access):

```bash
bash api_serve_high_concurrency/download_tiktoken_cache.sh
```

---

## Step 1 — start the API server (capacity queue, 24 h)

```bash
cd /lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm
qsub api_serve_high_concurrency/serve_gptoss_capacity.pbs

# watch startup (engines take ~5–10 min to load):
tail -f api_serve_high_concurrency/logs/serve.out

# when ready, the endpoint is in:
cat api_serve_high_concurrency/ENDPOINT.txt
```

Key parameters (pass with `qsub -v`):

| Var | Default | Description |
|-----|---------|-------------|
| `MNS` | 128 | Max concurrent sequences per engine (128 = best tested) |
| `KVGIB` | 20 | KV cache pool per engine (GiB); raise for more concurrency |
| `MML` | 4096 | Max context length (gpt-oss-120b native max) |
| `LB_PORT` | 8000 | External-facing load balancer port |

---

## Step 2 — open the API tunnel

### From inside ANL / another Aurora node (intra-cluster)

The compute node IP is already reachable; no tunnel needed:
```bash
NODE_IP=$(grep '^ip=' api_serve_high_concurrency/ENDPOINT.txt | cut -d= -f2)
curl http://$NODE_IP:8000/v1/models
```

### From a laptop / outside ANL — two hops

**Hop 1:** on a login node (UAN), run the tunnel script to forward the compute-node port to the UAN:
```bash
# on aurora.alcf.anl.gov (login node):
bash /lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/api_serve_high_concurrency/uan_tunnel.sh \
  /lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/api_serve_high_concurrency/ENDPOINT.txt 8000
# keeps running; prints: "UAN endpoint: http://<uan_ip>:8000/v1"
```

**Hop 2:** on your laptop, SSH-tunnel from the UAN to localhost:
```bash
# on your laptop:
ssh -N -L 8000:<uan_ip>:8000 <username>@aurora.alcf.anl.gov
# now the API is at http://127.0.0.1:8000/v1
```

Or combine both hops in one command:
```bash
ssh -N \
  -L 8000:<uan_ip>:8000 \
  <username>@aurora.alcf.anl.gov
```

---

## Step 3 — use the API

The endpoint is **OpenAI-compatible**. Any OpenAI SDK or `curl` works.

```bash
# models list
curl http://127.0.0.1:8000/v1/models

# chat completion
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss-120b","messages":[{"role":"user","content":"What is a MOF?"}],"max_tokens":256}'
```

**Python (OpenAI SDK):**
```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="dummy")
resp = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[{"role":"user","content":"What is a metal-organic framework?"}],
    max_tokens=256,
)
print(resp.choices[0].message.content)
```

**Quick load test:**
```bash
python api_serve_high_concurrency/client_example.py \
  --url http://127.0.0.1:8000/v1 --n 32 --concurrency 32
```

**Tool calling** is enabled (`--enable-auto-tool-choice --tool-call-parser openai`):
```python
resp = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[{"role":"user","content":"What's the weather in Chicago?"}],
    tools=[{"type":"function","function":{"name":"get_weather",
            "description":"Get weather","parameters":{"type":"object",
            "properties":{"city":{"type":"string"}},"required":["city"]}}}],
    tool_choice="auto",
)
```

---

## Operations

| Action | Command |
|--------|---------|
| Check status | `qstat -u $USER` |
| Watch logs | `tail -f api_serve_high_concurrency/logs/serve.out` |
| Per-engine logs | `tail -f api_serve_high_concurrency/logs/engine_r{0,1,2}.out` |
| LB health | `curl http://<node_ip>:8000/lb-health` |
| Graceful stop | `touch api_serve_high_concurrency/STOP` |
| Cancel job | `qdel <jobid>` |
| Update endpoint | `cat api_serve_high_concurrency/ENDPOINT.txt` (node IP changes each (re)start) |

---

## Notes

- **Endpoint moves every (re)start** — the compute node IP changes per job. Update your tunnel or
  re-read `ENDPOINT.txt`. For a fixed URL across restarts, keep the UAN tunnel running and only
  update which compute node it points at.
- **No auth** — the LB has no API key enforcement. The compute node IP is only reachable
  intra-cluster; the UAN tunnel is your access-control boundary.
- **Context limit** — gpt-oss-120b native max is 131072; the PBS defaults to 4096 (`MML=4096`).
  Raise with `-v MML=32768` (also raise `KVGIB` proportionally, e.g. `KVGIB=40`).
- **Resubmit** — capacity jobs run up to 24 h. To extend, `qsub` again; the new job writes a fresh
  `ENDPOINT.txt`. Update your tunnel to the new node IP.
