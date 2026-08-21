# In-Use Recipe: gpt-oss-120b High-Concurrency API on Aurora

**Status:** deployed and in production use (verified end-to-end via the OpenAI-compatible API).
This document records the exact recipe currently running so it can be reproduced or resumed.

## What is deployed

An OpenAI-compatible REST API serving **gpt-oss-120b** on a single Aurora node, using the validated
high-concurrency data-parallel recipe: **3 × vLLM (TP=4) engines behind one round-robin load balancer**
— a single unified API endpoint.

| Property | Value |
|----------|-------|
| Model | gpt-oss-120b (MXFP4 GGUF/HF, 120B params) |
| Engine | vLLM 0.15 (Aurora `frameworks/2025.3.1` module) + IPEX Marlin MXFP4 on XPU |
| Topology | 3 engines × TP=4 = 12 tiles (one full node); disjoint tiles 0-3 / 4-7 / 8-11 |
| Router | `lb.py` round-robin load balancer → one endpoint on `:8000` |
| Max context | 131072 (gpt-oss-120b native max) |
| Concurrency | `--max-num-seqs 128` per engine; KV pool 40 GiB/engine |
| Tool calling | enabled (`--enable-auto-tool-choice --tool-call-parser openai`) |
| Queue | `capacity`, 24 h walltime, 1 node |
| Peak throughput | ~4565 tok/s aggregate generation at saturation (~36 req/s) |
| Single-stream | ~25–30 tok/s decode |

## Files (this directory)

| File | Purpose |
|------|---------|
| `serve_gptoss_capacity.pbs` | The serving job: 3× vLLM TP=4 + LB; writes `ENDPOINT.txt`; runs 24 h |
| `download_tiktoken_cache.sh` | One-time: pre-cache the openai_harmony tiktoken vocab (login node) |
| `uan_tunnel.sh` | Forward the endpoint from the compute node to a login node (external access) |
| `client_example.py` | Concurrency test client (auto-reads `ENDPOINT.txt`) |
| `README.md` | Full API access guide (intra-cluster + laptop tunnel + SDK usage) |
| `ENDPOINT.txt` | Runtime state: current node IP / URL (NOT tracked; regenerated per run) |

The load balancer itself is reused from the campaign kit:
`../gpt-oss-120b/build-vllm-xpu/opus4.8-new_campaign/serve/lb.py`.

## Reproduce / restart

```bash
cd /lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm

# 0. one-time: cache the harmony vocab (login node, has proxy internet)
bash api_serve_high_concurrency/download_tiktoken_cache.sh

# 1. submit the serving job (defaults: MML=131072, MNS=128, KVGIB=40)
qsub api_serve_high_concurrency/serve_gptoss_capacity.pbs

# 2. wait for the endpoint (engines load ~5-15 min); URL appears here:
cat api_serve_high_concurrency/ENDPOINT.txt
```

Override knobs with `qsub -v`, e.g. `-v MML=32768,KVGIB=24,MNS=64`.

## Access the API

**API base URL:** read from `ENDPOINT.txt` — `lb_url=http://<node_ip>:8000/v1`
(model `gpt-oss-120b`, api_key any non-empty string).

- **From any Aurora login or compute node (intra-cluster):** connect directly to the node IP.
  ```bash
  IP=$(grep '^ip=' api_serve_high_concurrency/ENDPOINT.txt | cut -d= -f2)
  curl http://$IP:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"gpt-oss-120b","messages":[{"role":"user","content":"Hello"}],"max_tokens":256}'
  ```
- **From a laptop (outside ANL):** two-hop SSH tunnel — see `README.md`.

OpenAI SDK:
```python
from openai import OpenAI
client = OpenAI(base_url="http://<node_ip>:8000/v1", api_key="dummy")
r = client.chat.completions.create(model="gpt-oss-120b",
        messages=[{"role":"user","content":"What is a MOF?"}], max_tokens=512)
print(r.choices[0].message.content)
```

## Operational notes & lessons learned

- **Unified API, not 3 APIs.** Clients hit only `:8000`; `lb.py` round-robins to the 3 internal engines
  (`127.0.0.1:8001/8002/8003`). Each engine is a full data-parallel replica; the LB spreads the request
  stream and each engine's vLLM continuous-batcher handles its own concurrency. Check `/lb-health`.
- **Do NOT set `ONEAPI_DEVICE_SELECTOR` per engine.** Set only `ZE_AFFINITY_MASK` per engine (matching
  the validated `vllm_dp_node.pbs`). Forcing `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` conflicts with the
  IPEX Marlin JIT and crashes engines with "No device of requested type available" in the profiling
  forward. (This bug cost one failed launch; the recipe here is already fixed.)
- **harmony vocab must be cached.** gpt-oss loads an o200k tiktoken vocab via `openai_harmony`; run
  `download_tiktoken_cache.sh` once (proxy internet) or the engines fail with `HarmonyError`.
- **`max_tokens` errors / apparent hangs.** Client errors like
  `max_tokens must be at least 1, got -6837` come from the client computing
  `context_limit − prompt_tokens` with a too-small context — fixed by serving at `MML=131072` (done)
  and/or setting a sane `max_tokens` (e.g. 512). Long "hangs" are usually a large `max_tokens` × the
  single-stream decode rate; use `stream:true` and a modest `max_tokens`.
- **Response shape.** gpt-oss (harmony) returns a chain-of-thought in `message.reasoning_content` and
  the answer in `message.content` — read `content` for the answer.
- **Endpoint moves each (re)start** (node IP changes per job). Always re-read `ENDPOINT.txt`.
- **Stop:** `touch api_serve_high_concurrency/STOP` (graceful) or `qdel <jobid>`.

## Provenance

Recipe derived from the validated gpt-oss-120b campaign:
`../gpt-oss-120b/build-vllm-xpu/opus4.8-new_campaign/BEST_RECIPES.md`
(Recipe A — vLLM 3× TP=4 data-parallel, ~4565 tok/s) and its `serve/` kit.
