#!/usr/bin/env python3
# Quick test client: send N concurrent requests to the gpt-oss-120b API endpoint.
# Works from any node that can reach the endpoint (intra-cluster or via SSH tunnel).
#
# Usage:
#   # read endpoint from ENDPOINT.txt (auto):
#   python client_example.py
#   # or specify explicitly:
#   python client_example.py --url http://<node_ip>:8000/v1 --n 10 --concurrency 10

import argparse, json, time, concurrent.futures, urllib.request, pathlib, sys

def read_endpoint():
    ep = pathlib.Path(__file__).parent / "ENDPOINT.txt"
    if ep.exists():
        d = dict(line.split("=",1) for line in ep.read_text().splitlines() if "=" in line)
        return d.get("lb_url", "http://127.0.0.1:8000/v1"), d.get("model","gpt-oss-120b")
    return "http://127.0.0.1:8000/v1", "gpt-oss-120b"

def chat(url, model, prompt, max_tokens=128):
    body = json.dumps({"model": model,
                       "messages": [{"role":"user","content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(url+"/chat/completions", data=body,
                                 headers={"Content-Type":"application/json",
                                          "Authorization":"Bearer dummy"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        obj = json.load(r)
    wall = time.perf_counter()-t0
    txt = obj["choices"][0]["message"].get("content","")
    toks = obj.get("usage",{}).get("completion_tokens",0)
    return wall, toks, txt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url",         default=None)
    ap.add_argument("--model",       default=None)
    ap.add_argument("--n",           type=int, default=4,  help="total requests")
    ap.add_argument("--concurrency", type=int, default=4,  help="parallel workers")
    ap.add_argument("--max-tokens",  type=int, default=128)
    args = ap.parse_args()

    base_url, model = read_endpoint()
    url   = args.url   or base_url
    model = args.model or model

    print(f"endpoint: {url}  model: {model}  n={args.n}  concurrency={args.concurrency}")
    prompts = [
        "In one sentence, what is a metal-organic framework?",
        "List three applications of high-performance computing in materials science.",
        "What is the difference between DFT and force-field molecular dynamics?",
        "Explain what an Aurora GPU tile is in one paragraph.",
    ]
    reqs = [prompts[i % len(prompts)] for i in range(args.n)]

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(chat, url, model, p, args.max_tokens) for p in reqs]
        results = [f.result() for f in concurrent.futures.as_completed(futs)]
    wall = time.perf_counter()-t0

    total_toks = sum(r[1] for r in results)
    print(f"\n{'='*60}")
    print(f"n={args.n}  concurrency={args.concurrency}  wall={wall:.1f}s")
    print(f"agg gen tok/s : {total_toks/wall:.1f}")
    print(f"req/s         : {args.n/wall:.3f}")
    for i,(w,t,txt) in enumerate(results):
        print(f"\n[{i}] {w:.2f}s  {t} tokens  {txt[:120]!r}")

if __name__ == "__main__":
    main()
