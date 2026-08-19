#!/usr/bin/env python
# Benchmark llama-server (OpenAI /v1) for Nemotron-3-Ultra: TTFT, prefill TPS, decode TPS.
# Uses streaming to measure TTFT precisely (time to first token chunk), and the server's own
# `timings` block (llama.cpp) for prefill/decode when available. Runs a single-stream measurement
# then a concurrency sweep against the SAME already-loaded server.
#
#   python bench_client.py --url http://127.0.0.1:8000/v1 --model nemotron-3-ultra \
#       --max-tokens 128 --concurrency 1 4 8 16 32 --requests-per-level 0
#
# requests-per-level 0 => requests == concurrency (one wave). Set >0 to send more per level.

import argparse, json, time, statistics, urllib.request, concurrent.futures, sys

PROMPTS = [
    "Explain what a metal-organic framework is and give two applications.",
    "Summarize the theory of general relativity in a short paragraph.",
    "Write a short haiku about high performance computing.",
    "What are the main differences between TCP and UDP?",
    "Describe how a lithium-ion battery works, briefly.",
    "Give three tips for writing efficient parallel code.",
    "What is a Mixture-of-Experts model? Explain in a few sentences.",
    "Explain photosynthesis to a high-school student in one paragraph.",
]


def one_request(url, model, prompt, max_tokens):
    """Streaming request. Returns dict with ttft_s, decode_tps, gen_tokens, wall_s, server timings."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(url + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer dummy"})
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    server_timings = None
    usage = None
    resp = urllib.request.urlopen(req, timeout=600)
    for raw in resp:
        line = raw.decode("utf-8", "ignore").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except Exception:
            continue
        # first content token => TTFT
        choices = obj.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            if ("content" in delta and delta["content"]) or ("reasoning_content" in delta and delta.get("reasoning_content")):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n_chunks += 1
        if "timings" in obj and obj["timings"]:
            server_timings = obj["timings"]
        if obj.get("usage"):
            usage = obj["usage"]
    wall = time.perf_counter() - t0
    gen_tokens = (usage or {}).get("completion_tokens", n_chunks)
    decode_tps = None
    if ttft is not None and gen_tokens and gen_tokens > 1 and wall > ttft:
        decode_tps = (gen_tokens - 1) / (wall - ttft)
    return dict(ttft_s=ttft, decode_tps=decode_tps, gen_tokens=gen_tokens,
                wall_s=wall, server_timings=server_timings)


def run_level(url, model, max_tokens, concurrency, n_requests):
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n_requests)]
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(lambda p: one_request(url, model, p, max_tokens), prompts))
    wall = time.perf_counter() - t0
    ttfts = [r["ttft_s"] for r in results if r["ttft_s"] is not None]
    dtps = [r["decode_tps"] for r in results if r["decode_tps"] is not None]
    total_gen = sum(r["gen_tokens"] or 0 for r in results)
    # server-side prefill/decode (avg of per-request timings when present)
    sp = [r["server_timings"]["prompt_per_second"] for r in results
          if r["server_timings"] and "prompt_per_second" in r["server_timings"]]
    sd = [r["server_timings"]["predicted_per_second"] for r in results
          if r["server_timings"] and "predicted_per_second" in r["server_timings"]]
    out = dict(
        concurrency=concurrency, n_requests=n_requests, wall_s=round(wall, 2),
        ttft_ms_mean=round(statistics.mean(ttfts) * 1000, 1) if ttfts else None,
        ttft_ms_p50=round(statistics.median(ttfts) * 1000, 1) if ttfts else None,
        ttft_ms_max=round(max(ttfts) * 1000, 1) if ttfts else None,
        client_decode_tps_per_req_mean=round(statistics.mean(dtps), 2) if dtps else None,
        agg_gen_tps=round(total_gen / wall, 1) if wall > 0 else None,
        req_per_s=round(n_requests / wall, 3) if wall > 0 else None,
        server_prefill_tps_mean=round(statistics.mean(sp), 1) if sp else None,
        server_decode_tps_per_req_mean=round(statistics.mean(sd), 2) if sd else None,
        total_gen_tokens=total_gen,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="nemotron-3-ultra")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    ap.add_argument("--requests-per-level", type=int, default=0,
                    help="0 => n_requests == concurrency (one wave); else fixed count per level")
    ap.add_argument("--warmup", action="store_true", help="one throwaway request first")
    args = ap.parse_args()

    print(f"BENCH url={args.url} model={args.model} max_tokens={args.max_tokens} "
          f"levels={args.concurrency}", flush=True)

    if args.warmup:
        try:
            w = one_request(args.url, args.model, "Say hello.", 8)
            print(f"warmup ttft_ms={None if w['ttft_s'] is None else round(w['ttft_s']*1000,1)} "
                  f"gen={w['gen_tokens']}", flush=True)
        except Exception as e:
            print("warmup failed:", e, flush=True)

    all_rows = []
    for c in args.concurrency:
        n = c if args.requests_per_level <= 0 else args.requests_per_level
        try:
            row = run_level(args.url, args.model, args.max_tokens, c, n)
        except Exception as e:
            row = dict(concurrency=c, error=f"{type(e).__name__}: {e}")
        all_rows.append(row)
        print("BENCH_ROW=" + json.dumps(row, separators=(",", ":")), flush=True)

    print("\n=== SUMMARY (concurrency: TTFT p50 ms | agg gen tps | req/s | per-req decode tps) ===",
          flush=True)
    for r in all_rows:
        if "error" in r:
            print(f"  c={r['concurrency']:<3} ERROR {r['error']}", flush=True); continue
        print(f"  c={r['concurrency']:<3} TTFT_p50={r['ttft_ms_p50']} ms | "
              f"agg_gen={r['agg_gen_tps']} tps | {r['req_per_s']} req/s | "
              f"per_req_decode={r['client_decode_tps_per_req_mean']} tps", flush=True)
    print("BENCH_JSON=" + json.dumps(all_rows, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
