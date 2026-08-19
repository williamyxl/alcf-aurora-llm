#!/usr/bin/env python
# Example OpenAI-compatible client for the nemotron-3-ultra service.
# Reads the live endpoint from serve/ENDPOINT.txt (or pass --url).
#   python client_example.py                      # single prompt
#   python client_example.py --url http://<ip>:8000/v1
import argparse, json, os, urllib.request, concurrent.futures, time

HERE = os.path.dirname(os.path.abspath(__file__))

def endpoint_from_file():
    f = os.path.join(HERE, "ENDPOINT.txt")
    if os.path.exists(f):
        for line in open(f):
            if line.startswith("lb_url="):
                return line.split("=", 1)[1].strip()
    return None

def chat(url, prompt, max_tokens=128):
    body = json.dumps({
        "model": "nemotron-3-ultra",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(url + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer dummy"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    dt = time.time() - t0
    return r["choices"][0]["message"]["content"], dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=endpoint_from_file() or "http://127.0.0.1:8000/v1")
    ap.add_argument("--prompt", default="What is a metal-organic framework? One paragraph.")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()
    print("endpoint:", args.url, flush=True)
    if args.concurrency <= 1 and args.n <= 1:
        text, dt = chat(args.url, args.prompt, args.max_tokens)
        print(f"[{dt:.2f}s]\n{text}")
        return
    prompts = [f"{args.prompt} (variant {i})" for i in range(args.n)]
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        outs = list(ex.map(lambda p: chat(args.url, p, args.max_tokens), prompts))
    wall = time.time() - t0
    ok = sum(1 for t, _ in outs if t.strip())
    print(f"n={args.n} concurrency={args.concurrency} wall={wall:.1f}s ok={ok}/{args.n} "
          f"req/s={args.n/wall:.2f}")

if __name__ == "__main__":
    main()
