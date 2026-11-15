#!/usr/bin/env python
# Minimal stdlib async round-robin load balancer for N OpenAI-compatible vLLM backends.
# Streams responses. Skips backends that are down. No external deps.
#   python lb.py --port 8000 --backends http://127.0.0.1:8001 http://127.0.0.1:8002 ...
import argparse, asyncio, itertools, urllib.request, json, sys, time

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

class LB:
    def __init__(self, backends):
        self.backends = backends
        self.rr = itertools.cycle(range(len(backends)))
        self.alive = [True]*len(backends)

    def pick(self):
        for _ in range(len(self.backends)):
            i = next(self.rr)
            if self.alive[i]:
                return i
        return next(self.rr)  # all down: try anyway

async def handle(reader, writer, lb):
    try:
        # read request line + headers
        req_line = await reader.readline()
        if not req_line:
            writer.close(); return
        headers = b""
        clen = 0
        while True:
            line = await reader.readline()
            headers += line
            if line in (b"\r\n", b"\n", b""):
                break
            if line.lower().startswith(b"content-length:"):
                clen = int(line.split(b":")[1].strip())
        body = await reader.readexactly(clen) if clen else b""

        method, path, _ = req_line.decode("latin1").split(" ", 2)
        # health endpoint
        if path == "/lb-health":
            payload = json.dumps({"backends": lb.backends, "alive": lb.alive}).encode()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                         b"Content-Length: %d\r\n\r\n%s" % (len(payload), payload))
            await writer.drain(); writer.close(); return

        i = lb.pick(); backend = lb.backends[i]
        host = backend.split("://",1)[1]
        bhost, bport = (host.split(":") + ["80"])[:2]
        try:
            bre, bwr = await asyncio.open_connection(bhost, int(bport))
        except Exception as e:
            lb.alive[i] = False
            msg = b'{"error":"backend down"}'
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: %d\r\n\r\n%s" % (len(msg), msg))
            await writer.drain(); writer.close(); return

        # forward request (rewrite Host)
        out = b"%s %s HTTP/1.1\r\n" % (method.encode(), path.encode())
        out += b"Host: %s\r\n" % host.encode()
        for hl in headers.split(b"\r\n"):
            if hl and not hl.lower().startswith(b"host:"):
                out += hl + b"\r\n"
        out += b"\r\n" + body
        bwr.write(out); await bwr.drain()

        # stream response back
        while True:
            chunk = await bre.read(65536)
            if not chunk: break
            writer.write(chunk); await writer.drain()
        bwr.close()
    except Exception as e:
        try: writer.close()
        except Exception: pass

async def healthcheck(lb, interval=15):
    while True:
        for i, b in enumerate(lb.backends):
            try:
                urllib.request.urlopen(b + "/health", timeout=3)
                if not lb.alive[i]: log("backend UP", b)
                lb.alive[i] = True
            except Exception:
                if lb.alive[i]: log("backend DOWN", b)
                lb.alive[i] = False
        await asyncio.sleep(interval)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--backends", nargs="+", required=True)
    args = ap.parse_args()
    lb = LB(args.backends)
    asyncio.create_task(healthcheck(lb))
    server = await asyncio.start_server(lambda r,w: handle(r,w,lb), args.host, args.port)
    log(f"LB listening on {args.host}:{args.port} -> {args.backends}")
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
