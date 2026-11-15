#!/usr/bin/env python
# In-process engine high-concurrency test (offline LLM engine; same scheduler as `vllm serve`).
# Reports: load_s, single-stream, N-way concurrency (agg tok/s, req/s), quality gate. Chat template optional.
import argparse, json, os, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--mns", type=int, default=128)
    ap.add_argument("--nprompts", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--kv-gib", type=float, default=None)
    ap.add_argument("--mem-util", type=float, default=0.85)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--no-chat", action="store_true")
    ap.add_argument("--load-format", default="auto")
    ap.add_argument("--hf-overrides", default=None, help="JSON string, e.g. reduce layers for dummy-weight arch tests")
    ap.add_argument("--tag", default="serve")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    Q = "What is a metal-organic framework (MOF)? Answer in one short paragraph of clear English."
    def chat(q):
        if a.no_chat or not getattr(tok, "chat_template", None):
            return q
        m = [{"role": "user", "content": q}]
        try:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    r = dict(tag=a.tag, model=os.path.basename(a.model), tp=a.tp, eager=a.eager, ok=False, stage="init")
    try:
        kw = dict(model=a.model, tensor_parallel_size=a.tp, dtype=a.dtype, trust_remote_code=True,
                  max_model_len=a.max_model_len, enforce_eager=a.eager, gpu_memory_utilization=a.mem_util,
                  max_num_seqs=a.mns, disable_log_stats=False, load_format=a.load_format)
        if a.kv_gib: kw["kv_cache_memory_bytes"] = int(a.kv_gib * (1 << 30))
        if a.hf_overrides:
            kw["hf_overrides"] = json.load(open(a.hf_overrides[1:])) if a.hf_overrides.startswith("@") else json.loads(a.hf_overrides)
        r["stage"] = "construct"; t0 = time.perf_counter(); llm = LLM(**kw); r["load_s"] = round(time.perf_counter() - t0, 1)
        print("LLM_constructed", r["load_s"], flush=True)
        r["stage"] = "quality"
        sp = SamplingParams(temperature=0.0, max_tokens=160)
        o = llm.generate([chat(Q)], sp)[0].outputs[0]; txt = o.text
        r["text"] = txt[:240]; r["quality_ok"] = len(set(txt.split())) > 15
        print("QUALITY_TEXT=" + json.dumps(txt[:300]), flush=True)
        r["stage"] = "single"
        p1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        pN = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, ignore_eos=True)
        pr = chat(Q); npt = len(tok.encode(pr)); res = []
        for lab in ("cold", "warm"):
            t = time.perf_counter(); llm.generate([pr], p1); w1 = time.perf_counter() - t
            t = time.perf_counter(); oo = llm.generate([pr], pN)[0]; wN = time.perf_counter() - t
            n = len(oo.outputs[0].token_ids)
            res.append(dict(label=lab, ttft_s=round(w1, 3), prefill_tok_s=round(npt / w1, 1), decode_tok_s=round((n - 1) / (wN - w1), 2)))
        r["single"] = res; print("SINGLE", json.dumps(res), flush=True)
        r["stage"] = "conc"
        base = "What is a metal-organic framework (MOF)? Answer in one clear paragraph. Variation {i}: also mention one application."
        prompts = [chat(base.format(i=i)) for i in range(a.nprompts)]
        llm.generate(prompts[:8], pN)
        t = time.perf_counter(); outs = llm.generate(prompts, pN); wall = time.perf_counter() - t
        nout = sum(len(x.outputs[0].token_ids) for x in outs)
        okc = sum(1 for x in outs if len(set(x.outputs[0].text.split())) > 5)
        r["conc"] = dict(mns=a.mns, nprompts=a.nprompts, wall_s=round(wall, 2), agg_out_tok_s=round(nout / wall, 1), req_s=round(a.nprompts / wall, 2), ok_frac=round(okc / len(outs), 3))
        r["ok"] = True; r["stage"] = "done"
    except BaseException as e:
        import traceback
        r["err"] = f"{type(e).__name__}: {str(e)[:500]}"
        r["trace_tail"] = " | ".join(traceback.format_exc().splitlines()[-5:])[:700]
    print("SERVE_JSON=" + json.dumps(r), flush=True)

if __name__ == "__main__":
    main()
