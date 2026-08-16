#!/usr/bin/env python
# vLLM high-concurrency throughput bench: send N prompts at once, measure aggregate tok/s.
from __future__ import annotations
import argparse, json, os, time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

BASE = ("What is a metal-organic framework (MOF)? Answer in one clear paragraph. "
        "Variation {i}: also mention one application.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--num-prompts", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--kv-cache-memory-gib", type=float, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    _rank = int(os.environ.get("RANK", os.environ.get("PMI_RANK", "0")))
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    def mk(i):
        msg = [{"role": "user", "content": BASE.format(i=i)}]
        try:
            if getattr(tok, "chat_template", None):
                return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
        return BASE.format(i=i)
    prompts = [mk(i) for i in range(args.num_prompts)]
    n_prompt_tok = sum(len(tok.encode(p)) for p in prompts)

    kw = dict(model=args.model, tensor_parallel_size=args.tp, dtype=args.dtype,
              trust_remote_code=True, max_model_len=args.max_model_len,
              enforce_eager=True, gpu_memory_utilization=args.gpu_mem_util,
              max_num_seqs=args.max_num_seqs, disable_log_stats=False)
    if args.kv_cache_memory_gib is not None:
        kw["kv_cache_memory_bytes"] = int(args.kv_cache_memory_gib * (1 << 30))

    print(f"model={args.model} tp={args.tp} mml={args.max_model_len} mns={args.max_num_seqs} "
          f"num_prompts={args.num_prompts} max_tokens={args.max_tokens}", flush=True)
    llm = LLM(**kw)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

    # warmup small batch
    llm.generate(prompts[:min(8, len(prompts))], params)

    t0 = time.perf_counter()
    outs = llm.generate(prompts, params)
    wall = time.perf_counter() - t0
    if _rank != 0:
        return
    n_out = sum(len(o.outputs[0].token_ids) for o in outs)
    ok = sum(1 for o in outs if (o.outputs[0].text or "").strip()
             and set((o.outputs[0].text or "").strip()) - {"!"," ","\n","\t"})
    agg_out_tps = n_out / wall
    agg_total_tps = (n_prompt_tok + n_out) / wall
    perf = dict(tag=args.tag, tp=args.tp, max_num_seqs=args.max_num_seqs,
                num_prompts=args.num_prompts, max_model_len=args.max_model_len,
                wall_s=round(wall, 3), n_out=n_out, n_prompt=n_prompt_tok,
                agg_output_tok_s=round(agg_out_tps, 1),
                agg_total_tok_s=round(agg_total_tps, 1),
                req_per_s=round(args.num_prompts / wall, 3),
                per_stream_out_tok_s=round(agg_out_tps / args.num_prompts, 3),
                quality_ok_frac=round(ok / len(outs), 3))
    print("CONC_JSON=" + json.dumps(perf, separators=(",", ":")), flush=True)
    print(f"[{args.tag}] mns={args.max_num_seqs} nprompts={args.num_prompts} "
          f"agg_out={agg_out_tps:.1f} tok/s req/s={args.num_prompts/wall:.2f} "
          f"wall={wall:.1f}s q_ok={ok}/{len(outs)}", flush=True)


if __name__ == "__main__":
    main()
