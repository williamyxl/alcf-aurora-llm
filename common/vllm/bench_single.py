#!/usr/bin/env python
# vLLM 0.27.1 XPU bench for the new campaign (torch 2.13+xpu stack).
# Records TTFT (engine), prefill tok/s, decode tok/s, e2e tok/s, quality gate.
# Works for gpt-oss-120b (MXFP4 HF) and other HF models.

from __future__ import annotations
import argparse, json, os, time

# Map Aurora PALS PMI env -> torch/vLLM distributed env (for external_launcher).
def _map_pmi_env():
    rank = os.environ.get("PALS_RANKID") or os.environ.get("PMI_RANK")
    size = os.environ.get("PALS_LOCAL_SIZE") or os.environ.get("PMI_SIZE") \
        or os.environ.get("WORLD_SIZE")
    lrank = os.environ.get("PALS_LOCAL_RANKID") or os.environ.get("MPI_LOCALRANKID") or rank
    wsize = os.environ.get("PMI_SIZE") or os.environ.get("WORLD_SIZE") or size
    if rank is not None:
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("LOCAL_RANK", str(lrank if lrank is not None else rank))
    if wsize is not None:
        os.environ.setdefault("WORLD_SIZE", str(wsize))
    os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "127.0.0.1"))
    os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29500"))

_map_pmi_env()

# Optional: patch vLLM CPU weight-offload to NOT use the CUDA-only UVA op (XPU has no
# get_cuda_view_from_cpu_tensor). With uva_offloading=False, offloaded params stay as plain
# CPU tensors (p.data = cpu_data). Enables --cpu-offload-gb on XPU (single-tile experiment).
if os.environ.get("VLLM_BENCH_XPU_OFFLOAD_PATCH") == "1":
    try:
        import vllm.model_executor.models.utils as _mu
        import torch as _torch
        _orig = _mu.maybe_offload_to_cpu
        def _xpu_offload(module):
            params = next(module.parameters(), None)
            if params is None or params.device == _torch.device("cpu"):
                return module
            if _mu._CPU_OFFLOAD_BYTES >= _mu._CPU_OFFLOAD_MAX_BYTES:
                return module
            offloaded = False
            for p in module.parameters():
                if _mu._CPU_OFFLOAD_BYTES >= _mu._CPU_OFFLOAD_MAX_BYTES:
                    break
                cpu_data = _torch.empty_strided(size=p.data.size(), stride=p.data.stride(),
                                                dtype=p.data.dtype, layout=p.data.layout,
                                                device="cpu")
                cpu_data.copy_(p.data)
                p.data = cpu_data  # plain CPU tensor, no UVA
                _mu._CPU_OFFLOAD_BYTES += p.data.numel() * p.data.element_size()
                offloaded = True
            if offloaded:
                _orig_fwd = module.forward
                _dev = params.device
                def _fwd(*a, **k):
                    for p in module.parameters():
                        if p.data.device.type == "cpu":
                            p.data = p.data.to(_dev)
                    return _orig_fwd(*a, **k)
                module.forward = _fwd
            return module
        _mu.maybe_offload_to_cpu = _xpu_offload
        print("XPU offload patch active (uva_offloading=False, per-fwd copy-to-device)", flush=True)
    except Exception as _e:
        print("XPU offload patch FAILED:", _e, flush=True)

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MOF_MSG = [{"role": "user", "content": (
    "What is a metal-organic framework (MOF)? "
    "Answer in one short paragraph of clear English."
)}]


def quality_ok(text, tok_ids):
    if not tok_ids or all(t == 0 for t in tok_ids):
        return False
    s = (text or "").strip()
    if not s or set(s) <= {"!", " ", "\n", "\t"}:
        return False
    return True


def _gen(llm, prompt, max_tokens):
    from vllm import SamplingParams
    p = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    out = llm.generate([prompt], p)[0]
    wall = time.perf_counter() - t0
    o = out.outputs[0]
    return wall, len(o.token_ids), o


def run(llm, prompt, params, n_prompt, label, max_tokens):
    # Two-call method (version-agnostic): t1 = prefill+1 token, tN = prefill + N tokens.
    # decode_tps = (N-1)/(tN - t1); prefill/ttft ~= t1 (prompt eval + first token).
    w1, n1, _ = _gen(llm, prompt, 1)
    wN, nN, oN = _gen(llm, prompt, max_tokens)
    ttft = w1
    prefill_tps = (n_prompt / w1) if w1 > 0 else None
    decode_tps = ((nN - 1) / (wN - w1)) if (wN > w1 and nN > 1) else None
    e2e = nN / wN if wN > 0 else 0.0
    q = quality_ok(oN.text, oN.token_ids)
    r = dict(label=label, ttft_s=ttft, ttft_source="two_call", prefill_tok_s=prefill_tps,
             decode_tok_s=decode_tps, e2e_tok_s=e2e, n_prompt=n_prompt, n_out=nN,
             quality_ok=q, text=oN.text[:200])
    print(f"[{label}] ttft={ttft:.4f}s prefill_tps={prefill_tps} "
          f"decode_tps={decode_tps} e2e={e2e:.4f} n_out={nN} q={q}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=2)
    ap.add_argument("--kv-cache-memory-gib", type=float, default=None)
    ap.add_argument("--enforce-eager", action="store_true", default=True)
    ap.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tag", default="")
    ap.add_argument("--distributed-executor-backend", default=None,
                    help="e.g. external_launcher when launched under mpiexec -n TP")
    ap.add_argument("--cpu-offload-gb", type=float, default=0.0)
    ap.add_argument("--attention-backend", default=None, help="e.g. TRITON_ATTN")
    args = ap.parse_args()

    # Under external_launcher, only rank 0 should print PERF_JSON.
    _rank = int(os.environ.get("RANK", os.environ.get("PMI_RANK", "0")))

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        if getattr(tok, "chat_template", None):
            prompt = tok.apply_chat_template(MOF_MSG, tokenize=False, add_generation_prompt=True)
        else:
            raise ValueError("no chat template")
    except Exception:
        prompt = MOF_MSG[0]["content"]
    n_prompt = len(tok.encode(prompt))
    print(f"model={args.model} tp={args.tp} mml={args.max_model_len} "
          f"mem_util={args.gpu_mem_util} mns={args.max_num_seqs} "
          f"eager={args.enforce_eager} kv_gib={args.kv_cache_memory_gib} n_prompt={n_prompt}", flush=True)

    kw = dict(model=args.model, tensor_parallel_size=args.tp, dtype=args.dtype,
              trust_remote_code=True, max_model_len=args.max_model_len,
              enforce_eager=args.enforce_eager, gpu_memory_utilization=args.gpu_mem_util,
              max_num_seqs=args.max_num_seqs, disable_log_stats=False)
    if os.environ.get("VLLM_BENCH_SIMPLE_SCHED") == "1":
        kw["enable_prefix_caching"] = False
        kw["enable_chunked_prefill"] = False
    if os.environ.get("VLLM_BENCH_NO_CUSTOM_OPS") == "1":
        kw["compilation_config"] = {"custom_ops": ["none"]}
    if args.kv_cache_memory_gib is not None:
        kw["kv_cache_memory_bytes"] = int(args.kv_cache_memory_gib * (1 << 30))
    if args.distributed_executor_backend:
        kw["distributed_executor_backend"] = args.distributed_executor_backend
    if args.cpu_offload_gb and args.cpu_offload_gb > 0:
        kw["cpu_offload_gb"] = args.cpu_offload_gb
    if args.attention_backend:
        # vLLM 0.11.x: no attention_backend kwarg; use env. 0.27.x: accepts kwarg.
        import vllm as _v
        _ver = tuple(int(x) for x in _v.__version__.split(".")[:2] if x.isdigit())
        if _ver >= (0, 27):
            kw["attention_backend"] = args.attention_backend
        else:
            os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_backend

    t0 = time.perf_counter()
    llm = LLM(**kw)
    print(f"LLM_constructed load_s={time.perf_counter()-t0:.1f}", flush=True)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    cold = run(llm, prompt, params, n_prompt, "cold", args.max_tokens)
    warm = run(llm, prompt, params, n_prompt, "warm", args.max_tokens)
    warm2 = run(llm, prompt, params, n_prompt, "warm2", args.max_tokens)

    if _rank != 0:
        print("=== done (non-zero rank) ===", flush=True)
        return
    perf = dict(tag=args.tag, model=args.model, tp=args.tp,
                max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                enforce_eager=args.enforce_eager, gpu_mem_util=args.gpu_mem_util,
                kv_cache_memory_gib=args.kv_cache_memory_gib,
                warm2_ttft_s=warm2["ttft_s"], warm2_prefill_tok_s=warm2["prefill_tok_s"],
                warm2_decode_tok_s=warm2["decode_tok_s"], warm2_e2e_tok_s=warm2["e2e_tok_s"],
                warm_decode_tok_s=warm["decode_tok_s"], cold_decode_tok_s=cold["decode_tok_s"],
                quality_ok=all(r["quality_ok"] for r in (cold, warm, warm2)),
                text_preview=warm2["text"])
    print("PERF_JSON=" + json.dumps(perf, separators=(",", ":")), flush=True)
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
