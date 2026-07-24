#!/usr/bin/env python
# Phase 0 warm-baseline: cold / warm / warm2 generates + PERF_JSON.
# Same PASS recipe as one_chat.py (TP=8, TRITON_ATTN, REF MoE via env).

from __future__ import annotations

import argparse
import json
import os
import time

import vllm_xpu_kernels._C  # noqa: F401
import vllm_xpu_kernels._moe_C  # noqa: F401

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DEFAULT_MODEL = (
    "/lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b/models/openai-gpt-oss-120b"
)

MESSAGES = [
    {
        "role": "user",
        "content": (
            "A researcher has adsorption isotherms for CO2 in a Cu-BTC MOF at 298 K. "
            "The uptake rises steeply at low pressure, then plateaus near 8 mmol/g by 1 bar.\n\n"
            "1. Name the isotherm type (IUPAC) most consistent with this shape and why.\n"
            "2. Give one physical reason the plateau appears.\n"
            "3. Propose one follow-up experiment to distinguish Langmuir-like saturation from pore filling.\n"
            "Answer in under 200 words, with numbered points."
        ),
    },
]


def parse_bool(s: str) -> bool:
    v = s.strip().lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {s!r}")


FILLER = (
    "Background note for long-context stress: porous crystalline frameworks store "
    "guest molecules in cages and channels; isotherm shape, hysteresis, and heat of "
    "adsorption constrain mechanism. Repeat filler packs the prefill KV budget. "
)


def quality_ok(text: str, token_ids) -> bool:
    """Reject all-bang garbage / token-id-0 collapse from known bad MoE/attn paths."""
    if not token_ids:
        return False
    if all(t == 0 for t in token_ids):
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if set(stripped) <= {"!", " ", "\n", "\t"}:
        return False
    return True


def build_prompt(tokenizer, prefill_tokens: int, max_model_len: int, max_tokens: int) -> tuple[str, int]:
    """Chat prompt; optionally pack filler so encoded length ≈ prefill_tokens."""
    margin = 64
    prompt = tokenizer.apply_chat_template(
        MESSAGES,
        tokenize=False,
        add_generation_prompt=True,
    )
    n_prompt_tokens = len(tokenizer.encode(prompt))

    if prefill_tokens <= 0:
        return prompt, n_prompt_tokens

    target = min(prefill_tokens, max_model_len - max_tokens - margin)
    if target <= n_prompt_tokens:
        print(
            f"prefill_tokens={prefill_tokens} capped_target={target} "
            f"base_n_prompt={n_prompt_tokens} (no pack needed)",
            flush=True,
        )
        return prompt, n_prompt_tokens

    # Binary-search filler repeat count so encoded length lands near target.
    lo, hi = 1, max(1, (target - n_prompt_tokens) * 2)
    best_prompt, best_n = prompt, n_prompt_tokens
    while lo <= hi:
        mid = (lo + hi) // 2
        packed = [
            {
                "role": "user",
                "content": (FILLER * mid) + "\n\n" + MESSAGES[0]["content"],
            }
        ]
        cand = tokenizer.apply_chat_template(
            packed,
            tokenize=False,
            add_generation_prompt=True,
        )
        n = len(tokenizer.encode(cand))
        if n <= target:
            best_prompt, best_n = cand, n
            lo = mid + 1
        else:
            hi = mid - 1

    print(
        f"prefill_pack target={target} actual_n_prompt_tokens={best_n} "
        f"(requested={prefill_tokens} max_model_len={max_model_len} "
        f"max_tokens={max_tokens} margin={margin})",
        flush=True,
    )
    return best_prompt, best_n


def run_metrics(out, n_prompt_tokens: int, wall_s: float) -> dict:
    """Prefer engine RequestStateStats; never invent TTFT from wall_s.

    Requires LLM(..., disable_log_stats=False) so RequestOutput.metrics is populated.
    """
    n_output_tokens = len(out.outputs[0].token_ids)
    text = out.outputs[0].text
    m = getattr(out, "metrics", None)

    ttft_s = None
    ttft_source = "fallback_wall"
    decode_tok_s = None
    prefill_tok_s = None
    if m is not None and getattr(m, "first_token_latency", 0) > 0:
        ttft_s = float(m.first_token_latency)
        ttft_source = "engine"
        prefill_tok_s = (n_prompt_tokens / ttft_s) if ttft_s > 0 else None
        if m.first_token_ts and m.last_token_ts and m.last_token_ts > m.first_token_ts:
            decode_s = m.last_token_ts - m.first_token_ts
            if n_output_tokens > 1 and decode_s > 0:
                decode_tok_s = (n_output_tokens - 1) / decode_s

    e2e_tok_s = (n_output_tokens / wall_s) if wall_s > 0 and n_output_tokens > 0 else 0.0
    return {
        "ttft_s": ttft_s,
        "ttft_source": ttft_source,
        "prefill_tok_s": prefill_tok_s,
        "e2e_tok_s": e2e_tok_s,
        "decode_tok_s": decode_tok_s,
        "n_prompt_tokens": n_prompt_tokens,
        "n_output_tokens": n_output_tokens,
        "wall_s": wall_s,
        "finish_reason": out.outputs[0].finish_reason,
        "text": text,
        "token_ids_head": list(out.outputs[0].token_ids[:16]),
        "quality_ok": quality_ok(text, out.outputs[0].token_ids),
    }


def generate_timed(llm, prompt, params, n_prompt_tokens: int, label: str) -> dict:
    print(f"=== {label} generate ===", flush=True)
    t0 = time.perf_counter()
    out = llm.generate([prompt], params)[0]
    wall_s = time.perf_counter() - t0
    metrics = run_metrics(out, n_prompt_tokens, wall_s)
    print(f"=== {label}_reply ===", flush=True)
    print(metrics["text"], flush=True)
    ttft = metrics["ttft_s"]
    ttft_disp = "null" if ttft is None else f"{ttft:.6f}"
    pref = metrics["prefill_tok_s"]
    pref_disp = "null" if pref is None else f"{pref:.6f}"
    dec = metrics["decode_tok_s"]
    dec_disp = "null" if dec is None else f"{dec:.6f}"
    print(
        f"{label}_ttft_s={ttft_disp} source={metrics['ttft_source']} "
        f"prefill_tok_s={pref_disp} decode_tok_s={dec_disp} "
        f"e2e_tok_s={metrics['e2e_tok_s']:.6f} "
        f"wall_s={metrics['wall_s']:.6f} "
        f"n_out={metrics['n_output_tokens']} "
        f"quality_ok={metrics['quality_ok']}",
        flush=True,
    )
    return metrics


def main():
    ap = argparse.ArgumentParser(description="gpt-oss-120b cold/warm/warm2 PERF_JSON bench")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="vLLM max_model_len (default 4096; P6 long-context uses 131072)",
    )
    ap.add_argument(
        "--prefill-tokens",
        type=int,
        default=0,
        help="if >0, pack filler so encoded prompt ≈ this many tokens "
        "(capped at max_model_len - max_tokens - margin)",
    )
    ap.add_argument("--moe-mode", default=None, help="label only; actual MoE path is env-driven")
    ap.add_argument(
        "--enforce-eager",
        type=parse_bool,
        default=True,
        help="vLLM enforce_eager (default true = PASS)",
    )
    ap.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.82,
        help="vLLM gpu_memory_utilization (default 0.82)",
    )
    ap.add_argument(
        "--max-num-seqs",
        type=int,
        default=16,
        help="vLLM max_num_seqs (default 16; use 1–2 for single-stream TP scaling)",
    )
    ap.add_argument(
        "--kv-cache-memory-gib",
        type=float,
        default=None,
        help="if set, pass kv_cache_memory_bytes=GiB*2^30 (bypasses util-based KV planner; "
        "needed on XPU when util over-allocates KV on top of large TP weight shards)",
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HF model dir (default: MXFP4 openai-gpt-oss-120b)",
    )
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "auto"),
        help="vLLM dtype (use float16 for FP16 checkpoints)",
    )
    args = ap.parse_args()
    model_path = args.model
    dtype = args.dtype

    moe_mode = args.moe_mode
    if moe_mode is None:
        if os.environ.get("VLLM_XPU_FUSED_MOE_USE_REF", "") == "1":
            moe_mode = "ref"
        elif os.environ.get("VLLM_XPU_FUSED_MOE_USE_MXFP4_FP8", "") == "1":
            moe_mode = "mxfp4_fp8"
        elif "bf16" in model_path.lower() or "fp16" in model_path.lower():
            moe_mode = "unquant_" + ("fp16" if "fp16" in model_path.lower() else "bf16")
        else:
            moe_mode = "fused"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt, n_prompt_tokens = build_prompt(
        tokenizer,
        prefill_tokens=args.prefill_tokens,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
    )

    print("=== prompt ===", flush=True)
    if args.prefill_tokens > 0 and n_prompt_tokens > 2000:
        # Avoid dumping ~120k tokens into the PBS log.
        print(prompt[:1500], flush=True)
        print(f"... [prompt truncated for log; n_prompt_tokens={n_prompt_tokens}] ...", flush=True)
        print(prompt[-500:], flush=True)
    else:
        print(prompt, flush=True)
    print(f"n_prompt_tokens={n_prompt_tokens}", flush=True)
    kv_bytes = None
    if args.kv_cache_memory_gib is not None:
        kv_bytes = int(args.kv_cache_memory_gib * (1 << 30))
    print(
        f"about_to_construct_LLM model={model_path} tp={args.tp} moe_mode={moe_mode} "
        f"dtype={dtype} max_model_len={args.max_model_len} "
        f"enforce_eager={args.enforce_eager} "
        f"gpu_memory_utilization={args.gpu_memory_utilization} "
        f"max_num_seqs={args.max_num_seqs} "
        f"kv_cache_memory_gib={args.kv_cache_memory_gib}",
        flush=True,
    )
    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=args.tp,
        dtype=dtype,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
        disable_custom_all_reduce=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        attention_backend="TRITON_ATTN",
        # P7: LLM() defaults disable_log_stats=True which leaves metrics=None
        # and forces fallback_wall TTFT. Must enable for first_token_latency.
        disable_log_stats=False,
    )
    if kv_bytes is not None:
        llm_kwargs["kv_cache_memory_bytes"] = kv_bytes
    llm = LLM(**llm_kwargs)
    print("LLM_constructed", flush=True)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    cold = generate_timed(llm, prompt, params, n_prompt_tokens, "cold")
    warm = generate_timed(llm, prompt, params, n_prompt_tokens, "warm")
    warm2 = generate_timed(llm, prompt, params, n_prompt_tokens, "warm2")

    all_ok = cold["quality_ok"] and warm["quality_ok"] and warm2["quality_ok"]
    preview = warm2["text"][:200] if warm2["text"] else cold["text"][:200]

    perf = {
        "n_tiles": args.tp,
        "model": model_path,
        "moe_mode": moe_mode,
        "attn": "TRITON_ATTN",
        "dtype": dtype,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "kv_cache_memory_gib": args.kv_cache_memory_gib,
        "cold_ttft_s": cold["ttft_s"],
        "warm_ttft_s": warm["ttft_s"],
        "warm2_ttft_s": warm2["ttft_s"],
        "cold_ttft_source": cold["ttft_source"],
        "warm_ttft_source": warm["ttft_source"],
        "warm2_ttft_source": warm2["ttft_source"],
        "cold_e2e_tok_s": cold["e2e_tok_s"],
        "warm_e2e_tok_s": warm["e2e_tok_s"],
        "warm2_e2e_tok_s": warm2["e2e_tok_s"],
        "cold_decode_tok_s": cold["decode_tok_s"],
        "warm_decode_tok_s": warm["decode_tok_s"],
        "warm2_decode_tok_s": warm2["decode_tok_s"],
        "cold_prefill_tok_s": cold["prefill_tok_s"],
        "warm_prefill_tok_s": warm["prefill_tok_s"],
        "warm2_prefill_tok_s": warm2["prefill_tok_s"],
        "n_prompt_tokens": n_prompt_tokens,
        "n_output_tokens": warm2["n_output_tokens"],
        "text_preview": preview,
        "quality_ok": all_ok,
        "runs": {
            "cold": {k: v for k, v in cold.items() if k != "text"},
            "warm": {k: v for k, v in warm.items() if k != "text"},
            "warm2": {k: v for k, v in warm2.items() if k != "text"},
        },
    }

    print("PERF_JSON=" + json.dumps(perf, separators=(",", ":")), flush=True)
    print("=== done ===", flush=True)
    if not all_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
