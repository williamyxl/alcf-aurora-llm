#!/usr/bin/env python
# One-round gpt-oss-120b chat via vLLM with TTFT / tok/s metrics.

import json
import time

# Register XPU custom ops (torch.ops._C / _moe_C) before vLLM imports.
import vllm_xpu_kernels._C  # noqa: F401
import vllm_xpu_kernels._moe_C  # noqa: F401

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "/lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b/models/openai-gpt-oss-120b"

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


def metrics_from_output(out, n_prompt_tokens, wall_s):
    """Prefer engine RequestStateStats; never invent TTFT from wall_s."""
    m = getattr(out, "metrics", None)
    n_output_tokens = len(out.outputs[0].token_ids)
    text = out.outputs[0].text

    if m is not None and getattr(m, "first_token_latency", 0) > 0:
        ttft_s = float(m.first_token_latency)
        ttft_source = "engine"
        decode_s = None
        if m.first_token_ts and m.last_token_ts and m.last_token_ts > m.first_token_ts:
            decode_s = m.last_token_ts - m.first_token_ts
        if decode_s and n_output_tokens > 1:
            decode_tok_s = (n_output_tokens - 1) / decode_s
        else:
            decode_tok_s = "n/a"
        prefill_tok_s = n_prompt_tokens / ttft_s if ttft_s > 0 else None
    else:
        # No reliable first-token latency — do not masquerade wall as TTFT.
        ttft_s = None
        ttft_source = "fallback_wall"
        decode_tok_s = "n/a"
        prefill_tok_s = None

    e2e_tok_s = (n_output_tokens / wall_s) if wall_s > 0 and n_output_tokens > 0 else 0.0
    return {
        "ttft_s": ttft_s,
        "ttft_source": ttft_source,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "e2e_tok_s": e2e_tok_s,
        "wall_s": wall_s,
        "n_prompt_tokens": n_prompt_tokens,
        "n_output_tokens": n_output_tokens,
        "finish_reason": out.outputs[0].finish_reason,
        "text_preview": text[:200],
        "token_ids_head": list(out.outputs[0].token_ids[:16]),
    }, text


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        MESSAGES,
        tokenize=False,
        add_generation_prompt=True,
    )
    n_prompt_tokens = len(tokenizer.encode(prompt))

    print("=== prompt ===")
    print(prompt)
    print("=== loading model ===")
    print("about_to_construct_LLM", flush=True)
    # Best practice 2026-07-21: TP=2 ≫ TP=8 for BS=1 REF MoE (see BEST_PRACTICE.md).
    # Pin KV to avoid util-planner OOM on small TP (weights ~31 GiB/tile).
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_custom_all_reduce=True,
        gpu_memory_utilization=0.82,
        max_num_seqs=2,
        kv_cache_memory_bytes=8 * (1 << 30),
        # FLASH_ATTN decode garbles gpt-oss on XPU; Triton attn is the workaround.
        attention_backend="TRITON_ATTN",
        # P7: enable RequestStateStats (TTFT / prefill / decode).
        disable_log_stats=False,
    )
    print("LLM_constructed", flush=True)
    params = SamplingParams(temperature=0.0, max_tokens=128)

    print("=== warmup ===")
    warm = llm.generate([prompt], params)[0]
    print("=== warmup_reply ===")
    print(warm.outputs[0].text)
    print(
        "warmup_token_ids_head=",
        list(warm.outputs[0].token_ids[:16]),
        "finish=",
        warm.outputs[0].finish_reason,
    )

    print("=== timed generate ===")
    t0 = time.perf_counter()
    out = llm.generate([prompt], params)[0]
    wall_s = time.perf_counter() - t0
    metrics, text = metrics_from_output(out, n_prompt_tokens, wall_s)

    print("=== reply ===")
    print(text)
    print("=== metrics ===")
    ttft = metrics["ttft_s"]
    print(f"TTFT_s={'null' if ttft is None else f'{ttft:.6f}'} source={metrics['ttft_source']}")
    pref = metrics["prefill_tok_s"]
    print(f"prefill_tok_s={'n/a' if pref is None else f'{pref:.6f}'}")
    dts = metrics["decode_tok_s"]
    print(f"decode_tok_s={dts if isinstance(dts, str) else f'{dts:.6f}'}")
    print(f"wall_s={wall_s:.6f}")
    print(f"e2e_tok_s={metrics['e2e_tok_s']:.6f}")
    print("METRICS_JSON=" + json.dumps(metrics, separators=(",", ":")))
    print("=== done ===")


if __name__ == "__main__":
    main()
