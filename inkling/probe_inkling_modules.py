#!/usr/bin/env python
"""Phase 1: list LoRA-able modules + attempt XPU load smoke for Inkling HF.

Checkpoint is multimodal (`inkling_mm_model` / InklingForConditionalGeneration).
Use AutoModelForImageTextToText — AutoModelForCausalLM only maps inkling_text.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

MODEL = Path("/lus/flare/projects/MOFA/xiaoliyan/workdir/llm/inkling/models/inkling-hf")
OUT = Path("/lus/flare/projects/MOFA/xiaoliyan/workdir/llm/inkling/build-llamacpp-sycl/logs/phase1_probe.json")


def _scan_modules(model) -> dict:
    names = [n for n, _ in model.named_modules()]
    candidates = sorted(
        {
            n.split(".")[-1]
            for n in names
            if n.split(".")[-1]
            in {
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "w1",
                "w2",
                "w3",
                "wq",
                "wk",
                "wv",
                "wo",
            }
            or any(x in n.split(".")[-1] for x in ("q_proj", "v_proj", "query", "value"))
        }
    )
    leaf: dict[str, int] = {}
    for _, m in model.named_modules():
        leaf[type(m).__name__] = leaf.get(type(m).__name__, 0) + 1
    prefer = [c for c in ("q_proj", "v_proj") if c in candidates]
    return {
        "module_type_counts": dict(sorted(leaf.items(), key=lambda kv: -kv[1])[:40]),
        "lora_candidate_suffixes": candidates,
        "n_named_modules": len(names),
        "suggested_lora_targets": prefer or candidates[:8],
    }


def main():
    import torch
    import intel_extension_for_pytorch as ipex  # noqa: F401
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    report: dict = {"model": str(MODEL), "ok": False, "loader": "AutoModelForImageTextToText"}
    print(f"host={os.uname().nodename}", flush=True)
    print(f"xpu_count={torch.xpu.device_count()}", flush=True)
    report["xpu_count"] = int(torch.xpu.device_count())

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    report["tokenizer"] = type(tok).__name__
    report["vocab_size"] = getattr(tok, "vocab_size", None)
    print(f"tokenizer_ok={report['tokenizer']} vocab={report['vocab_size']}", flush=True)

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    report["architectures"] = getattr(cfg, "architectures", None)
    report["model_type"] = getattr(cfg, "model_type", None)
    print(f"config model_type={report['model_type']} arch={report['architectures']}", flush=True)

    # --- A) meta / empty weights: module graph without 1.9TB HBM ---
    print("building empty model (init_empty_weights) for module scan…", flush=True)
    t_meta = time.perf_counter()
    with init_empty_weights():
        empty = AutoModelForImageTextToText.from_config(cfg, trust_remote_code=True)
    scan = _scan_modules(empty)
    report.update(scan)
    report["meta_s"] = round(time.perf_counter() - t_meta, 1)
    print("suggested_lora_targets=", report["suggested_lora_targets"], flush=True)
    print(f"n_named_modules={report['n_named_modules']} meta_s={report['meta_s']}", flush=True)
    report["modules_ok"] = True
    del empty

    # --- B) real XPU load (may OOM; single rank only sees local-node tiles) ---
    nnodes = int(os.environ.get("PBS_NUM_NODES", os.environ.get("NNODES", "1")) or "1")
    do_load = os.environ.get("INKLING_PROBE_LOAD", "1") != "0"
    report["nnodes_env"] = nnodes
    report["load_attempted"] = do_load
    if do_load:
        print(
            f"loading weights device_map=auto bf16 "
            f"(nnodes_env={nnodes}; local xpu={report['xpu_count']}; "
            f"weight floor ~3 nodes / ~1894GB — expect OOM if local HBM < weights)…",
            flush=True,
        )
        t1 = time.perf_counter()
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            report["load_s"] = round(time.perf_counter() - t1, 1)
            report["load_ok"] = True
            print(f"model_loaded_s={report['load_s']}", flush=True)
            # refresh scan from real modules if load worked
            report.update(_scan_modules(model))
            del model
        except Exception as e:
            report["load_ok"] = False
            report["load_error"] = f"{type(e).__name__}: {e}"
            report["load_traceback"] = traceback.format_exc()[-2000:]
            print(f"LOAD_FAIL {report['load_error']}", flush=True)
    else:
        report["load_ok"] = None
        print("INKLING_PROBE_LOAD=0 — skip weight load", flush=True)

    report["ok"] = bool(report.get("modules_ok"))
    report["total_s"] = round(time.perf_counter() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print("PROBE_JSON=" + json.dumps({k: v for k, v in report.items() if k != "load_traceback"}), flush=True)
    print("PHASE1_PROBE_OK=1" if report["ok"] else "PHASE1_PROBE_OK=0", flush=True)
    if not report.get("load_ok"):
        raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
