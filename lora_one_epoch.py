#!/usr/bin/env python
"""Phase 6: one-epoch LoRA/SFT smoke on gpt-oss-120b (XPU, no vLLM)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch


def main():
    import intel_extension_for_pytorch as ipex  # noqa: F401
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    workdir = Path("/lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b")
    model_path = workdir / "models" / "openai-gpt-oss-120b"
    out_dir = workdir / "checkpoints" / "lora-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_xpu = torch.xpu.device_count()
    print(f"xpu_count={n_xpu}", flush=True)
    if n_xpu < 1:
        raise RuntimeError("no XPU devices visible")

    device = "xpu"
    print(f"loading tokenizer from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tiny smoke corpus: enough rows for a real epoch, short enough for debug walltime.
    texts = [
        "Q: What is Type I IUPAC isotherm?\nA: Steep uptake at low pressure then a plateau; common for microporous materials.",
        "Q: Why does Cu-BTC CO2 uptake plateau near 1 bar?\nA: Micropore volume is finite; sites fill and further uptake slows.",
        "Q: Name one follow-up experiment for adsorption mechanism.\nA: Compare isotherms at multiple temperatures or probe beyond 1 bar.",
        "Q: What does a Langmuir-like plateau suggest?\nA: Saturation of adsorption sites rather than multilayer condensation.",
        "Q: Define micropore filling briefly.\nA: Adsorbate occupies narrow pores with strong host-guest interactions at low P.",
        "Q: Give one MOF characterization method.\nA: N2 or CO2 adsorption isotherms with BET/Langmuir analysis.",
        "Q: What is Cu-BTC also known as?\nA: HKUST-1, a copper paddle-wheel MOF with open metal sites.",
        "Q: Why use low-pressure isotherm shape?\nA: It distinguishes microporosity (Type I) from meso/macro (Types II–IV).",
    ]
    n_samples = len(texts)
    ds = Dataset.from_dict({"text": texts})

    print("loading model (mxfp4 dequantize→bf16 for train; LoRA on attn)", flush=True)
    t0 = time.perf_counter()
    # MXFP4 path is inference-only in transformers; dequantize for LoRA/SFT.
    from transformers import Mxfp4Config

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=Mxfp4Config(dequantize=True),
    )
    load_s = time.perf_counter() - t0
    print(f"model_loaded_s={load_s:.1f}", flush=True)

    # Prefer modules listed as non-quantized in mxfp4 config (attn).
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # SFTConfig subclasses TrainingArguments in recent TRL.
    # Default loss_type=chunked_nll patches lm_head.forward and breaks when that
    # forward is already a functools.partial under device_map="auto".
    train_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
        max_length=256,
        dataset_text_field="text",
        packing=False,
        remove_unused_columns=False,
        loss_type="nll",
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print("=== train start epochs=1 ===", flush=True)
    t1 = time.perf_counter()
    result = trainer.train()
    train_s = time.perf_counter() - t1
    print("=== train done ===", flush=True)

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = {
        "ok": True,
        "device": device,
        "epochs": 1,
        "n_samples": n_samples,
        "n_steps": int(result.global_step),
        "train_loss": float(result.training_loss) if result.training_loss is not None else None,
        "train_s": train_s,
        "load_s": load_s,
        "xpu_count": n_xpu,
        "adapter_path": str(adapter_dir),
        "target_modules": ["q_proj", "v_proj"],
        "lora_r": 8,
    }
    print("TRAIN_JSON=" + json.dumps(metrics, separators=(",", ":")), flush=True)
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
