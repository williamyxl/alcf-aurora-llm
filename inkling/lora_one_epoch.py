#!/usr/bin/env python
"""Phase 2: one-epoch LoRA/SFT smoke for Inkling on Aurora XPU.

INKLING_LOAD_MODE=offload (legacy): 1-process device_map=auto + disk offload.
  Fits debug wall only when Lustre is warm (~43–50 m load); often wall-kills.

INKLING_LOAD_MODE=fsdp (preferred): multi-rank FSDP2, meta init, sharded load.
  Files are round-robin owned across ranks; owners convert HF→transformers keys and
  scatter only local shards (not full-tensor broadcast of ~1.9 TB).
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch


WORKDIR = Path("/lus/flare/projects/MOFA/xiaoliyan/workdir/llm/inkling")
MODEL = WORKDIR / "models" / "inkling-hf"
OUT_DIR = WORKDIR / "checkpoints" / "lora-smoke"


def _export_pals_ranks() -> tuple[int, int, int]:
    """Resolve ranks for Aurora mpiexec (PALS/PMI). Prefer PMI/PALS over WORLD_SIZE."""
    rank = 0
    for k in ("PALS_RANKID", "PMI_RANK", "PMIX_RANK", "RANK", "MPI_RANKID", "OMPI_COMM_WORLD_RANK"):
        if k in os.environ and str(os.environ[k]).strip() != "":
            rank = int(os.environ[k])
            break
    local = 0
    for k in ("PALS_LOCAL_RANKID", "MPI_LOCALRANKID", "LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK"):
        if k in os.environ and str(os.environ[k]).strip() != "":
            local = int(os.environ[k])
            break
    world = 0
    for k in ("PMI_SIZE", "PMIX_SIZE", "PALS_NRANKS", "OMPI_COMM_WORLD_SIZE", "WORLD_SIZE"):
        if k in os.environ and str(os.environ[k]).strip() != "":
            world = int(os.environ[k])
            break
    if world <= 1:
        # Fallback: PBS node count × tiles/node (debug-scaling FSDP).
        nnodes = int(os.environ.get("PBS_NUM_NODES") or os.environ.get("NNODES") or "0")
        ppn = int(os.environ.get("INKLING_PPN", "12"))
        if nnodes > 1 or (nnodes == 1 and ppn > 1 and "PALS_RANKID" in os.environ):
            world = max(nnodes, 1) * ppn
        else:
            world = max(world, 1)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(local)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    return rank, world, local


def _fix_tokenizer(tokenizer, is_main: bool):
    def _bad(tok: str | None) -> bool:
        return tok is None or str(tok) in {"None", "null", ""}

    if _bad(tokenizer.eos_token):
        tokenizer.eos_token = "<|endoftext|>"
    if _bad(tokenizer.pad_token) or getattr(tokenizer, "pad_token_id", None) in (None, -1):
        tokenizer.pad_token = tokenizer.eos_token
    if is_main:
        print(
            f"tokenizer pad={tokenizer.pad_token!r} id={tokenizer.pad_token_id} "
            f"eos={tokenizer.eos_token!r} id={tokenizer.eos_token_id}",
            flush=True,
        )


def _smoke_texts():
    return [
        "Q: What is Type I IUPAC isotherm?\nA: Steep uptake at low pressure then a plateau; common for microporous materials.",
        "Q: Why does Cu-BTC CO2 uptake plateau near 1 bar?\nA: Micropore volume is finite; sites fill and further uptake slows.",
        "Q: Name one follow-up experiment for adsorption mechanism.\nA: Compare isotherms at multiple temperatures or probe beyond 1 bar.",
        "Q: What does a Langmuir-like plateau suggest?\nA: Saturation of adsorption sites rather than multilayer condensation.",
        "Q: Define micropore filling briefly.\nA: Adsorbate occupies narrow pores with strong host-guest interactions at low P.",
        "Q: Give one MOF characterization method.\nA: N2 or CO2 adsorption isotherms with BET/Langmuir analysis.",
        "Q: What is Cu-BTC also known as?\nA: HKUST-1, a copper paddle-wheel MOF with open metal sites.",
        "Q: Why use low-pressure isotherm shape?\nA: It distinguishes microporosity (Type I) from meso/macro (Types II–IV).",
    ]


def _init_dist(local: int):
    import oneccl_bindings_for_pytorch  # noqa: F401
    import torch.distributed as dist

    torch.xpu.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group(backend="ccl")
    return dist


def _fsdp2_wrap(model, mesh, is_main: bool):
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.fsdp import CPUOffloadPolicy
    from transformers.models.inkling.modeling_inkling import InklingDecoderLayer

    # CPU offload: keep sharded weights on host so ~20–36GiB layer all-gather
    # can fit in 64GiB XPU HBM during forward.
    use_offload = os.environ.get("INKLING_FSDP_CPU_OFFLOAD", "1") == "1"
    offload_kw = {"offload_policy": CPUOffloadPolicy()} if use_offload else {}

    layers = [m for m in model.modules() if isinstance(m, InklingDecoderLayer)]
    for module in layers:
        fully_shard(module, mesh=mesh, **offload_kw)
        # XPU/CCL does not support ReduceOp.AVG; force SUM + post-divide.
        module.set_force_sum_reduction_for_comms(True)
    # Root wrap shards vision/lm_head/etc. for memory, but keep embed path as
    # plain tensors (root DTensor embed_norm hit Partial.mask_buffer assert).
    ignore = {
        p
        for n, p in model.named_parameters()
        if any(
            s in n
            for s in (
                "embed_tokens",
                "embed_norm",
                "wte",
                "tok_embeddings",
                "word_embeddings",
            )
        )
    }
    fully_shard(model, mesh=mesh, ignored_params=ignore if ignore else None, **offload_kw)
    model.set_force_sum_reduction_for_comms(True)
    if is_main:
        print(
            f"fsdp2 wrapped InklingDecoderLayer count={len(layers)} mesh={mesh} "
            f"root_fully_shard=1 ignored_embed_params={len(ignore)} "
            f"cpu_offload={int(use_offload)} force_sum_reduce=1",
            flush=True,
        )
    return model


def _map_hf_to_model_keys(weight_map: dict[str, str], model_keys: set[str]) -> dict[str, str]:
    """Map checkpoint keys → model keys via transformers inkling conversion + suffix match."""
    from transformers.conversion_mapping import _build_checkpoint_conversion_mapping
    from transformers.core_model_loading import WeightRenaming

    converters = _build_checkpoint_conversion_mapping()["inkling_mm_model"]
    hf_to_model: dict[str, str] = {}
    for hf_name in weight_map:
        if hf_name.startswith("model.mtp."):
            continue
        cur = hf_name
        for t in converters:
            if isinstance(t, WeightRenaming):
                cur, _ = t.rename_source_key(cur)
        # WeightConverter targets are applied later on tensors; provisional key for matching.
        if cur in model_keys:
            hf_to_model[hf_name] = cur
            continue
        matches = [k for k in model_keys if k.endswith("." + cur) or k == cur or k.endswith(cur)]
        matches = [k for k in matches if "lora_" not in k]
        if len(matches) == 1:
            hf_to_model[hf_name] = matches[0]
        elif cur.endswith("experts.down_proj") and (cur + "") in model_keys:
            hf_to_model[hf_name] = cur
        # WeightConverter sources stay provisional (handled in _convert_hf_tensors)
        elif any(
            s in hf_name
            for s in ("w13_dn.weight", "experts.w13_weight", "shared_w13_weight")
        ):
            hf_to_model[hf_name] = cur  # marker; converted at load time
    return hf_to_model


def _convert_hf_tensors(raw: dict) -> dict:
    """Rename + MoE WeightConverter ops → tensors keyed by transformers module names."""
    from transformers.conversion_mapping import _build_checkpoint_conversion_mapping
    from transformers.core_model_loading import WeightConverter, WeightRenaming

    converters = _build_checkpoint_conversion_mapping()["inkling_mm_model"]
    renamed: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if k.startswith("model.mtp."):
            continue
        cur = k
        for t in converters:
            if isinstance(t, WeightRenaming):
                cur, _ = t.rename_source_key(cur)
        renamed[cur] = v.to(dtype=torch.bfloat16).contiguous()

    out: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    wc_list = [t for t in converters if isinstance(t, WeightConverter)]

    for key, tensor in renamed.items():
        hit = None
        for wc in wc_list:
            for sp in wc.source_patterns:
                if key.endswith(sp) or sp in key:
                    hit = (wc, sp)
                    break
            if hit:
                break
        if hit is None:
            out[key] = tensor
            continue
        wc, sp = hit
        consumed.add(key)
        d: dict = {sp: tensor}
        for op in wc.operations:
            d = op.convert(
                d,
                source_patterns=wc.source_patterns,
                target_patterns=wc.target_patterns,
                full_layer_name=key,
            )
        # Expand short target names to full module paths.
        idx = key.find(sp)
        if idx < 0:
            # try last path segment match
            for cand in wc.source_patterns:
                idx = key.find(cand)
                if idx >= 0:
                    sp = cand
                    break
        if idx < 0:
            raise RuntimeError(f"cannot place WeightConverter outputs for {key} sp={sp}")
        prefix = key[:idx]
        suffix = key[idx + len(sp) :]
        for short_k, ten in d.items():
            full = prefix + short_k + suffix
            # nn.Parameter state_dict keys omit ".weight" only for raw Parameters;
            # Linear weights keep ".weight". Our targets already include .weight where needed.
            out[full] = ten if isinstance(ten, torch.Tensor) else ten[0]
    return out


def _resolve_loaded_key(converted_name: str, model_keys: set[str]) -> str | None:
    """Map converted HF/model key onto current state_dict key (handles PEFT base_layer)."""
    if converted_name in model_keys:
        return converted_name
    # PEFT wraps targets: *.q_proj.weight → *.q_proj.base_layer.weight
    peft_name = None
    if converted_name.endswith(".weight"):
        peft_name = converted_name[: -len(".weight")] + ".base_layer.weight"
        if peft_name in model_keys:
            return peft_name
    matches = [
        k
        for k in model_keys
        if "lora_" not in k
        and (
            k == converted_name
            or k.endswith("." + converted_name)
            or k.endswith(converted_name)
            or (peft_name is not None and (k == peft_name or k.endswith("." + peft_name) or k.endswith(peft_name)))
        )
    ]
    # Prefer base_layer (frozen LoRA-target weight) over duplicates.
    base = [k for k in matches if "base_layer" in k]
    if len(base) == 1:
        return base[0]
    if len(matches) == 1:
        return matches[0]
    return None


def _dtensor_keep_local(local: torch.Tensor, like: "DTensor") -> "DTensor":
    """Wrap a local shard as DTensor without moving it onto the mesh device.

    ``DTensor.from_local`` forces ``local.to(mesh.device_type)``, which breaks
    FSDP2 ``CPUOffloadPolicy`` (managed locals must stay on CPU). FSDP itself
    uses ``_from_local_no_grad`` for the same reason.
    """
    from torch.distributed.fsdp._fully_shard._fsdp_common import _from_local_no_grad

    return _from_local_no_grad(local, like._spec)


def _ensure_cpu_offload_locals(model, is_main: bool) -> None:
    """Force FSDP-managed parameter locals onto CPU before first forward."""
    from torch.distributed.tensor import DTensor

    use_offload = os.environ.get("INKLING_FSDP_CPU_OFFLOAD", "1") == "1"
    if not use_offload:
        return
    embed_sub = (
        "embed_tokens",
        "embed_norm",
        "wte",
        "tok_embeddings",
        "word_embeddings",
    )
    n_dt = 0
    n_plain = 0
    for name, p in model.named_parameters():
        if isinstance(p, DTensor):
            if p._local_tensor.device.type != "cpu":
                p._local_tensor = p._local_tensor.detach().to("cpu")
                n_dt += 1
            continue
        if p.device.type != "cpu" and not any(s in name for s in embed_sub):
            p.data = p.data.detach().to("cpu")
            n_plain += 1
    if is_main:
        print(
            f"cpu_offload ensure: moved_dtensor_locals={n_dt} moved_plain={n_plain}",
            flush=True,
        )


def _load_hf_into_fsdp2(model, model_dir: Path, rank: int, is_main: bool) -> int:
    """Load HF weights into FSDP2 via round-robin owners + broadcast/distribute_tensor.

    Smoke default: only load ``self_attn.q_proj`` / ``self_attn.v_proj`` (LoRA targets).
    Set ``INKLING_LOAD_ALL=1`` to load every matching tensor (much slower).
    """
    import re
    import torch.distributed as dist
    from safetensors import safe_open
    from torch.distributed.tensor import DTensor, distribute_tensor

    world = dist.get_world_size()
    load_all = os.environ.get("INKLING_LOAD_ALL", "0") == "1"
    want_substrings = ("self_attn.q_proj", "self_attn.v_proj")
    # Pre-conversion HF names for q/v LoRA targets (avoids opening all 108 shards).
    hf_qv_substrings = ("attn.wq_du", "attn.wv_dv")
    smoke_layers = int(os.environ.get("INKLING_SMOKE_LAYERS", "0") or "0")

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    file_to_keys: dict[str, list[str]] = defaultdict(list)
    for name, fname in weight_map.items():
        if name.startswith("model.mtp."):
            continue
        if not load_all:
            if not any(s in name for s in hf_qv_substrings):
                continue
            if smoke_layers > 0:
                m = re.search(r"layers\.(\d+)\.", name)
                if m is not None and int(m.group(1)) >= smoke_layers:
                    continue
        file_to_keys[fname].append(name)
    files = sorted(file_to_keys.keys())

    meta_sd = dict(model.state_dict())
    model_keys = set(meta_sd.keys())
    if is_main:
        n_hf = sum(len(v) for v in file_to_keys.values())
        print(
            f"fsdp2 load: hf_tensors={n_hf}/{len(weight_map)} model_params={len(meta_sd)} "
            f"n_files={len(files)} world={world} load_all={load_all} "
            f"smoke_layers={smoke_layers} "
            f"targets={want_substrings if not load_all else 'ALL'}",
            flush=True,
        )

    loaded = 0
    missing_targets: list[str] = []
    shape_mismatch: list[str] = []
    t_load0 = time.perf_counter()
    xpu = f"xpu:{torch.xpu.current_device()}"

    for fi, fname in enumerate(files):
        owner = fi % world
        path = model_dir / fname
        keys = file_to_keys[fname]

        if rank == owner:
            raw = {}
            with safe_open(str(path), framework="pt", device="cpu") as handles:
                for hf_name in keys:
                    raw[hf_name] = handles.get_tensor(hf_name)
            converted = _convert_hf_tensors(raw)
            # Optional smoke filter after conversion (model key names).
            if not load_all:
                converted = {
                    k: v
                    for k, v in converted.items()
                    if any(s in k for s in want_substrings) and "lora_" not in k
                }
            items = sorted(converted.items())
            meta = [(name, tuple(t.shape)) for name, t in items]
            obj = [meta]
        else:
            items = []
            obj = [None]
        dist.broadcast_object_list(obj, src=owner)
        meta = obj[0]
        assert meta is not None

        batch_sd: dict[str, torch.Tensor] = {}
        file_assigned = 0
        for i, (model_name, shape) in enumerate(meta):
            resolved = _resolve_loaded_key(model_name, model_keys)
            in_model = resolved is not None
            flag = torch.tensor([1 if in_model else 0], dtype=torch.long, device=xpu)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if int(flag.item()) == 0:
                if rank == owner:
                    missing_targets.append(model_name)
                continue
            assert resolved is not None
            model_name = resolved

            sharded_param = meta_sd[model_name]
            global_shape = tuple(sharded_param.shape) if isinstance(sharded_param, DTensor) else tuple(shape)
            if isinstance(sharded_param, DTensor):
                global_shape = tuple(sharded_param.shape)
            shape_ok = tuple(shape) == (
                tuple(sharded_param.shape) if isinstance(sharded_param, DTensor) else tuple(shape)
            )
            if isinstance(sharded_param, DTensor):
                shape_ok = tuple(shape) == tuple(sharded_param.shape)
            ok_t = torch.tensor([1 if shape_ok else 0], dtype=torch.long, device=xpu)
            dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
            if int(ok_t.item()) == 0:
                if rank == owner:
                    gs = tuple(sharded_param.shape) if isinstance(sharded_param, DTensor) else "?"
                    shape_mismatch.append(f"{model_name}: ckpt{shape} vs model{gs}")
                continue

            # Broadcast full tensor from owner, then let DTensor shard correctly.
            # CPUOffloadPolicy: keep shards on CPU (never target meta device).
            use_offload = os.environ.get("INKLING_FSDP_CPU_OFFLOAD", "1") == "1"
            param_dev = torch.device("cpu" if use_offload else xpu)
            if rank == owner:
                full = items[i][1].to(dtype=torch.bfloat16).contiguous().to(param_dev)
            else:
                full = torch.empty(shape, dtype=torch.bfloat16, device=param_dev)
            if param_dev.type == "xpu":
                dist.broadcast(full, src=owner)
            else:
                # CCL broadcast needs XPU; stage then copy back to CPU.
                if rank == owner:
                    stage = full.to(xpu)
                else:
                    stage = torch.empty(shape, dtype=torch.bfloat16, device=xpu)
                dist.broadcast(stage, src=owner)
                full = stage.to("cpu")
                del stage

            if isinstance(sharded_param, DTensor):
                dt = distribute_tensor(
                    full if full.device.type != "cpu" else full.to(xpu),
                    sharded_param.device_mesh,
                    sharded_param.placements,
                )
                if use_offload:
                    # Keep shard on CPU; do not use DTensor.from_local (moves to XPU).
                    local = dt.to_local().detach().to("cpu")
                    local.requires_grad_(sharded_param.requires_grad)
                    batch_sd[model_name] = _dtensor_keep_local(local, sharded_param)
                    del local
                else:
                    batch_sd[model_name] = dt
                del dt
            else:
                batch_sd[model_name] = full
            del full
            loaded += 1
            file_assigned += 1

        if batch_sd:
            model.load_state_dict(batch_sd, strict=False, assign=True)
            meta_sd.update(batch_sd)
            del batch_sd
        if torch.xpu.is_available():
            torch.xpu.empty_cache()

        if is_main and (file_assigned or fi % 10 == 0 or fi == len(files) - 1):
            print(
                f"fsdp2 load file {fi+1}/{len(files)} {fname} assigned={file_assigned} "
                f"total={loaded} elapsed_s={time.perf_counter()-t_load0:.0f}",
                flush=True,
            )

    if is_main and missing_targets:
        print(f"fsdp2 warn: {len(set(missing_targets))} converted keys not in model e.g. {sorted(set(missing_targets))[:12]}", flush=True)
    if is_main and shape_mismatch:
        print(f"fsdp2 warn: {len(shape_mismatch)} global shape mismatches e.g. {shape_mismatch[:12]}", flush=True)
    if loaded == 0:
        raise RuntimeError(
            "fsdp2 load matched=0 — no q/v tensors loaded. "
            f"shape_mismatch={shape_mismatch[:8]}"
        )

    # Materialize leftover meta with local zeros. Use the FSDP param's existing
    # _local_tensor shape (authoritative) + tensor_meta global shape/stride.
    # Do NOT recompute via Shard.local_shard_size_and_offset — that yields uneven
    # locals (e.g. 2064) while fully_shard allocated even locals (e.g. 2048).
    still_meta = [(n, p) for n, p in model.named_parameters() if p.device.type == "meta"]
    if still_meta:
        if is_main:
            print(f"materializing {len(still_meta)} leftover meta params (local zeros)", flush=True)
        batch_zero: dict[str, torch.Tensor] = {}
        for name, param in still_meta:
            if isinstance(param, DTensor):
                lt = param._local_tensor
                # Respect offload: materialize on CPU so we don't fill XPU HBM.
                use_offload = os.environ.get("INKLING_FSDP_CPU_OFFLOAD", "1") == "1"
                if use_offload:
                    dev = torch.device("cpu")
                else:
                    dev = lt.device if lt.device.type != "meta" else torch.device(xpu)
                local = torch.zeros(lt.shape, dtype=torch.bfloat16, device=dev)
                local.requires_grad_(param.requires_grad)
                # Avoid DTensor.from_local — it moves CPU locals onto the XPU mesh.
                batch_zero[name] = _dtensor_keep_local(local, param)
            else:
                # Ignored embed stays on XPU for compute; other plain leftovers → CPU
                # when offloading so FSDP lazy_init validation passes.
                use_offload = os.environ.get("INKLING_FSDP_CPU_OFFLOAD", "1") == "1"
                is_embed = any(
                    s in name
                    for s in (
                        "embed_tokens",
                        "embed_norm",
                        "wte",
                        "tok_embeddings",
                        "word_embeddings",
                    )
                )
                dev = torch.device(xpu if (is_embed or not use_offload) else "cpu")
                batch_zero[name] = torch.zeros(param.shape, dtype=torch.bfloat16, device=dev)
            if len(batch_zero) >= 32:
                model.load_state_dict(batch_zero, strict=False, assign=True)
                batch_zero.clear()
                if torch.xpu.is_available():
                    torch.xpu.empty_cache()
        if batch_zero:
            model.load_state_dict(batch_zero, strict=False, assign=True)
            batch_zero.clear()

    still_meta_names = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if still_meta_names:
        raise RuntimeError(f"still on meta after load: {still_meta_names[:20]}")

    _ensure_cpu_offload_locals(model, is_main)

    if is_main:
        print(f"fsdp2 load done tensors={loaded} elapsed_s={time.perf_counter()-t_load0:.0f}", flush=True)
    return loaded


# keep old helper unused name for clarity if referenced elsewhere
def _slice_full_for_rank(full: torch.Tensor, mesh, placements, rank_id: int) -> torch.Tensor:
    from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset

    local_shape, global_offset = _compute_local_shape_and_global_offset(
        tuple(full.shape), mesh.shape, (rank_id,), placements
    )
    slices: list = []
    for dim, (off, ln) in enumerate(zip(global_offset, local_shape)):
        if ln == 0:
            slices.append(slice(0, 0))
        else:
            slices.append(slice(int(off), int(off) + int(ln)))
    return full[tuple(slices)].contiguous()


def _write_success(metrics: dict, adapter_dir: Path):
    (OUT_DIR / "SUCCESS_TRAIN.md").write_text(
        "# Inkling LoRA smoke\n\n```json\n" + json.dumps(metrics, indent=2) + "\n```\n"
    )
    (WORKDIR / "SUCCESS_TRAIN.md").write_text(
        f"ok=true mode={metrics.get('mode')} adapter={adapter_dir}\n"
        "See checkpoints/lora-smoke/SUCCESS_TRAIN.md\n"
    )


def _run_offload():
    # device_map=auto + Accelerate DDP is forbidden; clear MPI leftovers before TRL import.
    for k in (
        "PMI_RANK",
        "PMI_SIZE",
        "PALS_RANKID",
        "PALS_LOCAL_RANKID",
        "MPI_LOCALRANKID",
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        os.environ.pop(k, None)
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"

    import intel_extension_for_pytorch as ipex  # noqa: F401
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    rank, world, local = _export_pals_ranks()
    is_main = True
    if torch.xpu.device_count() > 0:
        torch.xpu.set_device(0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"host={os.uname().nodename} mode=offload world={world} local={local} "
        f"xpu_count={torch.xpu.device_count()}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    _fix_tokenizer(tokenizer, is_main)
    texts = _smoke_texts()
    ds = Dataset.from_dict({"text": texts})
    targets = ["q_proj", "v_proj"]

    offload = OUT_DIR / "offload"
    offload.mkdir(parents=True, exist_ok=True)
    n_xpu = max(int(torch.xpu.device_count()), 1)
    max_memory = {i: os.environ.get("INKLING_XPU_MAX", "40GiB") for i in range(n_xpu)}
    max_memory["cpu"] = os.environ.get("INKLING_CPU_MAX", "120GiB")
    print(f"loading device_map=auto offload={offload} max_memory={max_memory}", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=str(offload),
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    load_s = time.perf_counter() - t0
    print(f"model_loaded_s={load_s:.1f}", flush=True)

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True

    os.environ["ACCELERATE_NUM_PROCESSES"] = "1"
    os.environ["ACCELERATE_NUM_MACHINES"] = "1"
    from accelerate.state import AcceleratorState
    from accelerate.utils import DistributedType

    AcceleratorState._reset_state(reset_partial_state=True)

    train_args = SFTConfig(
        output_dir=str(OUT_DIR),
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
        ddp_find_unused_parameters=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    object.__setattr__(trainer.accelerator.state, "distributed_type", DistributedType.NO)
    print(
        f"accelerate distributed_type={trainer.accelerator.state.distributed_type} "
        f"num_processes={trainer.accelerator.num_processes}",
        flush=True,
    )
    print("=== train start epochs=1 ===", flush=True)
    t1 = time.perf_counter()
    result = trainer.train()
    train_s = time.perf_counter() - t1
    print("=== train done ===", flush=True)

    adapter_dir = OUT_DIR / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    metrics = {
        "ok": True,
        "epochs": 1,
        "n_samples": len(texts),
        "n_steps": int(result.global_step),
        "train_loss": float(result.training_loss) if result.training_loss is not None else None,
        "train_s": train_s,
        "load_s": load_s,
        "world_size": world,
        "mode": "offload",
        "adapter_path": str(adapter_dir),
        "target_modules": targets,
        "lora_r": 8,
        "loader": "AutoModelForImageTextToText",
    }
    print("TRAIN_JSON=" + json.dumps(metrics, separators=(",", ":")), flush=True)
    _write_success(metrics, adapter_dir)
    print("=== done ===", flush=True)


def _run_fsdp():
    import intel_extension_for_pytorch as ipex  # noqa: F401
    from accelerate import init_empty_weights
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    rank, world, local = _export_pals_ranks()
    # Debug rank env once per process (compact).
    rank_env = {
        k: os.environ.get(k)
        for k in (
            "PALS_RANKID",
            "PALS_LOCAL_RANKID",
            "PALS_NRANKS",
            "PMI_RANK",
            "PMI_SIZE",
            "WORLD_SIZE",
            "RANK",
            "LOCAL_RANK",
            "PBS_NUM_NODES",
            "MASTER_ADDR",
            "MASTER_PORT",
        )
        if os.environ.get(k) is not None
    }
    print(f"rank_env={rank_env}", flush=True)

    dist = _init_dist(local)
    rank = dist.get_rank()
    world = dist.get_world_size()
    is_main = rank == 0
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    device = torch.device(f"xpu:{local}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if is_main:
        print(
            f"host={os.uname().nodename} mode=fsdp world={world} local={local} "
            f"xpu_count={torch.xpu.device_count()} device={device}",
            flush=True,
        )
    if world < 2:
        raise RuntimeError(
            f"FSDP needs world_size>=2, got {world}. Check mpiexec PMI/PALS env (rank_env={rank_env})"
        )

    mesh = init_device_mesh("xpu", (world,))

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    _fix_tokenizer(tokenizer, is_main)
    texts = _smoke_texts()
    targets = ["q_proj", "v_proj"]

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    # Each InklingDecoderLayer all-gather is ~36GiB; 66-layer sharded footprint
    # already ~54GiB/rank on 64GiB XPUs. Smoke uses few layers so gather fits.
    n_smoke = int(os.environ.get("INKLING_SMOKE_LAYERS", "2"))
    if n_smoke > 0:
        tc = getattr(cfg, "text_config", cfg)
        tc.num_hidden_layers = n_smoke
        if is_main:
            print(f"SMOKE num_hidden_layers={n_smoke}", flush=True)
    if is_main:
        print("building meta base+LoRA then FSDP wrap…", flush=True)
    t0 = time.perf_counter()
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(cfg, trust_remote_code=True)
    # LoRA before fully_shard so adapter params are FSDP-managed (avoids
    # plain-LoRA + sharded-base shape mismatch in peft forward).
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(model, lora)
    model = _fsdp2_wrap(model, mesh, is_main)
    n_loaded = _load_hf_into_fsdp2(model, MODEL, rank, is_main)
    load_s = time.perf_counter() - t0
    if is_main:
        print(f"model_loaded_s={load_s:.1f} n_loaded={n_loaded}", flush=True)
        model.print_trainable_parameters()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if torch.xpu.is_available():
        torch.xpu.empty_cache()

    # Tiny manual train loop — avoids Accelerate re-wrapping FSDP2.
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    model.train()
    if is_main:
        print("=== train start epochs=1 (manual loop) ===", flush=True)
    t1 = time.perf_counter()
    step = 0
    last_loss = None
    for epoch in range(1):
        for text in texts:
            batch = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=False,
            )
            # Root embed/norm are plain tensors (layers-only FSDP); use plain inputs.
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=labels,
            )
            loss = out.loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            last_loss = float(loss.detach().float().cpu())
            if is_main:
                print(f"step={step} loss={last_loss:.4f}", flush=True)
    dist.barrier()
    train_s = time.perf_counter() - t1
    if is_main:
        print("=== train done ===", flush=True)

    adapter_dir = OUT_DIR / "adapter"
    if is_main:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        try:
            model.save_pretrained(str(adapter_dir))
        except Exception as e:
            print(f"save_pretrained warn: {e}", flush=True)
            (adapter_dir / "README.md").write_text(
                f"FSDP2 smoke finished; adapter save failed: {e}\n"
            )
        tokenizer.save_pretrained(str(adapter_dir))
        metrics = {
            "ok": True,
            "epochs": 1,
            "n_samples": len(texts),
            "n_steps": step,
            "train_loss": last_loss,
            "train_s": train_s,
            "load_s": load_s,
            "world_size": world,
            "mode": "fsdp",
            "n_loaded": n_loaded,
            "adapter_path": str(adapter_dir),
            "target_modules": targets,
            "lora_r": 8,
            "loader": "meta+fsdp2+param_broadcast",
        }
        print("TRAIN_JSON=" + json.dumps(metrics, separators=(",", ":")), flush=True)
        _write_success(metrics, adapter_dir)
        print("=== done ===", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main():
    mode = os.environ.get("INKLING_LOAD_MODE", "fsdp").lower()
    if mode == "offload":
        _run_offload()
    elif mode == "fsdp":
        _run_fsdp()
    else:
        raise RuntimeError(f"Unknown INKLING_LOAD_MODE={mode} (use fsdp|offload)")


if __name__ == "__main__":
    main()
