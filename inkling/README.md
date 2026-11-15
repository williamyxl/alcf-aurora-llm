# Inkling (Thinking Machines, ~975B MoE / ~41B active)

llama.cpp only. vLLM does not work, for two reasons: only GGUF weights are local (vLLM 0.26.1 has no GGUF loader), and vLLM's Inkling model code (`vllm/models/inkling/nvidia`) is written for NVIDIA (CUTLASS / FA4 attention, NVLink all-reduce); the `amd` variant needs ROCm's `aiter`. There is no XPU branch, so a build would not fix it.

| Recipe | Result |
|---|---|
| `llamacpp/serve_12tile.pbs` : UD-IQ1_S, 12 tiles pure GPU, layer split | 6.6 tok/s decode single-stream; batched-bench 15.3 tok/s aggregate at 64 sequences, **17.1 at 128** |
| `llamacpp/serve_api.pbs` : llama-server over 12 tiles | OpenAI endpoint (untimed here) |
| `llamacpp/RPC_MULTINODE.md` | multi-node over ggml-rpc (build with `RPC=1`) |

Needs the `inkling` llama.cpp build (PR 25731). Setup: `bash inkling/download_gguf.sh` (UD-IQ1_S, ~250 GB).
`training/` is a July 2026 LoRA/FSDP smoke test that no longer reproduces (see its README).
