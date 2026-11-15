#!/usr/bin/env bash
# Download checkpoints for gpt-oss-120b. Edit/run only the lines you need (multi-hundred-GB pulls). Login node.
REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"; source "$REPO/common/env.sh"
D="$DATA/store/gpt-oss"
"$REPO/common/tools/hf_download.sh" openai/gpt-oss-120b            "$D/openai-gpt-oss-120b"                       # MXFP4 HF checkpoint (vLLM)  ~66 GB
"$REPO/common/tools/hf_download.sh" ggml-org/gpt-oss-120b-GGUF     "$D/ggml-org-gpt-oss-120b-GGUF" --include 'gpt-oss-120b-MXFP4*.gguf'   # llama.cpp
# optional BF16 (218 GB): see docs; produced from the MXFP4 checkpoint by dequantisation
mkdir -p "$MODELS/gpt-oss-120b"; ln -sfn "$D/openai-gpt-oss-120b" "$MODELS/gpt-oss-120b/mxfp4-hf"
ln -sfn "$D/ggml-org-gpt-oss-120b-GGUF/gpt-oss-120b-MXFP4.gguf" "$MODELS/gpt-oss-120b/mxfp4.gguf"
