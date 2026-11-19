#!/usr/bin/env bash
# llama.cpp GGUF checkpoint. Login node.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; source "$REPO_ROOT/common/env.sh"
"$REPO_ROOT/common/tools/hf_download.sh" unsloth/Nemotron-3-5-Super-VL-8B-8B-gguf "$DATA/store/nemotron-3-5-super-vl/gguf"
mkdir -p "$MODELS/nemotron-3-5-super-vl"; ln -sfn "$DATA/store/nemotron-3-5-super-vl/gguf" "$MODELS/nemotron-3-5-super-vl/gguf-hf"
