#!/usr/bin/env bash
# Build llama.cpp SYCL for Aurora PVC into $LLAMACPP_BUILDS/<name>/{llama.cpp,build}. Run on a COMPUTE node (qsub common/llamacpp/build.pbs).
#   NAME=gptoss REF=9d57ce4                 # gpt-oss: master snapshot of 2026-08-15
#   NAME=inkling PR=25731 REF=368b24c35     # inkling + nemotron + laguna archs (PR #25731 @ 368b24c35, 2026-08-12)
# Notes: oneAPI 2025.3.1 + icpx. RPC build for multi-node Inkling: add RPC=1 (-DGGML_RPC=ON).
set -eo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; source "$REPO/common/env.sh"; export_proxy
: "${NAME:?set NAME}"; ROOT="$LLAMACPP_BUILDS/$NAME"; SRC="$ROOT/llama.cpp"; BUILD="$ROOT/build"
module load oneapi/release/2025.3.1
mkdir -p "$ROOT"
[ -d "$SRC/.git" ] || git clone https://github.com/ggml-org/llama.cpp.git "$SRC"
cd "$SRC"
if [ -n "$PR" ]; then git fetch origin "pull/$PR/head:pr-$PR"; git checkout -f "pr-$PR"; fi
[ -n "$REF" ] && git checkout -f "$REF"
echo "llama.cpp @ $(git log -1 --format='%h %cd %s')"
rm -rf "$BUILD"
RPCFLAG=""; [ "$RPC" = 1 ] && RPCFLAG="-DGGML_RPC=ON"
cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DEVICE_ARCH=pvc \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON $RPCFLAG
cmake --build "$BUILD" -j "$(nproc)"
ls -lh "$BUILD/bin/llama-server" "$BUILD/bin/llama-completion" "$BUILD/bin/llama-batched-bench"
echo "BUILD_OK name=$NAME"
