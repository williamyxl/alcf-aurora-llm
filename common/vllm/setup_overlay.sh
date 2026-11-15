#!/usr/bin/env bash
# Create $OVERLAY (default data/overlay) with the two add-ons that sit on top of the read-only frameworks/2026.1.0 module:
#
#   py/       transformers 5.23.0 + tokenizers 0.23.3 + huggingface_hub 2.4.0   -> OVERLAY_TF=1
#   shim/     sitecustomize.py : None-rope (NoPE layer) compat for vLLM 0.26.1 + transformers>=5.22 (Beam, Muse)
#   kernels/  shadow of vllm_xpu_kernels with a rebuilt _moe_C that accepts top-k = 22 -> OVERLAY_KERNELS=1
#
#   common/vllm/setup_overlay.sh py          # login node (pip needs the proxy)
#   common/vllm/setup_overlay.sh kernels     # COMPUTE node (compiles C++ with icpx, ~10 min); qsub common/vllm/setup_overlay.pbs
set -eo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; source "$REPO/common/env.sh"
WHAT=${1:-all}
vllm_env
site_pkg="$FWROOT/lib/python3.12/site-packages"
setup_py() {
  export_proxy; mkdir -p "$OVERLAY/py" "$OVERLAY/shim"
  "$FWROOT/bin/pip" install --no-deps --target "$OVERLAY/py" "transformers==5.23.0" "tokenizers==0.23.3" "huggingface_hub==2.4.0"
  cp "$REPO/common/vllm/shim_sitecustomize.py" "$OVERLAY/shim/sitecustomize.py"
}
setup_kernels() {
  export_proxy
  local B="$OVERLAY/kernels_build" SRC="$OVERLAY/kernels_build/src" OBJ="$OVERLAY/kernels_build/obj" OUT="$OVERLAY/kernels/vllm_xpu_kernels"
  mkdir -p "$B" "$OBJ" "$OUT"
  [ -d "$SRC/.git" ] || git clone https://github.com/vllm-project/vllm-xpu-kernels "$SRC"
  cd "$SRC"; git checkout -q -f v0.1.11.1          # == a692986, the base of the module's vllm_xpu_kernels 0.1.11.2.dev0
  git apply "$REPO/common/vllm/kernels/topk22.patch"
  local TI TL PYINC CM DEV=${AOT_DEV:-pvc}
  TI=$("$FWROOT/bin/python" -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'include'))" 2>/dev/null | tail -1); TL=$(dirname "$TI")/lib
  PYINC=$("$FWROOT/bin/python" -c "import sysconfig;print(sysconfig.get_paths()['include'])"); CM=${CMPLR_ROOT:-/opt/aurora/26.181.0/oneapi/compiler/latest}
  local FLAGS="-fsycl -std=c++17 -O3 -DNDEBUG -fPIC -DTORCH_EXTENSION_NAME=_moe_C -DPy_LIMITED_API=0x03080000 -DVLLM_MOE_ENABLED -D_GLIBCXX_USE_CXX11_ABI=1 -I$SRC/csrc -I$TI -I$TI/torch/csrc/api/include -I$PYINC -I$CM/include -I$CM/include/sycl -I$CM/include/syclcompat"
  ls csrc/moe/*.cpp | xargs -P "${JOBS:-32}" -I{} sh -c 'f={}; icpx '"$FLAGS"' -fsycl-targets=spir64_gen -Xs "-device '"$DEV"'" -c $f -o '"$OBJ"'/$(basename $f .cpp).o'
  icpx -fsycl -shared -fsycl-targets=spir64_gen -Xs "-device $DEV" -fsycl-max-parallel-link-jobs=16 "$OBJ"/*.o -o "$OUT/_moe_C.abi3.so" \
    -L"$TL" -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu -Wl,-rpath,'$ORIGIN' -lze_loader
  # full-package shadow: Python imports vllm_xpu_kernels as one package, so link every other file from the module copy
  for f in "$site_pkg"/vllm_xpu_kernels/*; do b=$(basename "$f"); [ "$b" = "_moe_C.abi3.so" ] || ln -sf "$f" "$OUT/$b"; done
  echo "OK: $OUT/_moe_C.abi3.so"
}
case "$WHAT" in py) setup_py;; kernels) module load oneapi/release/2026.1.0 2>/dev/null || true; setup_kernels;; all) setup_py; module load oneapi/release/2026.1.0 2>/dev/null || true; setup_kernels;; *) echo "usage: $0 py|kernels|all"; exit 1;; esac
