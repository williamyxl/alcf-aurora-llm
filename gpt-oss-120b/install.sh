#!/bin/bash
# Install / refresh local co-built wheels into build-vllm-xpu/env (--no-deps).
# See README.md and build-vllm-xpu/VERSIONS.md. Do not pull CUDA torch from PyPI.
#
# Usage (from workdir):
#   bash install.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD="$ROOT/build-vllm-xpu"
ENV="$BUILD/env"
WHEELS="$BUILD/xiaoliyan"

source /lus/flare/projects/MOFA/xiaoliyan/software/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV"

echo "Installing local wheels from $WHEELS (force-reinstall --no-deps)"
for w in \
  torch-2.10.0a0+git449b176-cp312-cp312-linux_x86_64.whl \
  intel_extension_for_pytorch-2.10.10+gitd0f992f-cp312-cp312-linux_x86_64.whl \
  oneccl_bind_pt-2.8.0+xpu-cp312-cp312-linux_x86_64.whl \
  vllm_xpu_kernels-0.1.dev1+g4002cea90.d20260717-cp312-cp312-linux_x86_64.whl \
  vllm-0.1.dev1+g109b736b8.d20260717.xpu-cp312-cp312-linux_x86_64.whl
do
  pip install --force-reinstall --no-deps "$WHEELS/$w"
done

echo "NOTE: install Triton 3.6 (225cdbde) frameworks wheel separately; do NOT use self-built 3.8."
echo "NOTE: re-apply build-vllm-xpu/patches after any Triton/vLLM reinstall (see README.md)."
echo "Done. Activate with: conda activate $ENV"
