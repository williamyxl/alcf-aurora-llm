#!/usr/bin/env bash
# Download openai/gpt-oss-120b weights into the project models/ dir.
# Login node OK (I/O only). Does NOT submit PBS jobs or run inference.
#
# Usage:
#   bash download_model.sh
#
# Needs HF access for openai/gpt-oss-120b (huggingface-cli login if gated).

cd /lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/gpt-oss-120b
mkdir -p models .cache/huggingface

# No `module load frameworks` (no usable gpt-oss path; also unneeded for a pure HF download).
# Use the aurora-llm env, which has huggingface_hub (the miniforge3 base does not).
source /lus/flare/projects/MatSciAI/xiaoliyan/miniforge3/etc/profile.d/conda.sh
conda activate /lus/flare/projects/MatSciAI/xiaoliyan/software/conda/envs/aurora-llm

export HF_HOME=/lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/gpt-oss-120b/.cache/huggingface
export TMPDIR=/tmp
# Aurora egress needs the site proxy.
export http_proxy=${http_proxy:-http://proxy.alcf.anl.gov:3128}
export https_proxy=${https_proxy:-http://proxy.alcf.anl.gov:3128}

echo "Downloading openai/gpt-oss-120b -> models/openai-gpt-oss-120b"
echo "This can take a long time and use 100+ GB."

if command -v hf >/dev/null 2>&1; then
  hf download openai/gpt-oss-120b \
    --local-dir /lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/gpt-oss-120b/models/openai-gpt-oss-120b
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download openai/gpt-oss-120b \
    --local-dir /lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/gpt-oss-120b/models/openai-gpt-oss-120b
else
  echo "ERROR: no hf / huggingface-cli in this env. Activate the build-vllm-xpu/env stack or 'pip install huggingface_hub'." >&2
  exit 1
fi

echo "Download complete."
ls /lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/gpt-oss-120b/models/openai-gpt-oss-120b | head
