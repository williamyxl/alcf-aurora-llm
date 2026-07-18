#!/usr/bin/env bash
# Download openai/gpt-oss-120b weights into the project models/ dir.
# Login node OK (I/O only). Does NOT submit PBS jobs or run inference.
#
# Usage:
#   bash download_model.sh
#
# Needs HF access for openai/gpt-oss-120b (huggingface-cli login if gated).

cd /lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b
mkdir -p models .cache/huggingface

module load frameworks/2025.3.1
conda activate /lus/flare/projects/MOFA/xiaoliyan/conda-env

export HF_HOME=/lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b/.cache/huggingface
export TMPDIR=/tmp

echo "Downloading openai/gpt-oss-120b -> models/openai-gpt-oss-120b"
echo "This can take a long time and use 100+ GB."

huggingface-cli download openai/gpt-oss-120b \
  --local-dir /lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b/models/openai-gpt-oss-120b

echo "Download complete."
ls /lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b/models/openai-gpt-oss-120b | head
