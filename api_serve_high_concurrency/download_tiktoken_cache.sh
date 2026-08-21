#!/bin/bash
# Pre-cache the openai_harmony o200k tiktoken vocab that gpt-oss requires at load time.
# Run ONCE on a login node (has internet via proxy). Compute nodes can then load it offline.
# Must use the frameworks vLLM python (same stack as the serve job).

export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
GPTOSS=/lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/gpt-oss-120b
export TIKTOKEN_RS_CACHE_DIR=${TIKTOKEN_RS_CACHE_DIR:-$GPTOSS/.tiktoken_cache}
mkdir -p "$TIKTOKEN_RS_CACHE_DIR"

module use /opt/aurora/26.26.0/frameworks/modulefiles && module load frameworks
FWPY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python

echo "Downloading harmony vocab -> $TIKTOKEN_RS_CACHE_DIR"
"$FWPY" -c "
import openai_harmony as oh
oh.load_harmony_encoding(oh.HarmonyEncodingName.HARMONY_GPT_OSS)
print('HARMONY_CACHE_OK')
"
echo "Cached files:"; ls -lh "$TIKTOKEN_RS_CACHE_DIR"
