#!/usr/bin/env bash
# Download checkpoints for muse-spark-open. Edit/run only the lines you need (multi-hundred-GB pulls). Login node.
REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"; source "$REPO/common/env.sh"
D="$DATA/store/muse-spark-open"; mkdir -p "$MODELS/muse-spark-open"
"$REPO/common/tools/hf_download.sh" unsloth/Muse-Spark-Open-816B-A42B "$D/bf16"                   # 1.5 TB BF16: does NOT fit one node (ungated mirror of meta-models/)
ln -sfn "$D/bf16" "$MODELS/muse-spark-open/bf16-hf"
