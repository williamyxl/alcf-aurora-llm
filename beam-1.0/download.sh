#!/usr/bin/env bash
# Download checkpoints for beam-1.0. Edit/run only the lines you need (multi-hundred-GB pulls). Login node.
REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"; source "$REPO/common/env.sh"
D="$DATA/store/beam-1.0"; mkdir -p "$MODELS/beam-1.0"
"$REPO/common/tools/hf_download.sh" reflection/Beam-1.0-501B-A23B "$D/bf16"                       # 934 GB BF16: does NOT fit one node (768 GB HBM)
ln -sfn "$D/bf16" "$MODELS/beam-1.0/bf16-hf"
# smaller: reflection/Beam-1.0-501B-A23B-FP8 (503 GB; FP8 is unreliable on PVC)
