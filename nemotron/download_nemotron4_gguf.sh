#!/bin/bash
# Download Nemotron-4-340B-Instruct GGUF (dense NemotronForCausalLM) from mradermacher (multipart
# <name>.gguf.partNofM byte chunks) and assemble into one .gguf.
#
# Storage: MatSciAI project (IQC project quota is 1MB and exceeded -> unusable). To stay within the
# soft quota, assemble by STREAMING: download one part, append it to the output, delete the part,
# repeat -> peak extra usage ~= one part (tens of GB), not 2x the model.
# Resumable at part granularity via a .done marker.
#
#   bash download_nemotron4_gguf.sh                 # i1-Q4_K_M (~210 GB, default)
#   QUANT=i1-IQ4_XS bash download_nemotron4_gguf.sh # ~183 GB
#   QUANT=i1-IQ3_M  bash download_nemotron4_gguf.sh # ~155 GB

set -euo pipefail
NEMO=/lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/nemotron
CKPT=${CKPT:-$NEMO/models/gguf-n4}         # on MatSciAI; override with CKPT=<dir> if a big-quota dir exists
export REPO=${REPO:-mradermacher/Nemotron-4-340B-Instruct-hf-i1-GGUF}
export QUANT=${QUANT:-i1-Q4_K_M}
export DL_DIR=${DL_DIR:-$CKPT/dl}
export HF_HOME=${HF_HOME:-$NEMO/.cache/huggingface}
export http_proxy=${http_proxy:-http://proxy.alcf.anl.gov:3128}
export https_proxy=${https_proxy:-http://proxy.alcf.anl.gov:3128}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}   # xet errors on these multipart repos
export HF_XET_HIGH_PERFORMANCE=0

BASE="Nemotron-4-340B-Instruct-hf.${QUANT}"
ASSEMBLED="$CKPT/${BASE}.gguf"
DONE="$CKPT/.${BASE}.assembled_parts"     # records how many parts already appended
mkdir -p "$DL_DIR" "$CKPT" "$HF_HOME"

if [ -f /lus/flare/projects/MatSciAI/xiaoliyan/software/conda/envs/aurora-llm/bin/hf ]; then
  source /lus/flare/projects/MatSciAI/xiaoliyan/miniforge3/etc/profile.d/conda.sh
  conda activate /lus/flare/projects/MatSciAI/xiaoliyan/software/conda/envs/aurora-llm
fi

echo "host=$(hostname) date=$(date -Is) REPO=$REPO QUANT=$QUANT CKPT=$CKPT"
df -h "$CKPT" | tail -1

if [ -f "$ASSEMBLED" ] && head -c4 "$ASSEMBLED" 2>/dev/null | grep -q "GGUF" && [ ! -f "$DONE.inprogress" ]; then
  # already complete if magic ok and no in-progress marker and no dl parts remain
  if ! ls "$DL_DIR/${BASE}.gguf.part"*of* >/dev/null 2>&1; then
    echo "already assembled: $ASSEMBLED ($(du -h "$ASSEMBLED" | cut -f1))"
  fi
fi

# Discover M by fetching just part1 first (small metadata call: download part1, read its 'partNofM').
echo "STEP 1: fetch part 1 to learn part count ..."
hf download "$REPO" --include "${BASE}.gguf.part1of*" --local-dir "$DL_DIR"
FIRST=$(ls "$DL_DIR/${BASE}.gguf.part1of"* 2>/dev/null | head -1 || true)
[ -n "$FIRST" ] || { echo "ERROR: could not download part1"; ls -la "$DL_DIR" | head; exit 3; }
M=$(echo "$FIRST" | sed -E 's/.*part1of([0-9]+)$/\1/')
echo "part count M=$M"

START=1
if [ -f "$DONE" ]; then START=$(( $(cat "$DONE") + 1 )); echo "resuming: $((START-1))/$M parts already appended"; fi
[ "$START" = 1 ] && { rm -f "$ASSEMBLED"; : > "$ASSEMBLED"; }
touch "$DONE.inprogress"

echo "STEP 2: streaming download+append parts $START..$M"
for i in $(seq "$START" "$M"); do
  p="$DL_DIR/${BASE}.gguf.part${i}of${M}"
  if [ ! -s "$p" ]; then
    echo "  downloading part $i/$M ..."
    hf download "$REPO" --include "${BASE}.gguf.part${i}of${M}" --local-dir "$DL_DIR"
  fi
  [ -s "$p" ] || { echo "ERROR: part $i missing after download"; exit 3; }
  echo "  appending part $i/$M ($(du -h "$p" | cut -f1)) ..."
  cat "$p" >> "$ASSEMBLED"
  echo "$i" > "$DONE"
  rm -f "$p"                     # free the part immediately (keeps peak usage low)
done
rm -f "$DONE.inprogress"

echo "assembled size: $(du -h "$ASSEMBLED" | cut -f1)"
head -c4 "$ASSEMBLED" | grep -q "GGUF" && echo "GGUF magic OK" || { echo "ERROR: bad GGUF magic"; exit 4; }

# symlink into workdir models for the recipes
mkdir -p "$NEMO/models/gguf-n4"
ln -sf "$ASSEMBLED" "$NEMO/models/gguf-n4/nemotron4-340b-${QUANT}.gguf"
echo "symlink: $NEMO/models/gguf-n4/nemotron4-340b-${QUANT}.gguf -> $ASSEMBLED"
echo "DOWNLOAD_OK -> $ASSEMBLED"
