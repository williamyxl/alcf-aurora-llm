#!/bin/bash
# Build a long English filler prompt ending with the MOF quality question.
# Usage: make_fill_prompt.sh <approx_target_tokens> <outfile>
# Token estimate ≈ ceil(chars / 4) for English prose (conservative for gpt-oss).
set -euo pipefail
TARGET_TOK=${1:?target tokens}
OUT=${2:?outfile}
CHARS_PER_TOK=${CHARS_PER_TOK:-4}
TARGET_CHARS=$((TARGET_TOK * CHARS_PER_TOK))

QUESTION='What is a metal-organic framework (MOF)? Answer in one short paragraph of clear English.'
PARA='Metal-organic frameworks are crystalline porous materials built from metal nodes and organic linkers. They are studied for gas storage, catalysis, sensing, and separations. Researchers vary metal identity, linker length, and functional groups to tune pore size, surface area, and chemical affinity. Stability under humidity and thermal cycling remains an important design constraint. '
PLEN=${#PARA}

HEADER="### Long-context filler (approx ${TARGET_TOK} tokens) for GPT-OSS-120B max-context stress.
Ignore the filler; answer only the final question.

"
FOOTER="
### Final question
${QUESTION}
"

BUDGET=$((TARGET_CHARS - ${#QUESTION} - ${#HEADER} - 64))
[ "$BUDGET" -lt 500 ] && BUDGET=500

{
  printf '%s' "$HEADER"
  # Emit whole paragraphs without per-iteration wc (fast at 128k scale)
  _n=$(( (BUDGET + PLEN - 1) / PLEN ))
  _i=0
  while [ "$_i" -lt "$_n" ]; do
    printf '%s' "$PARA"
    _i=$((_i + 1))
  done
  printf '%s' "$FOOTER"
} > "$OUT"

echo "WROTE=$OUT bytes=$(wc -c < "$OUT") approx_tok_est=$(( $(wc -c < "$OUT") / CHARS_PER_TOK ))"
