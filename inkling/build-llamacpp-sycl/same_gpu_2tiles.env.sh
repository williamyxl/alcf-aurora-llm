#!/bin/bash
# Pin tiles of Max 1550 for llama.cpp SYCL jobs.
#
# Default (Phase A–E, explicit 2 tiles, FLAT): GPU0 tile0 + tile1.
#   ZE_FLAT_DEVICE_HIERARCHY=FLAT
#   ZE_AFFINITY_MASK=0,1
#
# Phase F (1 tile + MoE on CPU): set in cycles/F*.env before sourcing this file.
#   ZE_FLAT_DEVICE_HIERARCHY=FLAT
#   ZE_AFFINITY_MASK=0
#
# Alternate (implicit whole GPU0, COMPOSITE):
#   ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
#   ZE_AFFINITY_MASK=0
#
# Override via PBS -v ZE_FLAT_DEVICE_HIERARCHY=... , ZE_AFFINITY_MASK=...

# NOTE: oneapi module often exports ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE.
# Callers must source cycle/*.env AFTER `module load oneapi`, then this file.
# FLAT + 0,1 = two tiles of GPU0. COMPOSITE + 0,1 = two whole GPUs (forbidden here).
export ZE_FLAT_DEVICE_HIERARCHY=${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0,1}
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}
export ZES_ENABLE_SYSMAN=${ZES_ENABLE_SYSMAN:-1}
# Required on Aurora PVC for gpt-oss-120b MXFP4: default VMM=1 OOMs at warmup mul_mat
export GGML_SYCL_ENABLE_VMM=${GGML_SYCL_ENABLE_VMM:-0}
unset SYCL_CACHE_PERSISTENT

echo "ZE_FLAT_DEVICE_HIERARCHY=$ZE_FLAT_DEVICE_HIERARCHY"
echo "ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK"
echo "ONEAPI_DEVICE_SELECTOR=$ONEAPI_DEVICE_SELECTOR"
echo "ZES_ENABLE_SYSMAN=$ZES_ENABLE_SYSMAN"
echo "GGML_SYCL_ENABLE_VMM=$GGML_SYCL_ENABLE_VMM"

if [ "$ZE_FLAT_DEVICE_HIERARCHY" = "COMPOSITE" ] && [[ "$ZE_AFFINITY_MASK" == *","* ]]; then
  echo "WARNING: COMPOSITE + multi-id affinity likely selects multiple GPUs, not two tiles of one GPU."
fi

# Count tiles in affinity mask (commas + 1)
_NMASK=$(echo "$ZE_AFFINITY_MASK" | awk -F',' '{print NF}')
export TP=${TP:-$_NMASK}
_PACK=${PACK_TILES_PER_GPU:-2}
echo "TP=$TP AFFINITY_TILE_COUNT=$_NMASK PACK_TILES_PER_GPU=$_PACK ALLOW_MULTI_GPU=${ALLOW_MULTI_GPU:-0}"

if [ "${ALLOW_MULTI_GPU:-0}" != "1" ] && [ "$_NMASK" -gt 2 ]; then
  echo "REFUSING: affinity has $_NMASK tiles but ALLOW_MULTI_GPU!=1 (same-GPU 2-tile rule)."
  echo "For P14 TP scaling set ALLOW_MULTI_GPU=1 in the cycle env."
  exit 4
fi

# Dense packing: Max 1550 has 2 tiles/GPU — require full GPU pairs, contiguous from 0.
# MULTI_NODE=1: affinity is local-node tiles (0..11); TP may exceed local count (see PLAN.md).
if [ "${ALLOW_MULTI_GPU:-0}" = "1" ] && [ "$_NMASK" -gt 2 ]; then
  if [ $((_NMASK % _PACK)) -ne 0 ]; then
    echo "REFUSING: tile count $_NMASK is not a multiple of PACK_TILES_PER_GPU=$_PACK (incomplete GPU)."
    exit 4
  fi
  _EXPECT=$(seq -s, 0 $((_NMASK - 1)))
  if [ "$ZE_AFFINITY_MASK" != "$_EXPECT" ]; then
    echo "REFUSING: affinity '$ZE_AFFINITY_MASK' is not densely packed prefix '$_EXPECT'."
    echo "Rule: use both tiles of each GPU before adding another GPU; contiguous FLAT ids from 0."
    exit 4
  fi
  _NGPU=$((_NMASK / _PACK))
  echo "NOTE: packed multi-GPU run — ${_NGPU} Max 1550(s) × ${_PACK} tiles (affinity tiles=$_NMASK, TP=${TP})."
  if [ "${MULTI_NODE:-0}" = "1" ] && [ "${TP:-$_NMASK}" -gt "$_NMASK" ]; then
    echo "WARNING: MULTI_NODE=1 TP=$TP > local affinity tiles=$_NMASK — single-process SYCL is node-local; expect FAIL unless RPC/multi-rank is wired."
  fi
fi

if command -v sycl-ls >/dev/null 2>&1; then
  echo "=== sycl-ls (pinned) ==="
  sycl-ls --ignore-device-selectors 2>&1 | head -5 || true
  sycl-ls 2>&1 | head -40
  # Count visible Level Zero GPU lines
  NDEV=$(sycl-ls 2>/dev/null | grep -c 'level_zero:gpu' || true)
  echo "SYCL_GPU_LINES=$NDEV"
fi
