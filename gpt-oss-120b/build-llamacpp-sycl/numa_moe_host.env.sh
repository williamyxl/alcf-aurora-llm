#!/bin/bash
# Host CPU + memory binding for llama.cpp on Aurora (MoE→CPU and host threads).
#
# Source of truth:
#   https://docs.alcf.anl.gov/aurora/running-jobs-aurora/#mpi-rank-and-thread-binding-to-cores-and-gpus
#   https://docs.alcf.anl.gov/aurora/running-jobs-aurora/#using-the-hbm-on-the-sapphire-rapids-cpus
#
# Aurora node (ALCF docs):
#   2 sockets × (1 SPR CPU + 3 Max 1550 GPUs)
#   Socket 0: CPUs HWT 0-51 + 104-155 ; GPUs 0,1,2
#   Socket 1: CPUs HWT 52-103 + 156-207 ; GPUs 3,4,5
#   NUMA (Flat): node0=DDR sock0 (~512G), node1=DDR sock1 (~512G),
#                node2=HBM sock0 (64G),   node3=HBM sock1 (64G)
#   Reserved OS cores (do not use): physical core 0 → HWT 0,104 ; core 52 → HWT 52,156
#
# Our campaign FLAT tile masks:
#   ZE_AFFINITY_MASK=0 or 0,1  → GPU0 tiles → MUST bind socket 0 (NUMA 0 / HBM 2)
#   Never use guessed HBM ids 8-15 — those are wrong on Aurora.
#
# Vars consumed by bench_llamacpp_sycl_perf.pbs:
#   NUMACTL_ENABLE=1
#   NUMACTL_ARGS='...'
#   LLAMA_NUMA=numactl

# Default: GPU0-local DDR (capacity for ~50–60G MoE; avoids remote sock1 DDR/HBM)
export NUMACTL_ENABLE=${NUMACTL_ENABLE:-1}
# physcpubind excludes reserved cores 0/52; membind=0 → socket-0 DDR only
export NUMACTL_ARGS=${NUMACTL_ARGS:---physcpubind=1-51,105-155 --membind=0}
export LLAMA_NUMA=${LLAMA_NUMA:-numactl}

# Optional overrides (set in cycle env before/after sourcing):
#   CPU-HBM preferred on sock0 (64G; overflow → DDR):
#     NUMACTL_ARGS='--physcpubind=1-51,105-155 --preferred=2'
#   CPU-HBM only sock0 (OOM if MoE > ~64G):
#     NUMACTL_ARGS='--physcpubind=1-51,105-155 --membind=2'
#   Socket 1 (only if ZE_AFFINITY_MASK targets GPUs 3–5 / FLAT tiles 6–11):
#     NUMACTL_ARGS='--physcpubind=53-103,157-207 --membind=1'
#
# Always verify on the compute node: numactl -H
