#!/bin/bash
# Submit a chain of N week-long gpt-oss-120b serving jobs on the capacity queue.
# Each successor starts when its predecessor FINISHES (afterany) -> one node at a time,
# minimal cost, brief queue-wait gap at each weekly handoff.
#
#   ./submit_chain.sh [N]     # N = number of chained jobs (default 2 = ~2 weeks)
#
# After each handoff the node changes; refresh the client endpoint from ENDPOINT.txt
# (written by the running job). For weekly cadence this manual refresh is fine.
set -euo pipefail
SERVE=/lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm/gpt-oss-120b/build-vllm-xpu/opus4.8-new_campaign/serve
JOB="$SERVE/vllm_serve_node.pbs"
N=${1:-2}
rm -f "$SERVE/STOP"

prev=$(qsub "$JOB")
echo "job 1: $prev"
for i in $(seq 2 "$N"); do
  jid=$(qsub -W depend=afterany:"$prev" "$JOB")
  echo "job $i: $jid (after $prev)"
  prev="$jid"
done
echo
echo "Chain of $N submitted. Each runs up to 168h."
echo "Live endpoint (updates at each handoff): $SERVE/ENDPOINT.txt"
echo "Stop the chain: touch $SERVE/STOP   (current job exits; successors with afterany still start -"
echo "                to fully stop, also 'qdel' the queued successors: qstat -u \$USER)"
