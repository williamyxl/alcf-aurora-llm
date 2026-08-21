#!/bin/bash
# Run on an Aurora LOGIN NODE (UAN) to tunnel the LB endpoint from the compute node to the UAN.
# This makes the API reachable at http://<uan_ip>:LOCAL_PORT/v1 from inside ANL,
# or via a further SSH tunnel from your laptop (see README.md).
#
# Usage (on a UAN after qsub):
#   bash api_serve_high_concurrency/uan_tunnel.sh [ENDPOINT.txt] [LOCAL_PORT]
#
# The tunnel keeps running in the foreground; Ctrl-C or job end kills it.
# It auto-reconnects if the SSH connection drops.

ENDPOINT_FILE=${1:-$(dirname "$0")/ENDPOINT.txt}
LOCAL_PORT=${LOCAL_PORT:-${2:-8000}}

if [ ! -f "$ENDPOINT_FILE" ]; then
  echo "ERROR: ENDPOINT.txt not found at $ENDPOINT_FILE"
  echo "  — wait for the PBS job to start and write it, then rerun."
  exit 1
fi

NODE_IP=$(grep '^ip=' "$ENDPOINT_FILE" | cut -d= -f2)
LB_PORT=$(grep '^lb_url=' "$ENDPOINT_FILE" | sed 's|.*:\([0-9]*\)/.*|\1|')
MODEL=$(grep '^model=' "$ENDPOINT_FILE" | cut -d= -f2)
JOB=$(grep '^job=' "$ENDPOINT_FILE" | cut -d= -f2)

echo "=== UAN API tunnel ==="
echo "  compute node:  $NODE_IP:$LB_PORT  (job $JOB, model $MODEL)"
echo "  UAN endpoint:  http://$(hostname -i | awk '{print $1}'):$LOCAL_PORT/v1"
echo "  For laptop access, see README.md (SSH tunnel from laptop through this UAN)."
echo "  Press Ctrl-C to stop."
echo

# Auto-reconnect loop (in case the SSH connection drops)
while true; do
  echo "$(date -Is) opening tunnel $LOCAL_PORT -> $NODE_IP:$LB_PORT"
  ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -o ExitOnForwardFailure=yes \
      -L "$LOCAL_PORT:$NODE_IP:$LB_PORT" localhost 2>&1
  rc=$?
  # Exit clean if compute node job ended (connection refused)
  if ! curl -sf "http://$NODE_IP:$LB_PORT/health" >/dev/null 2>&1; then
    echo "$(date -Is) compute node endpoint gone (job ended?); exiting tunnel."
    break
  fi
  echo "$(date -Is) tunnel dropped (rc=$rc); reconnecting in 10s…"
  sleep 10
done
