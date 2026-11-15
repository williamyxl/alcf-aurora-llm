#!/usr/bin/env bash
# Small helpers shared by recipes. Source after env.sh.
wait_health() {                                              # wait_health <url> [tries] [pid]
  local url=$1 tries=${2:-180} pid=${3:-}
  for _ in $(seq 1 "$tries"); do
    curl -sf "$url" >/dev/null 2>&1 && return 0
    [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null && { echo "server died"; return 2; }
    sleep 10
  done; return 1
}
require_file() { [ -e "$1" ] || { echo "ERROR: missing $1 ${2:+($2)}"; exit 2; }; }
first_shard() { ls "$1"/*-00001-of-*.gguf 2>/dev/null | head -1; }   # multi-part GGUF dir -> first shard
write_endpoint() {                                           # write_endpoint <file> <port> <served-name>
  local ip; ip=$(hostname -i | awk '{print $1}')
  printf 'node=%s\nip=%s\nurl=http://%s:%s/v1\nmodel=%s\njob=%s\nstarted=%s\n' \
    "$(hostname)" "$ip" "$ip" "$2" "$3" "${PBS_JOBID:-none}" "$(date -Is)" > "$1"
}
