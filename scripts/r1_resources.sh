#!/usr/bin/env bash
# Log memory of the running training jobs every INTERVAL seconds, to spot leaks over the epochs.
#
# Usage: setsid nohup scripts/r1_resources.sh [interval_s] > /dev/null 2>&1 &
# Appends to models/r1_queue/resources.csv:
#   time, run, pid, rss_gb (process plus its dataloader workers), gpu_mb, free_ram_gb, gpu_used_mb
set -u
cd "$(dirname "$0")/.."
INTERVAL=${1:-300}
OUT=models/r1_queue/resources.csv
mkdir -p models/r1_queue
echo $$ > models/r1_queue/resources.pid  # stop with: kill $(cat models/r1_queue/resources.pid)
[ -s "$OUT" ] || echo "time,run,pid,rss_gb,gpu_mb,free_ram_gb,gpu_used_mb" > "$OUT"
while true; do
    t=$(date +%Y-%m-%dT%H:%M:%S)
    free_gb=$(awk '/MemAvailable/ {printf "%.1f", $2 / 1048576}' /proc/meminfo)
    gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    for pid in $(pgrep -f "src.trainers.train_terratorch --config"); do
        run=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -o 'config-[^ ]*\.yaml' | sed 's/config-//; s/\.yaml//')
        [ -n "$run" ] || continue
        # the trainer plus all descendants (dataloader workers hang below a forkserver process)
        all=$pid; frontier=$pid
        while [ -n "$frontier" ]; do
            frontier=$(for p in $frontier; do pgrep -P "$p"; done | tr '\n' ' ')
            all="$all $frontier"
        done
        pids=$(echo $all | tr ' ' ',')
        rss_kb=$(ps -o rss= -p "$pids" 2>/dev/null | awk '{s += $1} END {print s + 0}')
        gpu_mb=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
            | awk -F', ' -v p="$pid" '$1 == p {print $2}')
        echo "$t,$run,$pid,$(awk -v k="$rss_kb" 'BEGIN {printf "%.2f", k / 1048576}'),${gpu_mb:-0},$free_gb,$gpu_used" >> "$OUT"
    done
    sleep "$INTERVAL"
done
