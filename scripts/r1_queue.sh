#!/usr/bin/env bash
# Run the revision training queue on one machine, N runs at a time.
#
# Usage: scripts/r1_queue.sh <queue file> [slots]
# The queue file lists run names (one per line, '#' comments allowed), e.g. r1-unet-mit-b2-masked-s0;
# each needs src/trainers/terratorch/revision/config-<run>.yaml. State lives in models/r1_queue/:
#   <run>.log     training output
#   <run>.status  "running <start>" while active, "done <start> <end> rc=<code>" afterwards
# Runs with a status file are skipped, so the queue can be restarted; delete a .status file to rerun.
# Start in tmux:   tmux new-session -d -s r1queue "scripts/r1_queue.sh <queue file> 2 2>&1 | tee -a models/r1_queue/queue.log"
# Watch:           tmux attach -t r1queue   (detach with Ctrl-b d)
set -u
cd "$(dirname "$0")/.."
QUEUE=$1
SLOTS=${2:-2}
STATE=models/r1_queue
PY=${PY:-.venv314/bin/python}
mkdir -p "$STATE"
export HABITALP_DATA_ROOT=${HABITALP_DATA_ROOT:-$PWD/data/Originaldaten}
export PYTHONUNBUFFERED=1
# GDAL keeps a raster block cache of 5% of RAM per process by default; with 8 dataloader
# workers per run that is about 45 GB per run on a 125 GB machine. 512 MB per process is plenty.
export GDAL_CACHEMAX=${GDAL_CACHEMAX:-512}
[ -e /usr/lib64/libjpeg.so.8 ] && export LD_PRELOAD=/usr/lib64/libjpeg.so.8

now() { date +%Y-%m-%dT%H:%M:%S; }

run_one() {
    local run=$1 start rc
    start=$(now)
    echo "running $start" > "$STATE/$run.status"
    "$PY" -m src.trainers.train_terratorch --config "src/trainers/terratorch/revision/config-$run.yaml" \
        > "$STATE/$run.log" 2>&1
    rc=$?
    echo "done $start $(now) rc=$rc" > "$STATE/$run.status"
    echo "$(now) finished $run rc=$rc"
}

echo "$(now) queue $QUEUE, $SLOTS slots, pid $$"
for run in $(grep -v '^\s*#' "$QUEUE" | awk 'NF {print $1}'); do
    if [ -e "$STATE/$run.status" ]; then
        echo "$(now) skip $run ($(cat "$STATE/$run.status"))"
        continue
    fi
    if [ ! -e "src/trainers/terratorch/revision/config-$run.yaml" ]; then
        echo "$(now) no config for $run, skipped"
        continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do
        wait -n
    done
    echo "$(now) start $run"
    run_one "$run" &
    sleep 60  # let the first run allocate GPU memory before the next one starts
done
wait
echo "$(now) queue finished"
