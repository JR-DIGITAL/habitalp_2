#!/usr/bin/env bash
# Run the revision inference queue on one machine, N jobs at a time.
#
# Usage: scripts/r1_infer_queue.sh [slots]
# Generates the configs with scripts/make_r1_infer_configs.py (trained runs in models/),
# then runs src/inference/inference.py for each: <run>-2020, <run>-2013 (best val checkpoint) and
# <run>-2020-last (last epoch). Rasters go to <data root>/model_output/<run>/<year>_<run>.tif and
# <data root>/model_output/<run>-last/2020_<run>-last.tif.
# State lives in models/r1_infer/: <job>.log and <job>.status, as in scripts/r1_queue.sh; jobs with
# a status file are skipped, so the queue can be restarted; to rerun a job, delete its .status file and
# any tmp_patches_*/ or *_uncompressed.tif it left in its output folder. Rerun the queue after new
# training runs finish: it picks up their jobs.
# Output is also appended to models/r1_infer/queue.log. Start in tmux: scripts/r1_infer_queue.sh 2
set -u
cd "$(dirname "$0")/.."
SLOTS=${1:-2}
STATE=models/r1_infer
PY=${PY:-.venv314/bin/python}
mkdir -p "$STATE"
exec > >(tee -a "$STATE/queue.log") 2>&1
export HABITALP_DATA_ROOT=${HABITALP_DATA_ROOT:-$PWD/data/Originaldaten}
export PYTHONUNBUFFERED=1
export GDAL_CACHEMAX=${GDAL_CACHEMAX:-512}
# torch uses every core per process by default; with several jobs and training next to them that
# oversubscribes the CPU (does not change the outputs)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
[ -e /usr/lib64/libjpeg.so.8 ] && export LD_PRELOAD=/usr/lib64/libjpeg.so.8

now() { date +%Y-%m-%dT%H:%M:%S; }

run_one() {
    local job=$1 start rc year run out
    start=$(now)
    echo "running $start" > "$STATE/$job.status"
    # job name <run>-<year>[-last] -> output name <year>_<run>[-last]
    if [[ $job == *-last ]]; then year=${job%-last}; year=${year##*-}; run=${job%-*-last}; out="${year}_${run}-last"
    else year=${job##*-}; run=${job%-*}; out="${year}_${run}"; fi
    "$PY" src/inference/inference.py --config "$STATE/configs/infer-$job.yaml" --output "$out" \
        > "$STATE/$job.log" 2>&1
    rc=$?
    echo "done $start $(now) rc=$rc" > "$STATE/$job.status"
    echo "$(now) finished $job rc=$rc"
}

JOBS=$("$PY" scripts/make_r1_infer_configs.py --out "$STATE/configs") || exit 1
echo "$(now) inference queue, $(echo "$JOBS" | wc -l) jobs, $SLOTS slots, pid $$"
for job in $JOBS; do
    if [ -e "$STATE/$job.status" ]; then
        echo "$(now) skip $job ($(cat "$STATE/$job.status"))"
        continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do
        wait -n
    done
    [ -e "$STATE/$job.status" ] && continue  # started by another queue instance meanwhile
    echo "$(now) start $job"
    run_one "$job" &
    sleep 30
done
# plain `wait` would also wait for the tee of the log redirection above and never return
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do
    wait -n
done
echo "$(now) inference queue finished"
