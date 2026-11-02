#!/usr/bin/env bash
# Run on the Grid after reviewing the requested stage's missing-run inventory.
set -euo pipefail
: "${EXPERIMENT:?Set frontier or pareto}"
: "${STAGE:?Set search or evaluate}"
: "${E2_ROOT:?Expanded frontier root required}"
TC="${TC:-20}"
if [[ ! "$TC" =~ ^[1-9][0-9]*$ ]] || (( TC > 20 )); then echo 'TC must be 1..20' >&2; exit 2; fi
case "$EXPERIMENT:$STAGE" in
    frontier:search) root="$E2_ROOT"; disk_stage=search; job=jobs/run_experiment2_frontier.sh ;;
    frontier:evaluate) root="$E2_ROOT"; disk_stage=evaluation; job=jobs/run_experiment2_frontier.sh ;;
    pareto:evaluate) root="${E2_PARETO_ROOT:?Separate Pareto root required}"; disk_stage=evaluation; job=jobs/run_experiment2_pareto_completion.sh ;;
    *) echo 'Use frontier search/evaluate or pareto evaluate' >&2; exit 2 ;;
esac
command -v qsub >/dev/null
shards="$root/inventory/${disk_stage}_shards.txt"
test -f "$shards"
# Keep stable bank indices. -sync y serializes disjoint arrays and propagates
# failed tasks; no dependent selection/postprocessing starts automatically.
ranges=$("$PYTHON" - "$shards" "$root/bank.csv" <<'PY'
import sys
from pathlib import Path
import pandas as pd
ids=sorted(set(map(int,Path(sys.argv[1]).read_text().split())))
n=len(pd.read_csv(sys.argv[2]))
if any(i<1 or i>n for i in ids):raise SystemExit('Invalid shard index')
ranges=[]
for i in ids:
    if ranges and i==ranges[-1][1]+1:ranges[-1][1]=i
    else:ranges.append([i,i])
for lo,hi in ranges:print(f'{lo}-{hi}')
PY
)
if [[ -z "$ranges" ]]; then echo 'No missing shards to submit.'; exit 0; fi
while read -r range; do
    qsub -sync y -N "e2c_${EXPERIMENT}_${STAGE}" -cwd -V -t "$range" -tc "$TC" \
        -v "STAGE=$STAGE" "$job"
done <<< "$ranges"
