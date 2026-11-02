# Complete Experiment 2: 60%/80% Class 1 shares and 40 slots/day

This runbook prepares simulations only. Do not replace current graphs or committed
result tables until the expanded outputs pass completeness checks.

Final design: 9 behavior profiles x 6 rho values x 7 Class 1 shares x 3 sizes =
**1,134 scenarios**. Shares: 10/30/50/60/70/80/90%. Sizes: 30/40/50 slots/day.
The existing 972 frontier identities and inputs are unchanged; 60% adds 162.
Original 540 and partial share80 54 IDs remain recognizable. Stable array indices
belong to the NEW bank, so do not reuse old array index ranges.

Both experiments retain reservation-only and both-flexible; no horizon-only runs
are added. Both-flexible graphs can continue to filter out reservation-only.
The legacy 972 design remains the default; expansion is opt-in.

## 1. Sync only the simulation changes

The attached code ZIP contains these files, with repository-relative paths:

- analysis/experiment2_frontier.py
- analysis/experiment2_pareto_completion.py
- jobs/run_experiment2_frontier.sh
- jobs/run_experiment2_pareto_completion.sh
- jobs/submit_experiment2_completion.sh
- scripts/prepare_experiment2_completion_sources.py
- tests/test_experiment2_frontier.py
- tests/test_experiment2_completion.py
- docs/experiment2_completion.md

With WinSCP, upload those files into the matching folders under
`~/projects/CUIMC-Appointment-Simulation/`. Preserve the directory structure.
Do not upload figures, the presentation, downloaded result CSVs, or a `.git` folder.
Use binary transfer or preserve LF line endings for shell scripts.

Alternatively, commit/push only the listed files locally, then run
`git status --short` on the Grid before `git pull --ff-only`. A pull needs those
local commits pushed first. Do not run `git add .`, reset, or stash unrelated work.
No commits or pushes were made by Codex.

The earlier chat's accessible history did not contain its original sync commands.
These paths and Python defaults are verified from the existing Grid job scripts.

## 2. Configure the Grid session and test

Run from your persistent Grid shell (the submission commands wait for completion):

```bash
export REPO="$HOME/projects/CUIMC-Appointment-Simulation"
export PYTHON="$HOME/.conda/envs/cuimc/bin/python"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export E2_PREVIOUS_ROOT="/scratch/$USER/experiment2_frontier"
export E2_ROOT="/scratch/$USER/experiment2_completion_20261102/frontier"
export E2_PARETO_ROOT="/scratch/$USER/experiment2_completion_20261102/pareto"
export E2_SOURCES="$E2_ROOT/reuse_sources.json"
export INCLUDE_SHARE60=1
cd "$REPO"
"$PYTHON" -m unittest tests.test_experiment2_frontier tests.test_experiment2_completion
for job in jobs/run_experiment2_frontier.sh jobs/run_experiment2_pareto_completion.sh jobs/submit_experiment2_completion.sh; do
    bash -n "$job" || exit 1
done
```

If your completed frontier is stored elsewhere, change E2_PREVIOUS_ROOT. It must
contain `freeze.json`, `bank.csv`, `search/raw/` and `evaluation/raw/`. Aggregated
CSV outputs alone cannot replace those seed-level results. The new output roots
must differ from existing frozen roots. Do not copy a freeze file into a new root.

## 3. Create the expanded bank and reuse manifest

```bash
STAGE=bank bash jobs/run_experiment2_frontier.sh
"$PYTHON" scripts/prepare_experiment2_completion_sources.py \
    --previous-root "$E2_PREVIOUS_ROOT" --new-root "$E2_ROOT"
```

The manifest retains the old frontier's historical sources (including Pareto and
share80 raw results) and adds its completed frontier raw directories. All source
banks/directories must still exist; missing paths cause a failure. If an archive
was moved, locate it and correct the copied manifest BEFORE any selections freeze.
Do not silently remove missing sources, mark unknown conventions verified, or
substitute output summaries. Source conventions must match burn-in=30,
measurement=365, cooldown=max(14,H), fixed uncapped thresholds and the same
cancellation/release semantics. Newly generated runs use the unchanged engine.

Search uses seeds 1000..1004. Independent evaluation uses 2000..2019. Raw reuse
matches actual simulation inputs and H/Q/W/seed, not just scenario IDs. Conflicting
reused values abort rather than silently choosing a source.

## 4. Inventory and submit missing discovery runs

```bash
qsub -sync y -cwd -V -N e2c_inventory -v STAGE=inventory jobs/run_experiment2_frontier.sh
"$PYTHON" - <<'PY'
import os
from pathlib import Path
import pandas as pd
p=Path(os.environ['E2_ROOT'])/'inventory/search_coverage.csv'
x=pd.read_csv(p)
print(x[['planned','reused','pending']].sum())
print('Scenarios with missing runs:', int((x.pending>0).sum()))
PY
```

Read the inventory before submission. The extra 60% scenarios should need search;
additional historical gaps may also be found. Costs cannot be estimated from the
972-row summary alone. Then submit only the listed missing shards:

```bash
EXPERIMENT=frontier STAGE=search TC=20 bash jobs/submit_experiment2_completion.sh
qsub -sync y -cwd -V -N e2c_inventory -v STAGE=inventory jobs/run_experiment2_frontier.sh
"$PYTHON" - <<'PY'
import os,pandas as pd
from pathlib import Path
x=pd.read_csv(Path(os.environ['E2_ROOT'])/'inventory/search_coverage.csv')
assert x.pending.sum()==0, 'Search incomplete: inspect logs and rerun missing shards'
print('Discovery search complete')
PY
```

The helper groups stable indices into contiguous ranges, uses at most 20
concurrent tasks in one array, and waits for each array. An empty inventory is a
successful no-op. Do not submit overlapping arrays or run frontier and Pareto
evaluation arrays at the same time if you want the total concurrency capped at 20.
Standard Grid output/error files appear in the working directory. On failure,
inspect them, regenerate inventory, and submit only remaining rows. Each shard
rechecks raw coverage before running and flushes completed results for resumption.

## 5. Freeze and evaluate the frontier

```bash
qsub -sync y -cwd -V -N e2c_select -v STAGE=select jobs/run_experiment2_frontier.sh
qsub -sync y -cwd -V -N e2c_inventory -v STAGE=inventory jobs/run_experiment2_frontier.sh
EXPERIMENT=frontier STAGE=evaluate TC=20 bash jobs/submit_experiment2_completion.sh
qsub -sync y -cwd -V -N e2c_inventory -v STAGE=inventory jobs/run_experiment2_frontier.sh
"$PYTHON" - <<'PY'
import os,pandas as pd
from pathlib import Path
x=pd.read_csv(Path(os.environ['E2_ROOT'])/'inventory/evaluation_coverage.csv')
assert x.pending.sum()==0, 'Frontier evaluation is incomplete'
print('Frontier evaluation complete')
PY
qsub -sync y -cwd -V -N e2c_report -v STAGE=postprocess jobs/run_experiment2_frontier.sh
```

Selection uses discovery outcomes only, maximizing Class 1 gain at each tested
Class 2 loss budget (0, 0.5, 1, 2, 3, 4, 5, 7.5, 10pp), with existing tie rules.
Existing independent raw evaluations are reused wherever the frozen cell matches.
Do not repeat `select` after a freeze. Resume with inventory/evaluate/postprocess.

## 6. Freeze and evaluate the Pareto catalog from the SAME discovery runs

```bash
qsub -sync y -cwd -V -N e2c_pcatalog -v STAGE=catalog jobs/run_experiment2_pareto_completion.sh
qsub -sync y -cwd -V -N e2c_pinventory -v STAGE=inventory jobs/run_experiment2_pareto_completion.sh
"$PYTHON" - <<'PY'
import os,pandas as pd
from pathlib import Path
x=pd.read_csv(Path(os.environ['E2_PARETO_ROOT'])/'inventory/evaluation_coverage.csv')
print(x[['planned','reused','pending']].sum())
PY
EXPERIMENT=pareto STAGE=evaluate TC=20 bash jobs/submit_experiment2_completion.sh
qsub -sync y -cwd -V -N e2c_pinventory -v STAGE=inventory jobs/run_experiment2_pareto_completion.sh
"$PYTHON" - <<'PY'
import os,pandas as pd
from pathlib import Path
x=pd.read_csv(Path(os.environ['E2_PARETO_ROOT'])/'inventory/evaluation_coverage.csv')
assert x.pending.sum()==0, 'Pareto evaluation is incomplete'
print('Pareto evaluation complete')
PY
qsub -sync y -cwd -V -N e2c_preport -v STAGE=postprocess jobs/run_experiment2_pareto_completion.sh
```

Pareto freezes ALL discovery cells with neither class harmed and >=0.5pp gain for
at least one class. It does not evaluate only the frontier's L=0 winner. Additional
Pareto candidates receive their own 20-seed independent evaluations, reusing old
and newly completed frontier evaluations. Postprocessing retains candidates that
fail independent validation; evaluation is never used to replace discovery choices.
The full scenario-policy universe includes scenarios with no qualifying candidates.

This completion uses a uniform systematic genuine-reservation grid across all
1,134 scenarios, common to the frontier (H=2..30 both-flexible; H=100 reservation;
Q in 5-slot steps; historical W>=2 grid). Older adaptive/refined Pareto catalogs
remain untouched. Their extra off-grid candidates are not silently mixed into the
new common-grid catalog. This improves comparability but should be distinguished
from simply appending the old selectively refined catalog.

## 7. Download results for updating the graphs

From `$E2_ROOT`, download:

- bank.csv
- evaluated_selected_cells.csv
- evaluated_seed_deltas.csv
- strict_endpoint_comparison.csv
- legacy_rule_discovery_best.csv
- freeze.json
- reuse_sources.json
- inventory/search_coverage.csv and inventory/evaluation_coverage.csv

From `$E2_PARETO_ROOT`, download:

- patient_behavior_factorial_bank.csv
- validated_meaningful_pareto_cells.csv
- validated_meaningful_pareto_seed_deltas.csv
- validated_background_policy_universe.csv
- selected_pareto_discovery.csv
- pareto_freeze.json
- inventory/evaluation_coverage.csv

Keep the two experiment outputs separate. Send the downloaded outputs back here
before we update graphs. The frontier's background target/minimum-budget summary
CSVs will be regenerated from these completed core results; old summaries should
not be appended to or treated as the expanded design.

## Local verification and practical limits

Nine Python tests pass. They verify old IDs/rows are preserved, the expanded bank,
reuse, frozen choices, incomplete-search/evaluation rejection, all-Pareto-cell
selection and independent failures, and the complete scenario-policy denominator.
Tests use synthetic raw outcomes and do not run a large simulation sweep. Local
Bash execution is blocked by this Windows environment, so `bash -n` is explicitly
included in the Grid setup. Remote scratch availability and actual queue execution
must be checked by the user. No Grid jobs have been submitted by Codex.
