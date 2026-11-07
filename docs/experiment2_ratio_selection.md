# Revised policy selection: optional reservation and signed loss-per-gain

This revision is opt-in. Preserve the existing frozen selection and the 44,120
previously pending evaluations; do not submit those evaluations for the new rule.
Completed discovery simulations remain reusable.

## Both-flexible includes horizon-only

With `ALLOW_HORIZON_ONLY=1`, the both-flexible candidate pool includes the existing
reserved cells plus (H, Q, W)=(H,0,-1), H=2..30. Q=0 means no reserved slots;
W=-1 is the engine's no-reservation sentinel, not a literal negative duration.
No separate horizon-only series is required in the graphs: these candidates are
eligible to win under the both-flexible label. Existing positive-reservation cells
still use W>=2. Reservation-only retains its existing genuine-reservation grid.

This adds 29 discovery cells per scenario. Some are likely already present in the
historical horizon-only raw sources, but only a fresh inventory establishes how
many require new simulation. The maximum added discovery requirement is
1,134 x 29 x 5 = 164,430 seed runs before reuse.

## Trade-off rule

With `E2_SELECTION_RULE=min_loss_per_gain`, select using discovery means:

1. Class 1 gain >=0.05 (5 percentage points).
2. Class 2 delta >= -allowed budget, keeping the existing maximum 0.10 ceiling.
3. Minimize signed ratio: -delta_sr2 / delta_sr1.

Zero Class 2 loss scores zero. A Class 2 improvement produces a negative score.
Class 2 gain is not clipped to zero. Therefore negative scores rank ahead of zero,
and zero ahead of positive scores. A larger relative Class 2 improvement can win
even if its Class 1 gain is smaller (as long as the gain is >=5pp). This is the
literal signed ratio rule, not a rule that first maximizes Class 1 gain among
no-loss cells. Exact ties favor larger Class 1 gain, larger Class 2 gain, greater
utilization, lower Q fraction, longer H, then lower W.

The pipeline retains a choice for each tested budget. The 10pp-budget row is the
best ratio within the overall ceiling; other budget rows support sensitivity
analysis. These are not the old maximum-Class-1-gain frontier choices.

The ratio and eligibility are computed on discovery seeds only. Independent
20-seed evaluation checks the frozen cells; it does not select replacements.
Evaluation outputs include `evaluated_loss_per_gain` (only if evaluated C1 gain
is positive) and `evaluated_c1_target_met`. A case can fail its >=5pp independent
target and must then be reported as a failure, not removed or replaced silently.

## Pareto rule

The Pareto catalog still freezes ALL discovery cells with dSR1>=0, dSR2>=0 and
at least 0.5pp gain in either class. The user's >=5pp requirement applies to the
ratio/trade-off selection, not to this original Pareto qualifying definition.
With the horizon-only option, both-flexible Pareto includes eligible Q=0 cells.
The same option must be used at catalog, inventory, evaluation and postprocess.
Freeze metadata prevents accidentally changing the grid later.

## Sync and start a new Grid root

Changed files for this revision:

- analysis/experiment2_frontier.py
- analysis/experiment2_pareto_completion.py
- jobs/run_experiment2_frontier.sh
- jobs/run_experiment2_pareto_completion.sh
- tests/test_experiment2_frontier.py
- docs/experiment2_ratio_selection.md

Push only these files, then fetch on the Grid. The current Grid worktree is detached;
check its status is clean before `git merge --ff-only origin/main`. If that fails,
stop rather than resetting any files. Run the two test modules again (10 tests).

In the Grid completion checkout, retain REPO/PYTHON/PYTHONPATH and set:

```bash
export E2_OLD_COMPLETION="/scratch/$USER/experiment2_completion_20261102/frontier"
export E2_ROOT="/scratch/$USER/experiment2_ratio_horizon/frontier"
export E2_PARETO_ROOT="/scratch/$USER/experiment2_ratio_horizon/pareto"
export E2_SOURCES="$E2_ROOT/reuse_sources.json"
export INCLUDE_SHARE60=1
export ALLOW_HORIZON_ONLY=1
export E2_SELECTION_RULE=min_loss_per_gain
STAGE=bank bash jobs/run_experiment2_frontier.sh
```

Create a reuse manifest from the current completion freeze, also referencing all
newly completed discovery runs. Missing local evaluation directories are omitted
because no new evaluation runs have been submitted there yet; the inherited
manifest retains all earlier available evaluation sources.

```bash
"$PYTHON" - <<'PY'
import json, os
from pathlib import Path
from analysis.experiment2_frontier import source_catalog
old=Path(os.environ['E2_OLD_COMPLETION'])
p=Path(os.environ['E2_SOURCES'])
spec=json.loads((old/'freeze.json').read_text())['source_manifest']
spec += [dict(bank=str(old/'bank.csv'),search=[str(old/'search/raw')],
    evaluation=[str(old/'evaluation/raw')] if (old/'evaluation/raw').is_dir() else [],
    simulation_convention_verified=True)]
if p.exists():
    assert json.loads(p.read_text())==spec, 'Existing manifest differs; inspect first'
else:
    p.write_text(json.dumps(spec,indent=2))
source_catalog(p)
print(p)
PY
qsub -sync y -cwd -V -N e2ratio_inventory -v STAGE=inventory jobs/run_experiment2_frontier.sh
```

Send the resulting search coverage totals before submitting additional discovery
runs. All existing H/Q/W/seed results are reused by simulation-input signature.
Do not run selection until any newly required horizon-only discovery seeds finish.
Then freeze the new ratio selection, inventory missing evaluation, and proceed
using the existing completion runbook. Continue exporting both new option
variables for every stage; old root/grid/rule combinations fail closed.

No Grid jobs were submitted by Codex for this revision. Current plots and results
have not been overwritten. The old no-reservation-free search remains reproducible
when neither new option is set.
