# Experiment 2 expansion: discovery-frozen access frontier

Implemented additively in the local repository: `analysis/experiment2_frontier.py`, `jobs/run_experiment2_frontier.sh`, and `tests/test_experiment2_frontier.py`. Transfer these three files to the same paths on Grid. No simulation jobs have been submitted.

## Design and policy decisions

The complete design has **972 backgrounds**: six rho values (1.2/1.4/1.7/2/2.5/3), six shares (.1/.3/.5/.7/.8/.9), three capacities (30/40/50), and nine existing behavior profiles. The 540 original bank IDs and 54 share80 follow-up IDs are preserved; 378 additional bank identities are generated. These are bank counts, not claims about completed raw simulations. An existing background can still lack an entire policy family or H=27–30 cells.

Use the later Experiment 2 systematic coarse grid across **every background**, including previously negative backgrounds:

- Reservation-only: H=100, Q=5,10,...,capacity; historical window grid 1,3,...,25,26, with W=1 excluded as in the genuine-reservation catalog.
- Both-flexible: H=2..30; the same Q grid; historical odd-window grid plus upper endpoint, capped at min(H,26), retaining W>=2.
- Baseline: H=100, Q=0, W=-1.

The general factorial runner's H=2..26 differs from the later Experiment 2 H=2..30. This expansion fills the latter grid. It does not add integer-Q/window exhaustive searches or outcome-dependent fine refinements. The old refinements are uneven and were driven by different objectives. Existing raw rows may supply a required coarse cell even if their recorded arm differs; off-grid cells are not eligible for the main comparison. Thus “full policy space” here means the **full existing systematic Experiment 2 coarse grid**, not every mathematically possible integer policy. There is no restriction to prior Pareto cells or backgrounds.

Five discovery seeds: 1000..1004. For each background × policy × L in {0,.005,.01,.02,.03,.04,.05,.075,.10}, maximize mean paired dSR1 subject to dSR2>=-L. Ties: higher dSR2, higher utilization, smaller Q/capacity, longer H, then smaller W. All rates/budgets are proportions: .01 is one percentage point. Comparisons use 1e-12 numerical tolerance; objective ordering uses unrounded means.

No positive-gain threshold restricts the main optimizer. L=0 is the zero-C2-loss endpoint; flags distinguish actual nonharm to both classes and the legacy >=.005 meaningful-gain criterion. An infeasible budget gets one explicit placeholder row, no fabricated winner. Frozen winners are evaluated with paired baseline on seeds 2000..2019. Identical cells selected at several budgets run once per seed. Failed evaluation winners remain reported, with failed validation; they are never replaced. Practical dU>=-.005 is a secondary flag/summary only.

## 1. Initialize, inventory, and generate only new bank identities

The bank stage writes metadata only; no simulation runs. Use a new output root for a new selection campaign. Do not overwrite an existing frozen campaign.

```bash
cd "$HOME/projects/CUIMC-Appointment-Simulation"
export REPO="$PWD"
export PYTHON="$HOME/.conda/envs/cuimc/bin/python"
export E2_ROOT="/scratch/$USER/experiment2_frontier"
export E2_SOURCES="$REPO/outputs/hypotheses/experiment2_frontier_sources.json"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
mkdir -p "$E2_ROOT" outputs/hypotheses grid_logs/experiment2_frontier/{search,select,evaluate,postprocess}
bash -n jobs/run_experiment2_frontier.sh
"$PYTHON" -m unittest discover -s tests -p test_experiment2_frontier.py
STAGE=bank bash jobs/run_experiment2_frontier.sh
```

`bank.csv` contains the complete bank. `new_bank_entries.csv` contains only 378 new entries; the historical bank files are not modified.

The following source manifest covers the historical original/extended search roots and share80 follow-up. Confirm directory names against your Grid scratch inventory. These paths follow the checked-in job conventions; they could not be inspected remotely from this session. **A declared missing directory fails closed**. Correct the path, or remove an entry only when that source really is unavailable; otherwise its simulations would be counted as missing. Add any other raw archives before submitting work. Source banks must map the IDs used in their raw filenames to the original simulation inputs.

```bash
export PBF_SOURCE_ROOT="/scratch/$USER/patient_behavior_factorial_3_5"
export PBF_PARETO_ROOT="/scratch/$USER/patient_behavior_factorial_served_rate_pareto"
export PBF_FOLLOWUP_ROOT="/scratch/$USER/patient_behavior_factorial_share80"
"$PYTHON" - <<'PY'
import json, os
from pathlib import Path
repo=Path(os.environ['REPO'])
from experiments.patient_behavior_factorial_bank import generate_bank
from analysis.experiment2_share80_followup import bank_frame
banks=Path(os.environ['E2_ROOT'])/'source_banks'
banks.mkdir(exist_ok=True)
legacy=banks/'original_540.csv'
share=banks/'share80_54.csv'
generate_bank().to_csv(legacy,index=False)
bank_frame().to_csv(share,index=False)
catalog='served_rate_pareto/pareto_catalog_5seed/independent_evaluation/raw'
spec=[]
for variable,bank in [('PBF_SOURCE_ROOT',legacy),('PBF_PARETO_ROOT',legacy),('PBF_FOLLOWUP_ROOT',share)]:
    root=Path(os.environ[variable])
    search=root/'search/raw'
    if not search.is_dir(): raise SystemExit(f'Locate required raw source first: {search}')
    evaluation=[str(p) for p in [root/catalog,root/'evaluation/raw'] if p.is_dir()]
    spec.append(dict(bank=str(bank),search=[str(search)],evaluation=evaluation,
                     simulation_convention_verified=True))
Path(os.environ['E2_SOURCES']).write_text(json.dumps(spec,indent=2))
print(Path(os.environ['E2_SOURCES']).read_text())
PY
STAGE=inventory bash jobs/run_experiment2_frontier.sh
STAGE=search DRY_RUN=1 bash jobs/run_experiment2_frontier.sh
```

The manifest's `simulation_convention_verified` assertion applies to these inspected pipeline conventions: burn-in=30, measurement=365, cooldown=max(14,H), fixed uncapped thresholds, same-day cancellation/release. Do not add smoke runs or other engine/config versions under that assertion. Matching uses all background simulation inputs, not just IDs. Reuse requires complete finite key/metric rows; conflicting duplicate results abort rather than silently choosing one. Historical summary CSVs cannot replace seed-level raw data.

Inventory produces `inventory/search_coverage.csv` (planned/reused/pending seed runs by background), `inventory/search_missing/*.csv` (exact missing policy-cell-seed keys), and `inventory/search_shards.txt` (stable 1-based background indices with pending work). It reads existing results without copying them or mutating them. A full no-source run is large; review pending counts before submission. The inventory manifest itself may be large because it explicitly lists missing seed runs.

## 2. Submit only missing search shards

Run this block deliberately after reviewing inventory. It submits contiguous ranges of pending IDs, with at most 20 concurrent tasks across this stage. Each array task owns one background and rechecks its pending keys. Do not run overlapping arrays against the same output root.

```bash
"$PYTHON" - <<'PY' > "$E2_ROOT/submit_missing_search.sh"
import os
from pathlib import Path
ids=sorted(map(int,(Path(os.environ['E2_ROOT'])/'inventory/search_shards.txt').read_text().split()))
ranges=[]
for i in ids:
    if ranges and i==ranges[-1][1]+1: ranges[-1][1]=i
    else: ranges.append([i,i])
print('#!/usr/bin/env bash\nset -euo pipefail')
for lo,hi in ranges:
    # -sync y serializes disjoint ranges so total concurrency stays <=20.
    print(f'qsub -sync y -N e2f_search -cwd -V -t {lo}-{hi} -tc 20 '
          '-v STAGE=search -o grid_logs/experiment2_frontier/search '
          '-e grid_logs/experiment2_frontier/search jobs/run_experiment2_frontier.sh')
PY
bash "$E2_ROOT/submit_missing_search.sh"
STAGE=inventory bash jobs/run_experiment2_frontier.sh
STAGE=search DRY_RUN=1 bash jobs/run_experiment2_frontier.sh
```

Require zero pending search runs. After failures, inspect logs, rerun inventory and regenerate the submission script; completed cells are skipped, including those flushed before a task stopped. `-sync y` may keep the submission shell waiting; run it in your usual persistent login session. Nothing in this package automatically submits jobs.

## 3. Select and freeze discovery winners

```bash
qsub -sync y -N e2f_select -cwd -V -v STAGE=select \
  -o grid_logs/experiment2_frontier/select -e grid_logs/experiment2_frontier/select \
  jobs/run_experiment2_frontier.sh
STAGE=inventory bash jobs/run_experiment2_frontier.sh
STAGE=evaluate DRY_RUN=1 bash jobs/run_experiment2_frontier.sh
```

Selection refuses incomplete search. It writes 17,496 rows (972 × 2 × 9), including infeasible placeholders, in `selected_discovery.csv`; exact discovery nondominated curves in `discovery_nondominated.csv`; and `legacy_rule_discovery_best.csv` using the prior meaningful-Pareto discovery eligibility rule. `freeze.json` records seeds, source manifest, grid, tie rule, and SHA256 hashes of selection/bank. Re-selection in the same root is refused. Evaluation/postprocessing check the hashes. Do not edit frozen files; use a new root for methodological changes.

## 4. Evaluate only selected cells and matched baseline

```bash
"$PYTHON" - <<'PY' > "$E2_ROOT/submit_missing_evaluation.sh"
import os
from pathlib import Path
ids=sorted(map(int,(Path(os.environ['E2_ROOT'])/'inventory/evaluation_shards.txt').read_text().split()))
ranges=[]
for i in ids:
    if ranges and i==ranges[-1][1]+1: ranges[-1][1]=i
    else: ranges.append([i,i])
print('#!/usr/bin/env bash\nset -euo pipefail')
for lo,hi in ranges:
    print(f'qsub -sync y -N e2f_eval -cwd -V -t {lo}-{hi} -tc 20 '
          '-v STAGE=evaluate -o grid_logs/experiment2_frontier/evaluate '
          '-e grid_logs/experiment2_frontier/evaluate jobs/run_experiment2_frontier.sh')
PY
bash "$E2_ROOT/submit_missing_evaluation.sh"
STAGE=inventory bash jobs/run_experiment2_frontier.sh
STAGE=evaluate DRY_RUN=1 bash jobs/run_experiment2_frontier.sh
```

Require zero pending evaluation runs. Reused independent evaluation rows must have seeds 2000..2019 and matched inputs. Evaluation-source outcomes never enter selection. No evaluation is run for infeasible discovery placeholders. An empty submission script is a successful no-op.

## 5. Postprocess

```bash
qsub -sync y -N e2f_report -cwd -V -v STAGE=postprocess \
  -o grid_logs/experiment2_frontier/postprocess -e grid_logs/experiment2_frontier/postprocess \
  jobs/run_experiment2_frontier.sh
```

Postprocessing refuses incomplete selected-cell/baseline evaluation. Outputs under `$E2_ROOT`:

- `evaluated_selected_cells.csv`: one row per background × policy × budget, paired dSR1/dSR2/dU, H/Q/W/Q-per-slot, discovery feasibility, evaluation feasibility/completeness, validation, strict/meaningful Pareto, practical-frontier flags, and paired bootstrap intervals (2,000 draws, RNG 20260928). Validation means complete independent evaluation and mean C2 loss within budget; C1 gain and statistical C2 support are separately observable.
- `evaluated_seed_deltas.csv`: matched seed deltas for reproducible uncertainty analysis. A cell may be repeated across budget rows for reporting; it was simulated once.
- `summaries/`: means/medians and counts by policy/budget, rho, share, capacity, no-show, balking, and full joint strata. Every background gets one vote within a stratum/budget. Main means include all evaluated discovery winners, including evaluation failures; separate `validated_by_*` and `practical_by_*` tables provide clearly conditional summaries. Infeasible placeholders contribute to denominators and not numeric means.
- `budget_curve_knee_inputs.csv`: frozen discovery and evaluated curves, gain increments and gain per budget increment. Use discovery curves for locating exploratory knees; evaluated curves test those fixed choices and need not be monotone. Do not optimize a new winner or claim an independently validated knee using evaluation outcomes.
- `strict_endpoint_comparison.csv`: the new L=0 results. Join to `legacy_rule_discovery_best.csv` on background_id/policy to distinguish the old meaningful-gain screen from the new zero-loss optimizer. The old validated-cell CSVs can be joined on background_id/policy/H/Q/W to compare the same cells. If presenting retrospective old “best evaluated cell” summaries, label that evaluation-driven choice explicitly; it is not the new discovery-frozen estimator. Restrict historical comparisons to common backgrounds and grid coverage (40 slots and most share80 backgrounds have no historical comparator).

Local validation uses synthetic raw outcomes only and does not run the simulation engine. It checks the 972-background design, retained IDs, grids, tie ordering, alias-based reuse, conflicting duplicate rejection, completeness gates, immutable freeze, budget feasibility, and evaluation failure without reselection. The Grid source inventory and Bash/Grid execution remain to be verified on Grid.
