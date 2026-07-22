# H1 average-utilization update: installation and Grid run plan

This package updates H1 so the policy search can optimize either:

- `weighted_utilization` (the existing default, retained for backward compatibility), or
- `average_utilization` (the new primary objective).

For `average_utilization`, classification writes all six policy comparisons and stores the results separately under `summary_average_utilization`, leaving the existing weighted-objective summaries unchanged.

## 1. Put the files in these repository locations

Copy the package contents into the root of your local repository, preserving the folder structure:

```text
CUIMC-Appointment-Simulation/
├── experiments/
│   └── h1_short_horizon_reservation.py   # replace the existing file
├── tests/
│   └── test_h1_average_objective.py      # add this new file
└── jobs/
    ├── run_h1_average_refine.sh           # add this new file
    └── classify_h1_average.sh             # add this new file
```

Do not replace or regenerate:

```text
outputs/hypotheses/background_scenarios.csv
```

Do not delete the completed Grid raw results in:

```text
/scratch/yy3694/h1_short_horizon_reservation_10seed_v2
```

The supplemental run reuses those results and adds only missing fine-search cells selected by `average_utilization`.

## 2. Test locally

From the repository root:

```bash
python -m py_compile experiments/h1_short_horizon_reservation.py
python -m unittest tests.test_h1_average_objective
python -m unittest discover -s tests -p 'test_h1_short_horizon_reservation.py'
```

Confirm the new CLI option:

```bash
python experiments/h1_short_horizon_reservation.py --help
```

You should see:

```text
--objective {average_utilization,weighted_utilization}
```

Optional local smoke test:

```bash
python experiments/h1_short_horizon_reservation.py all \
  --variant strict \
  --objective average_utilization \
  --smoke \
  --n-seeds 2 \
  --workers 2 \
  --output-dir outputs/hypotheses/h1_average_smoke
```

The smoke summary should appear under:

```text
outputs/hypotheses/h1_average_smoke/strict/summary_average_utilization/
```

## 3. Commit and push

```bash
git add experiments/h1_short_horizon_reservation.py \
        tests/test_h1_average_objective.py \
        jobs/run_h1_average_refine.sh \
        jobs/classify_h1_average.sh

git commit -m "Optimize H1 policies by average utilization"
git push origin main
```

## 4. Pull and test on the CBS Grid

```bash
cd ~/projects/CUIMC-Appointment-Simulation
git pull --ff-only
chmod +x jobs/run_h1_average_refine.sh jobs/classify_h1_average.sh
```

Verify the CLI:

```bash
$HOME/.conda/envs/cuimc/bin/python \
  experiments/h1_short_horizon_reservation.py --help | grep objective
```

Run the tests:

```bash
$HOME/.conda/envs/cuimc/bin/python -m unittest \
  tests.test_h1_average_objective

$HOME/.conda/envs/cuimc/bin/python -m unittest discover \
  -s tests -p 'test_h1_short_horizon_reservation.py'
```

Do not submit Grid jobs unless both test commands finish with `OK`.

## 5. Create a separate Grid log directory

```bash
mkdir -p ~/projects/CUIMC-Appointment-Simulation/grid_logs/h1_average_refine
cd ~/projects/CUIMC-Appointment-Simulation/grid_logs/h1_average_refine
```

## 6. Submit two pilot refinement jobs

```bash
grid_run --grid_submit=batch --grid_ncpus=12 --grid_mem=16G \
  "$HOME/projects/CUIMC-Appointment-Simulation/jobs/run_h1_average_refine.sh" \
  strict 0 20 12
```

```bash
grid_run --grid_submit=batch --grid_ncpus=12 --grid_mem=16G \
  "$HOME/projects/CUIMC-Appointment-Simulation/jobs/run_h1_average_refine.sh" \
  release 0 20 12
```

Check status:

```bash
qstat
```

Inspect output and errors:

```bash
tail -n 40 run_h1_average_refine.sh.o*
cat run_h1_average_refine.sh.e*
```

Expected behavior:

- Batch 1 should generally report `to run=0`, because baseline, horizon-only, and coarse cells already exist.
- Batch 2 may report `to run=0` or a positive number, depending on whether the average-utilization coarse winner needs additional fine cells.
- Error files should be empty.

## 7. Submit the other 38 jobs

After both pilot jobs look healthy:

```bash
for i in $(seq 1 19); do
  grid_run --grid_submit=batch --grid_ncpus=12 --grid_mem=16G \
    "$HOME/projects/CUIMC-Appointment-Simulation/jobs/run_h1_average_refine.sh" \
    strict "$i" 20 12

  grid_run --grid_submit=batch --grid_ncpus=12 --grid_mem=16G \
    "$HOME/projects/CUIMC-Appointment-Simulation/jobs/run_h1_average_refine.sh" \
    release "$i" 20 12
done
```

Do not add `--no-resume`.

## 8. Verify the refinement jobs

When `qstat` is empty:

```bash
cd ~/projects/CUIMC-Appointment-Simulation/grid_logs/h1_average_refine
```

Count successful completion markers:

```bash
grep -l "Raw results (sharded):" run_h1_average_refine.sh.o* | wc -l
```

Expected:

```text
40
```

Search for failures:

```bash
grep -HnE "Traceback|MemoryError|Killed|Error" \
  run_h1_average_refine.sh.o* \
  run_h1_average_refine.sh.e* 2>/dev/null
```

Ideally, nothing prints.

## 9. Run average-utilization classification

From the same log directory:

```bash
grid_run --grid_submit=batch --grid_mem=16G \
  "$HOME/projects/CUIMC-Appointment-Simulation/jobs/classify_h1_average.sh" \
  strict
```

```bash
grid_run --grid_submit=batch --grid_mem=16G \
  "$HOME/projects/CUIMC-Appointment-Simulation/jobs/classify_h1_average.sh" \
  release
```

After `qstat` is empty, verify:

```bash
grep -H "Backgrounds classified" classify_h1_average.sh.o*
```

Both logs should report:

```text
Backgrounds classified: 840
```

Check error files:

```bash
find . -name 'classify_h1_average.sh.e*' -size +0 -print
```

## 10. Copy the new summaries into permanent home storage

```bash
mkdir -p \
  ~/projects/CUIMC-Appointment-Simulation/full_run_summaries/h1_10seed_average
```

```bash
cp -r \
  /scratch/$USER/h1_short_horizon_reservation_10seed_v2/strict/summary_average_utilization \
  ~/projects/CUIMC-Appointment-Simulation/full_run_summaries/h1_10seed_average/strict
```

```bash
cp -r \
  /scratch/$USER/h1_short_horizon_reservation_10seed_v2/release/summary_average_utilization \
  ~/projects/CUIMC-Appointment-Simulation/full_run_summaries/h1_10seed_average/release
```

Each variant should contain:

```text
condition_optima.csv
condition_deltas.csv
h1_summary.md
```

The prior weighted-objective results remain under:

```text
full_run_summaries/h1_10seed/
```

## What the new average-utilization output contains

For each background, `condition_optima.csv` selects the best policy under:

```text
average_utilization = completed appointments / measured appointment capacity
```

The four regimes are:

1. `baseline`: native horizon, no reservation;
2. `horizon_only`: optimized horizon, no reservation;
3. `reservation_only`: native fixed horizon, optimized reservation quantity and window;
4. `both_flexible`: jointly optimized horizon, reservation quantity, and reservation window.

`condition_deltas.csv` contains these six comparisons:

```text
horizon_only_vs_baseline
reservation_only_vs_baseline
both_flexible_vs_baseline
both_flexible_vs_horizon_only
both_flexible_vs_reservation_only
reservation_only_vs_horizon_only
```
