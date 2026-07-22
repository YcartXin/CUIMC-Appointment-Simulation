# H1 patient-group and objective postprocessing

This update extracts class-specific outcomes from the completed H1 raw shards. It does **not** run new simulations.

## Files and repository locations

Copy these files into the repository:

```text
CUIMC-Appointment-Simulation/
├── analysis/
│   └── h1_postprocess_policy_outcomes.py
├── tests/
│   └── test_h1_postprocess_policy_outcomes.py
└── jobs/
    └── run_h1_policy_postprocess.sh
```

## What it produces

For each of `strict` and `release`:

- `selected_policy_seed_outcomes.csv`: seed-level outcomes for the four policies selected under both objectives.
- `selected_policy_outcomes.csv`: mean outcomes and selected `(horizon, Q, window)`.
- `pairwise_group_deltas.csv`: all six policy comparisons under both objectives, with Class 1 and Class 2 served-rate changes.
- `objective_switch_deltas.csv`: weighted-optimal minus average-optimal policy outcomes.
- `selection_validation.csv`: checks raw-recomputed optima against the uploaded summary files.
- `postprocess_summary.md`: basic row counts and validation notes.

Combined strict/release CSVs are also written at the output root.

## Local test

From the repository root:

```powershell
python -m unittest tests.test_h1_postprocess_policy_outcomes
```

Expected:

```text
Ran 3 tests
OK
```

## Grid setup

After moving the files to the Grid repository:

```bash
cd ~/projects/CUIMC-Appointment-Simulation
sed -i 's/\r$//' jobs/run_h1_policy_postprocess.sh
chmod +x jobs/run_h1_policy_postprocess.sh
bash -n jobs/run_h1_policy_postprocess.sh
```

Run the tests:

```bash
$HOME/.conda/envs/cuimc/bin/python -m unittest tests.test_h1_postprocess_policy_outcomes
```

## Submit the postprocessing job

```bash
mkdir -p ~/projects/CUIMC-Appointment-Simulation/grid_logs/h1_policy_postprocess
cd ~/projects/CUIMC-Appointment-Simulation/grid_logs/h1_policy_postprocess

grid_run --grid_submit=batch \
  --grid_ncpus=1 \
  --grid_mem=8G \
  "$HOME/projects/CUIMC-Appointment-Simulation/jobs/run_h1_policy_postprocess.sh"
```

Wait until `qstat` is empty, then check:

```bash
cat run_h1_policy_postprocess.sh.e*
tail -n 50 run_h1_policy_postprocess.sh.o*
```

No traceback should appear. The final output line should identify:

```text
full_run_summaries/h1_policy_outcomes
```

## Verify expected row counts

```bash
cd ~/projects/CUIMC-Appointment-Simulation

wc -l full_run_summaries/h1_policy_outcomes/strict/selected_policy_outcomes.csv
wc -l full_run_summaries/h1_policy_outcomes/release/selected_policy_outcomes.csv
```

Each should normally be `6721`: 840 backgrounds × 2 objectives × 4 policies, plus the header.

```bash
wc -l full_run_summaries/h1_policy_outcomes/strict/pairwise_group_deltas.csv
wc -l full_run_summaries/h1_policy_outcomes/release/pairwise_group_deltas.csv
```

Each should normally be `10081`: 840 backgrounds × 2 objectives × 6 comparisons, plus the header.

```bash
wc -l full_run_summaries/h1_policy_outcomes/strict/objective_switch_deltas.csv
wc -l full_run_summaries/h1_policy_outcomes/release/objective_switch_deltas.csv
```

Each should normally be `3361`: 840 backgrounds × 4 policies, plus the header.

## Download with WinSCP

Create an archive on the Grid:

```bash
cd ~/projects/CUIMC-Appointment-Simulation
mkdir -p ~/transfers

tar -czf ~/transfers/h1_policy_outcomes.tar.gz \
  full_run_summaries/h1_policy_outcomes
```

In WinSCP, download:

```text
/user/yy3694/transfers/h1_policy_outcomes.tar.gz
```

Extract the archive from the local repository root:

```powershell
tar -xzf .\h1_policy_outcomes.tar.gz
```

Do not delete the Grid raw shards yet.
