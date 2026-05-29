# Context prompt for Claude cowork

## What this project is

Two-class discrete-event simulation of a first-come-first-served (FCFS) outpatient appointment system. Patients arrive daily, are offered the earliest available slot in a rolling booking horizon, and can balk (reject a long delay), cancel before the appointment, or no-show. Two patient classes share the same booking pool with no explicit priority. The simulation is written in Python and lives in a git repo.

The simulation was built to study how behavioral parameters (balking, cancellation, no-show) interact with capacity pressure to produce aggregate and class-level access outcomes. The stress-test baseline uses λ = 100 arrivals/day against S = 32 slots/day (3:1 overload) to isolate mechanisms. A realistic parameterization (λ = 24, S = 20, 1.2:1 ratio, asymmetric classes) was run separately for operational interpretation.

---

## What this package contains

```
deck/
├── fcfs_conclusions_deck.pdf        ← 5-slide meeting deck (16:9, compiled)
├── fcfs_conclusions_deck.tex        ← LaTeX beamer source
├── fcfs_conclusions_deck_audit.md   ← every number traced to a source CSV
├── README.md                        ← slide–figure map and number table
└── figures/   (13 PNGs)
    ├── s1_*   Slide 1 — utilization vs served rate
    ├── s2_*   Slide 2 — capacity stress
    ├── s3_*   Slide 3 — balking effect
    ├── s4_*   Slide 4 — no-show effect
    ├── s5_*   Slide 5 — class access gap
    └── bg_*   Background: regression driver chart, exec summary panel
```

All numbers in the deck have been verified against source CSVs. See `fcfs_conclusions_deck_audit.md` for the full trace.

---

## The five conclusions (one per slide)

1. **Utilization alone is not an access summary.** Utilization = completed visits per slot; served rate = completed visits per arrival. Always report both, plus offered wait, no-offer share, outcome decomposition, and class-level served rates.

2. **Under high demand, high utilization can coexist with poor access.** At the stress-test baseline (λ = 100, S = 32), utilization ≈ 0.841 while served rate ≈ 0.269. The 3:1 overload is deliberate — it is a mechanism-diagnosis tool, not a calibrated clinic model.

3. **Balking reallocates access; it does not recover aggregate capacity.** When Class 1 balks more, its own served rate falls and Class 2 gains through the shared pool. Removing balking entirely does not improve total completions — it relabels losses from "balked" to "no-offer" (rises to ~0.297) and raises cancellation.

4. **Cancellation can be reabsorbed; no-show capacity is lost.** Cancellations return future slots to the pool; other patients can fill them. No-show slots are not rebooked. No-show rate is the strongest behavioral driver of utilization in the regression screen.

5. **Aggregate metrics can hide class-level access gaps.** Symmetric parameters yield a class gap near zero (−0.000354 in the baseline). Under the realistic asymmetric parameterization the served-rate gap reaches 0.220 (Class 1: 0.881, Class 2: 0.661), entirely through compounding pipeline advantages under pooled FCFS — not explicit priority.

---

## Open meeting decision

Choose the reference case going forward:
- **Stress-test baseline** (λ = 100, S = 32): good for isolating mechanism behavior; not a clinic estimate.
- **Realistic scenario** (λ = 24, S = 20, asymmetric classes): good for operational interpretation; parameters are illustrative, not empirically calibrated.

---

## Key file paths in the full repo (not in this package)

| File | What it contains |
|------|-----------------|
| `configs/baseline.yaml` | Stress-test parameters |
| `configs/realistic.yaml` | Realistic scenario parameters |
| `docs/reports/metric_analysis/data/baseline_summary.csv` | 30-seed baseline aggregate results |
| `results/balking_effect_verification.csv` | A/B/C/D scenario comparison (30 seeds each) |
| `docs/reports/metric_analysis/data/regression_standardized_coefficients.csv` | OLS regression screen, standardized β |
| `docs/reports/metric_analysis/research_style/final/note/fcfs_research_note_final.pdf` | Full technical note |
| `docs/reports/metric_analysis/research_style/final/simulation_realistic/fcfs_realistic_comparison.pdf` | Realistic scenario comparison note |
| `docs/reports/conclusions/paper/fcfs_conclusion_sheet.pdf` | 2-page meeting memo |
