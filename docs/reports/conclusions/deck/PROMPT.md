# Briefing: FCFS Appointment Simulation

## The project in one paragraph

We built a two-class discrete-event simulation of a first-come-first-served outpatient booking system. Patients arrive daily, are offered the earliest open slot in a rolling horizon, and then either accept and book, or balk (reject a long delay). Booked patients can later cancel or no-show. Two patient classes share one booking pool with no explicit priority. The goal was to understand how behavioral parameters interact with capacity pressure to produce aggregate and class-level access outcomes — and to establish a minimal set of metrics that cannot be gamed by reporting only utilization.

---

## What this package contains

```
deck/
├── fcfs_conclusions_deck.pdf        ← 5-slide meeting deck (16:9, compiled)
├── fcfs_conclusions_deck.tex        ← LaTeX beamer source
├── fcfs_conclusions_deck_audit.md   ← every number traced to a source CSV
└── figures/  (13 PNGs, prefixed s1_…s5_ by slide, bg_ for background)
```

---

## The argument (read this before opening the slides)

The story has three layers that build on each other.

**Layer 1 — The metric problem.** Utilization and access measure different things. Utilization counts completed visits per available slot; it tells you how full the schedule is. Served rate counts completed visits per arriving patient; it tells you how many people who tried to get an appointment actually got one. Under capacity pressure these two numbers diverge sharply. At the stress-test baseline (λ = 100 arrivals/day, S = 32 slots/day), utilization is 0.841 while served rate is 0.269. High slot fill, poor patient access. You cannot summarize this system with one number.

**Layer 2 — What drives each metric.** Behavior shapes outcomes, but different behaviors hit different metrics through different channels.

- No-show is the primary utilization driver. A no-show removes a completed visit without freeing the slot for someone else. The regression screen (240 randomized configurations, OLS with HC3 errors) finds no-show threshold and step are the two strongest predictors of utilization (|β| = 0.51, 0.42), dwarfing all other behavioral parameters.
- Demand is the primary access driver. Total arrival rate has by far the largest effect on served rate (|β| = 0.77), roughly 3× larger than the next driver. At 3:1 overload, the structural ceiling on served rate is 32/100 = 0.32 before any behavioral losses — behavioral parameters can only reduce it from there.
- Balking reallocates, it does not recover. Suppressing all balking (B_no_balking scenario) leaves utilization and served rate almost unchanged (Δ ≈ −0.001). Losses are relabeled — no-offer rises from 0 to 0.297, cancellation rises — but total completions do not improve. Under FCFS, balking is a reallocation mechanism between classes, not a capacity recovery channel.
- Cancellation releases future slots; no-show does not. A canceled booking returns a slot to the horizon. A no-show does not trigger rebooking. These are fundamentally different behavioral channels, and conflating them leads to wrong intuitions about overbooking and scheduling policy.

**Layer 3 — Class-level gaps.** Symmetric parameters produce a near-zero class gap (−0.000354 in the baseline). Asymmetric behavioral parameters create substantial gaps under pooled FCFS even with no explicit class priority. In the realistic parameterization (λ = 24, S = 20, 28-day horizon, MRI-like vs behavioral-health classes), the served-rate gap reaches 0.220 (Class 1: 0.881, Class 2: 0.661). This is mechanically consistent with three compounding advantages for Class 1: higher balking tolerance (threshold 21 vs 14 days), lower baseline no-show probability (1% vs 15%), and lower cancellation rate (1% vs 2%). The regression confirms all three gap dimensions are significant drivers of the class access advantage (|β| = 0.45 for cancellation gap, 0.43 for balking threshold gap, 0.27/0.21 for no-show gaps).

---

## What we know and what we don't

### Established

- Metric definitions and why they diverge under oversubscription.
- Ranking of behavioral drivers for utilization, served rate, offered wait, and class gap (from the regression screen).
- No-balking diagnostic: removing balking relabels losses, does not recover capacity.
- Realistic-scenario class gap: 0.220, mechanically consistent with three compounding behavioral asymmetries.

### Not yet established — experiments that could sharpen the conclusions

**Experiment 1 — Asymmetry attribution (highest value).**
The realistic scenario has three sources of class asymmetry simultaneously. We know their regression coefficients in isolation, but not their joint contribution to the 0.220 gap specifically. Run a factorial: start from a symmetric version of the realistic scenario, then add each asymmetry one at a time and together.

| Run | Cancellation | Balking threshold | No-show |
|-----|-------------|------------------|---------|
| Symmetric | φ₁=φ₂=0.015 | θ₁=θ₂=17.5 | ξ₁=ξ₂=matched |
| +Cancel gap only | φ₁=0.01, φ₂=0.02 | symmetric | symmetric |
| +Balk gap only | symmetric | θ₁=21, θ₂=14 | symmetric |
| +No-show gap only | symmetric | symmetric | asymmetric |
| +All three | realistic | realistic | realistic |

This decomposes the 0.220 gap by mechanism and tells us which lever a planner should adjust first.

**Experiment 2 — Demand × no-show interaction.**
No-show is the strongest utilization driver at stress-test demand (λ=100, S=32). At plausible demand (λ=24, S=20), the no-show effect could dominate even more (less slack to absorb losses) or less (more cancellation recovery). Run a 2D grid over (λ/S ratio) × (no-show step) to find where no-show becomes the binding constraint vs where demand pressure swamps everything else.

**Experiment 3 — Horizon length and the cancellation compounding effect.**
The realistic scenario uses H=28 vs H=14 in the stress test. A longer horizon gives more time for cancellations to free slots — which disproportionately benefits classes with lower cancellation rates. Run a sweep over H ∈ {7, 14, 21, 28, 42} days holding all other realistic parameters fixed. If the class gap grows with H, it means horizon length is a policy lever that affects equity, not just throughput.

**Experiment 4 — Arrival share at symmetric behavior.**
The regression finds that Class 1's own arrival share negatively predicts its access advantage (β = −0.12, p = 0.009). This is counterintuitive — more arrivals, smaller advantage. It could mean that at higher C1 volume, C1 crowds out its own late-horizon slots (leaving nothing for C2 to lose). Run a sweep over class 1 arrival share ∈ {0.3, 0.4, 0.5, 0.6, 0.7} at symmetric behavioral parameters to isolate this effect.

---

## Open meeting decision

Choose the reference case for future analysis:

- **Stress-test baseline** (λ=100, S=32, 14-day horizon): overloaded, symmetric, good for isolating mechanisms cleanly. Not a clinic estimate.
- **Realistic scenario** (λ=24, S=20, 28-day horizon, asymmetric classes): operationally interpretable, parameters are illustrative but plausible. The class gap result lives here.

The two scenarios answer different questions. If the next step is to study a specific policy intervention (overbooking, priority rules, horizon extension), the realistic scenario is the right base. If the next step is theoretical mechanism work, the stress-test baseline is cleaner.

---

## Key file paths in the repo

| File | What it contains |
|------|-----------------|
| `configs/baseline.yaml` | Stress-test parameters |
| `configs/realistic.yaml` | Realistic scenario parameters |
| `docs/reports/metric_analysis/data/baseline_summary.csv` | 30-seed baseline aggregate results |
| `results/balking_effect_verification.csv` | A/B/C/D scenario comparison (30 seeds each) |
| `docs/reports/metric_analysis/data/regression_standardized_coefficients.csv` | OLS regression screen, all standardized β with HC3 CIs |
| `docs/reports/metric_analysis/research_style/final/note/fcfs_research_note_final.pdf` | Full technical note |
| `docs/reports/metric_analysis/research_style/final/simulation_realistic/fcfs_realistic_comparison.pdf` | Realistic scenario note |
| `docs/reports/conclusions/paper/fcfs_conclusion_sheet.pdf` | 2-page meeting memo |
| `simulation/engine.py` | Core simulation (bookings, cancellations, no-shows) |
| `experiments/sweep_class_1_balking.py` | Template for parameter sweep experiments |
