# Slide Update Materials — New Experiment Results

May 2026. Four new experiments (1350 total runs). This file lists what to
change or add per slide, the verified numbers, and the figure to use.

---

## Slide 4 — "No-show capacity is lost"

### What to correct

The bullet **"No-show behavior is the most direct behavioral driver of
utilization"** is true at stress-test demand (λ/S = 3.1) but does **not
hold at the realistic parameterization (λ/S = 1.2).**

**Experiment 2 finding:** Doubling no-show intensity at λ=24, S=20 moves
utilization from 0.935 to 0.934 — a difference of 0.001, i.e. statistically
inert. The no-show effect on utilization only activates above λ/S ≈ 1.4.

### Corrected bullet

> No-show drives utilization only when offered delays cross the no-show
> threshold — this requires high demand (λ/S ≥ ~1.4). At the realistic
> parameterization (λ/S = 1.2) the effect is negligible.

### Numbers

| Condition | Utilization at ns_scale=0 | Utilization at ns_scale=2× | Difference |
|-----------|--------------------------|---------------------------|------------|
| λ=20, S=20 (λ/S=1.0) | 0.903 | 0.903 | 0.000 |
| λ=24, S=20 (λ/S=1.2, realistic) | 0.935 | 0.934 | **0.001** |
| λ=28, S=20 (λ/S=1.4) | 0.957 | 0.900 | **0.057** |
| λ=32, S=20 (λ/S=1.6) | 0.978 | 0.864 | **0.114** |

### Figures

- `exp2_utilization_grid.png` — heatmap showing no-show effect on utilization
  across the demand × no-show grid
- `exp2_grid_panel_3up.png` — 3-panel: utilization + served rate + class gap

---

## Slide 5 — "Aggregate metrics can hide class-level access gaps"

### What to add

The 0.220 gap now has an **attribution**. Add a bullet with the breakdown.

### New numbers

Gap attribution (experiment 1, λ=24, S=20, H=28, 30 seeds):

| Source of asymmetry | Class gap | Share of full gap |
|---------------------|-----------|-------------------|
| Symmetric baseline (no gap) | −0.001 | — |
| + No-show asymmetry only | **+0.120** | **56 %** |
| + Cancellation asymmetry only | **+0.073** | **34 %** |
| + Balking threshold only | **+0.019** | **9 %** |
| All three together (realistic) | **+0.213** | 100 % |

All effects are near-additive — little compounding interaction.
This reverses the regression ranking: at the specific realistic
parameter values, no-show asymmetry is the dominant gap driver,
not cancellation.

### Corrected/new bullets

> - No-show asymmetry drives 56% of the class gap; cancellation 34%;
>   balking threshold 9%. Effects are near-additive.
> - Under symmetric behavior, arrival share has no effect on the class gap
>   (gaps within ±0.002 across 30%–70% class 1 share).

### Figures

- `exp1_gap_attribution_bar.png` — bar chart: gap by source of asymmetry
- `exp1_served_rates_by_config.png` — side-by-side: served rates + gap panel

---

## New slide candidate: Demand regime and no-show interaction

**Headline:** "No-show only bites when demand is high enough."

Three bullets:
- Below λ/S ≈ 1.4, no-show scale does not move utilization or served rate.
- Above λ/S ≈ 1.4, both utilization and the class gap amplify sharply.
- At λ=32 (λ/S=1.6): doubling no-show intensity drops utilization from 0.98 to
  0.86 and raises the class gap from 0.56 to 0.73.

**Figure:** `exp2_grid_panel_3up.png` (or use `exp2_utilization_grid.png` alone)

---

## New slide candidate: Horizon saturation

**Headline:** "The class gap saturates at a 3-week booking horizon."

Three bullets:
- Class gap grows from 0.167 (H=7) to 0.213 (H=21), then is flat.
- Extending the horizon beyond 21 days does not amplify the gap further.
- A short horizon (H=7) partially suppresses the gap by limiting how far
  out cancellation slots can compound.

**Figure:** `exp3_gap_vs_horizon.png`

Numbers:

| Horizon | Class gap | ±95% CI |
|---------|-----------|---------|
| 7 days  | 0.167 | 0.003 |
| 14 days | 0.203 | 0.003 |
| 21 days | 0.213 | 0.005 |
| 28 days | 0.213 | 0.005 |
| 42 days | 0.213 | 0.005 |

---

## What does NOT need to change

| Slide | Status |
|-------|--------|
| Slide 1 — Metric definitions | Unchanged — numbers still correct |
| Slide 2 — Capacity pressure (λ=100, S=32, util=0.841, served=0.269) | Unchanged — numbers verified |
| Slide 3 — Balking reallocates | Unchanged. Can add: "balking alone contributes only 9% of the realistic gap." |

---

## All new figures (in `figures/`)

| File | What it shows | Best use |
|------|--------------|----------|
| `exp1_gap_attribution_bar.png` | Gap by source of asymmetry (bar, single panel) | Slide 5 or new slide |
| `exp1_served_rates_by_config.png` | C1/C2 served rates + gap panel (2 panels) | Slide 5 backup |
| `exp2_utilization_grid.png` | Utilization heatmap over λ × no-show grid | Slide 4 correction |
| `exp2_served_rate_grid.png` | Served rate heatmap over λ × no-show grid | Slide 4 or new |
| `exp2_class_gap_grid.png` | Class gap heatmap over λ × no-show grid | New slide |
| `exp2_grid_panel_3up.png` | All three metrics in one 3-panel figure | New slide (single figure) |
| `exp3_gap_vs_horizon.png` | Class gap vs booking horizon (line) | New slide |
| `exp3_metrics_by_horizon.png` | Util + served rate + gap vs horizon (3 panels) | Horizon slide |
| `exp4_gap_vs_arrival_share.png` | Gap vs arrival share under symmetric behavior | Slide 5 or appendix |
| `exp4_served_rates_vs_share.png` | C1/C2 served rates + gap vs share (2 panels) | Appendix |

---

## Source files for all numbers

| Number | File |
|--------|------|
| Gap attribution (exp1) | `outputs/exp1_asymmetry_attribution/summary/summary.csv` |
| Demand × no-show grid (exp2) | `outputs/exp2_demand_noshow_grid/summary/grid_summary.csv` |
| Horizon sweep (exp3) | `outputs/exp3_horizon_sweep/summary/summary.csv` |
| Arrival share (exp4) | `outputs/exp4_arrival_share_symmetric/summary/summary.csv` |
