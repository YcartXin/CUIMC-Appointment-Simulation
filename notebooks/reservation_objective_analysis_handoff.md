# Strict Reservation Objective-Function Analysis Handoff

This note summarizes the current strict Class 1 reservation analysis so it can be sent to a supervisor or another LLM for review. It is intended to give enough context to evaluate whether the objective functions, parameter choices, and planned plots make sense.

## Context

The simulation compares two appointment-booking policies:

- **Pooled FCFS:** all classes compete for the same daily capacity.
- **Strict Class 1 reservation:** `Q = reserved_slots_per_day` slots are protected for Class 1 each day. Class 1 can use protected slots first and then general slots. Class 2 can only use general slots. Unused protected slots remain empty.

The soft reservation / Class 1 priority backfill idea has been dropped from the current analysis. It remains only as a possible future direction in the TeX reservation note.

The current goal is to understand when strict Class 1 reservation has better values than FCFS, and how that depends on the policy parameter `Q`.

## Source Files

Main implementation and analysis files:

- `notebooks/10_reservation_objective_functions.ipynb`
- `notebooks/11_reservation_objective_sweeps.ipynb`
- `notebooks/12_reservation_experiment_playground.ipynb`
- `docs/reference/simulation_explanation/tex/reservation/class1_reserved_slot_strategy_note.tex`
- `docs/reference/simulation_explanation/tex/fcfs/first_two_class_simulation_note_v2.tex`

The TeX notes are the source of truth for notation. The notebooks were updated to follow that notation:

- `S`: total daily capacity
- `H`: rolling booking horizon
- `Q`: protected Class 1 capacity per day
- `lambda_i`: class-specific daily arrival rate
- `lambda`: total daily arrival rate
- `p`: Class 1 demand share
- `tau`: offered delay
- `rho^{booked}`: booked-slot utilization, or booked service-day slots divided by available measured slots
- `rho^{attended}`: attended utilization, or served slots divided by available measured slots
- `served rate_i`: served_i divided by arrivals_i

## Baseline For The New Analysis

The near-term analysis is centered around the realistic demand scale:

- Total demand: `lambda ~= 25` arrivals/day across both classes.
- Capacity: `S = 20` appointment slots/day.
- Class 1 demand share: `p = 0.58`, close to the realistic config with Class 1 demand 14/day and Class 2 demand 10/day.
- Main policy value for examples: `Q = 5` protected Class 1 slots/day.

Notebook 11 uses a wider but still baseline-centered sweep:

- `Q = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12]`
- `lambda = [15, 20, 25, 30, 35, 40, 50]`
- `p = [0.40, 0.50, 0.58, 0.65, 0.75]`
- Seeds: `5101..5130`
- Behavior settings:
  - symmetric baseline
  - Class 1 advantaged
  - Class 1 disadvantaged

## Objective Functions

The objective functions are not new booking policies. They are after-the-fact scoring rules applied to completed FCFS and strict-reservation simulation runs.

### Weighted Booked Slot Use

Class-specific booked-slot utilization contribution:

```text
rho_i^{booked} = (served_i + no_show_i) / (S * measured_days)
```

Weighted booked-slot score:

```text
U_booked(w1, w2) = w1 * rho_1^{booked} + w2 * rho_2^{booked}
```

Class 2 is kept at `w2 = 1`. Larger `w1` means Class 1 booked slots count more. This is now the main capacity score because it directly shows whether protected slots remain empty. The older completed-visit utilization, `rho^{attended}`, is still useful for no-show losses.

### Weighted Served Rate

```text
U_served(w1, w2)
  = (w1 * served_1 + w2 * served_2)
    / (w1 * arrivals_1 + w2 * arrivals_2)
```

This score is between 0 and 1 before any penalty is applied.

### Net Booked-Slot Score With Slot Cost

```text
U_net_booked(w1, w2, c)
  = U_booked(w1, w2) - c * Q / S
```

This treats configured protected capacity as costly, even if the slots are later used. The idea is to penalize the loss of pooling flexibility.
The served-rate version, `U_served(w1, w2) - c * Q / S`, is kept as an access diagnostic.

### Wait-Adjusted Score

Weighted average offered delay:

```text
tau_bar_w
  = (w1 * sum_tau_offered_class_1 + w2 * sum_tau_offered_class_2)
    / (w1 * offered_1 + w2 * offered_2)
```

Wait-adjusted score:

```text
U_wait(w1, w2, c, gamma)
  = U_booked(w1, w2) - c * Q / S - gamma * tau_bar_w / H
```

This score penalizes longer offered waits after normalizing by the booking horizon.

## Current Parameter Guess

The first parameter set is intentionally an educated starting point, not a final claim.

Weights:

```text
w1 = [1.0, 1.25, 1.5, 2.0, 3.0]
w2 = 1.0
```

Slot-cost penalties:

```text
c = [0.0, 0.02, 0.05, 0.10]
```

Wait penalties:

```text
gamma = [0.0, 0.02, 0.05, 0.10]
```

Headline setting:

```text
w1 = 1.5
c = 0.05
gamma = 0.05
```

Interpretation:

- `w1 = 1.5` means Class 1 is prioritized, but not so strongly that Class 2 is ignored.
- `w1 = 2.0` is a stronger priority sensitivity case.
- `w1 = 3.0` is a high-priority stress case.
- `c = 0.05` makes protected capacity matter without dominating the score.
- `c = 0.10` is a conservative case where lost flexibility is expensive.
- `gamma = 0.05` gives offered waiting time a visible but not overwhelming effect.

## Requirement Checks

The notebooks also evaluate whether a run passes basic requirements:

```text
rho^{booked} >= rho_min
min_i served_rate_i >= x
abs(rho_1^{booked} - rho_2^{booked}) <= g
```

Current requirement grids:

```text
rho_min = [0.50, 0.75, 0.85, 0.90]
x       = [0.45, 0.55, 0.65, 0.70]
g       = [0.10, 0.20, 0.30]
```

Headline requirement check:

```text
rho_min = 0.85
x = 0.55
g = 0.20
```

## Paired Comparison To FCFS

The sweep uses paired comparisons. For each scenario and seed:

1. Run pooled FCFS once.
2. Run strict reservation for each `Q`.
3. Compare strict reservation against the matched FCFS run.

For an objective score `U`:

```text
Delta U(Q; scenario, seed)
  = U_strict(Q; scenario, seed) - U_FCFS(scenario, seed)
```

The sweep averages `Delta U` across seeds and classifies each setting:

- **win:** mean delta > 0 and the 95% confidence interval lower bound is above 0.
- **possible win:** mean delta > 0 but the interval crosses 0.
- **loss:** mean delta <= 0.

## Implemented Outputs

Notebook 10:

- Defines the objective functions.
- Uses a small paired FCFS vs strict-reservation example.
- Includes checks for:
  - FCFS has `Q = 0`.
  - Weighted served-rate values are in `[0, 1]`.
  - Larger slot-cost penalty never increases `U_net` for the same run.
  - Arrival accounting partitions outcomes into served/lost components.

Notebook 11:

- Runs the paired FCFS vs strict-reservation sweep.
- Builds:
  - best `Q` per scenario and score
  - win / possible-win / loss summaries
  - paired deltas by seed
  - requirement-check summaries
- Plots:
  - score difference heatmap by `Q` and demand
  - win-region maps by Class 1 share and behavior setting
  - best-`Q` curves by Class 1 weight
  - slot-cost sensitivity curves

## Suggested Plot Attachments

Attach plots from notebook 11 in this order if available.

### 1. Score Difference Heatmap

Suggested caption:

> Mean strict-reservation minus FCFS score difference for the headline net booked-slot score. Positive values indicate that strict reservation outperforms FCFS under the chosen weight and slot-cost penalty.

Recommended filters:

```text
score = net_booked_slot_score
w1 = 1.5
c = 0.05
Class 1 share = 0.58
behavior = symmetric_baseline
```

### 2. Win-Region Map

Suggested caption:

> Best-Q win classification across demand and Class 1 share. Green regions indicate scenarios where at least one strict-reservation `Q` has a positive paired advantage over FCFS.

### 3. Best-Q Curves

Suggested caption:

> Best protected capacity `Q` as total demand changes, shown separately for Class 1 weights. This shows how stronger priority weights move the preferred reservation level.

### 4. Slot-Cost Sensitivity

Suggested caption:

> Sensitivity of strict-reservation advantage to the slot-cost penalty `c`. Higher `c` makes large `Q` harder to justify because protected capacity is treated as more costly.

### 5. Requirement-Check Summary Table

Suggested caption:

> Share of seeds where strict reservation passes minimum booked-slot utilization, minimum class served-rate, and class booked-slot-gap requirements. This is useful for checking whether a high objective score is achieved without unacceptable side effects.

## Questions For Review

Questions that would be useful to ask a supervisor or another LLM:

1. Are the objective functions aligned with the policy question, or should one be dropped?
2. Is the slot-cost penalty `c * Q / S` the right way to represent the cost of protected capacity?
3. Should the cost depend on realized unused protected slots instead of configured `Q`?
4. Are `w1 = 1.5`, `c = 0.05`, and `gamma = 0.05` reasonable headline values?
5. Should the main decision rule be based on `U_net_booked`, `U_wait`, or requirement-constrained booked-slot utilization?
6. Are the requirement floors too permissive or too strict for the clinic context?
7. Should the analysis focus on the realistic behavior setting, or should symmetric / disadvantaged cases be treated equally?

## Current Caveats

- The objective functions are notebook-local. The simulation behavior was not changed; the metrics layer now also exposes `booked_slot_utilization`.
- The full notebook 11 sweep is intentionally slow. It has not been treated as a unit test.
- The first parameter grid is an educated starting point. It should be revised after seeing the first result plots.
- The slot-cost penalty currently uses configured protected capacity `Q`, not realized unused protected slots.
- A strict reservation policy can look good for Class 1 access while still reducing booked-slot utilization, completed-visit utilization, or Class 2 access.

## Suggested Next Step

Run notebook 11, export the suggested plots, and review whether the headline objective `U_net_booked(w1=1.5, c=0.05)` identifies sensible `Q` values near the 25-arrivals/day baseline. If the result is too sensitive to `c` or `w1`, the next step should be narrowing those parameters before broadening the simulation grid.
