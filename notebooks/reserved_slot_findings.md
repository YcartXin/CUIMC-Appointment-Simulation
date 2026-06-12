# Reserved-Slot Strategy Findings

**Refresh before presentation.** These findings now focus on pooled FCFS and strict Class 1 reservation. Regenerate the notebook and result CSVs before presenting the numeric comparisons.

These findings summarize the simulations documented in `notebooks/temp/reserved_slot_simulation_results.md`.

## Main Findings

1. **Pooled FCFS and strict Class 1 reservation are almost identical in aggregate.**

   With the current balanced scenario, both policies fully schedule measured service-day slots and produce the same average served-slot utilization: `0.9076`. Overall served rate is also effectively unchanged at `0.5801`.

2. **Strict reservation only creates a small Class 1 access advantage.**

   Strict reservation improves Class 1 served rate from `0.5812` to `0.5879`, while Class 2 falls from `0.5791` to `0.5724`. The Class 1 minus Class 2 served-rate gap moves from `0.0021` to `0.0155`.

3. **There are no no-offer losses in this scenario.**

   Both policies have a no-offer rate of `0.0000`. The main differences are not caused by horizon exhaustion; they come from how reservation changes offered delays, balking, cancellation, and no-show exposure.

## Interpretation

The strict reservation policy behaves like a mild access-priority rule. It gives Class 1 slightly shorter delays and Class 2 slightly longer delays, but the aggregate result remains nearly unchanged because both classes have identical behavior parameters and the calendar remains fully scheduled.

## Practical Takeaway

Strict reservation is the cleaner policy if the goal is genuine Class 1 protection.

## Suggested Next Checks

- Sweep `reserved_slots_per_day` from small to large values.
- Test asymmetric Class 1/Class 2 behavior parameters instead of identical behavior.

## Arrival-Rate Outcome-Share Sweep

The arrival-rate sweep is documented in `notebooks/temp/reservation_outcome_shares_by_arrival_rate.md`.

The sweep varies total daily arrival rate from `20` to `100` while keeping the class mix symmetric. It plots served/not-lost share and lost-outcome shares for strict reservation:

- `notebooks/temp/figures/reservation_outcome_shares_by_arrival_rate.png`
- `notebooks/temp/figures/reservation_lost_shares_by_arrival_rate.png`

Main pattern: as demand rises, strict reservation eventually converges toward high lost shares. At lower and moderate arrival rates, strict reservation first loses patients mainly through cancellation and no-show before balking becomes material.
