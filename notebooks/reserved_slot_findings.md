# Reserved-Slot Strategy Findings

**Stale after reservation-engine correction.** These findings were written from simulations generated before the Class-1-first backfill logic was corrected. Regenerate the notebook and result CSVs before presenting the numeric comparisons.

These findings summarize the simulations documented in `notebooks/temp/reserved_slot_simulation_results.md`.

## Main Findings

1. **Pooled FCFS and strict Class 1 reservation are almost identical in aggregate.**

   With the current balanced scenario, both policies fully schedule measured service-day slots and produce the same average served-slot utilization: `0.9076`. Overall served rate is also effectively unchanged at `0.5801`.

2. **Strict reservation only creates a small Class 1 access advantage.**

   Strict reservation improves Class 1 served rate from `0.5812` to `0.5879`, while Class 2 falls from `0.5791` to `0.5724`. The Class 1 minus Class 2 served-rate gap moves from `0.0021` to `0.0155`.

3. **Class-1-first backfill reservation improves aggregate utilization but reverses the class advantage.**

   Class-1-first backfill reservation increases measured served-slot utilization from `0.9076` to `0.9502` and raises total served patients by about `506.5` per run on average. However, it does this by shifting access strongly toward Class 2: Class 1 served rate falls to `0.3364`, while Class 2 served rate rises to `0.8792`.

4. **The Class-1-first backfill policy creates Class 1 balking in this scenario.**

   Under pooled and strict policies, no arrivals balk because offered delays stay below the balking threshold. Under Class-1-first backfill reservation, Class 1 mean offered delay rises to `8.1118`, and about `25.29%` of Class 1 offered patients balk. Class 2 mean offered delay falls to `2.2455`, and Class 2 has no balking.

5. **There are no no-offer losses in this scenario.**

   All three policies have a no-offer rate of `0.0000`. The main differences are not caused by horizon exhaustion; they come from how reservation changes offered delays, balking, cancellation, and no-show exposure.

## Interpretation

The strict reservation policy behaves like a mild access-priority rule. It gives Class 1 slightly shorter delays and Class 2 slightly longer delays, but the aggregate result remains nearly unchanged because both classes have identical behavior parameters and the calendar remains fully scheduled.

The Class-1-first backfill reservation policy now uses day-level Class 1 priority: Class 1 searches reserved then general capacity on each appointment day, and Class 2 backfills remaining protected capacity only after the Class 1 batch. The stale numbers below were generated before that correction and should not be used to characterize the corrected policy.

## Practical Takeaway

Strict reservation is the cleaner policy if the goal is genuine Class 1 protection.

Class-1-first backfill reservation should be re-evaluated after regenerating results with the corrected day-level priority rule.

## Suggested Next Checks

- Sweep `reserved_slots_per_day` from small to large values.
- Test asymmetric Class 1/Class 2 behavior parameters instead of identical behavior.
- Compare Class-1-first backfill reservation against a variant where unused protected slots are backfilled only after the full day, not within the same arrival batch.
- Add a visual day walkthrough for a seed where Class-1-first backfill reservation creates Class 1 balking.

## Arrival-Rate Outcome-Share Sweep

The arrival-rate sweep is documented in `notebooks/temp/reservation_outcome_shares_by_arrival_rate.md`.

The sweep varies total daily arrival rate from `20` to `100` while keeping the class mix symmetric. It plots served/not-lost share and lost-outcome shares for the two reservation policies:

- `notebooks/temp/figures/reservation_outcome_shares_by_arrival_rate.png`
- `notebooks/temp/figures/reservation_lost_shares_by_arrival_rate.png`

Main pattern: as demand rises, both reservation policies eventually converge toward high lost shares. At lower and moderate arrival rates, Class-1-first backfill reservation tends to introduce balking earlier, while strict reservation first loses patients mainly through cancellation and no-show before balking becomes material.
