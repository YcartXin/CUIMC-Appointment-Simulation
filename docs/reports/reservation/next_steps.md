# Reservation Analysis Next Steps

## Summary

The two current strict-reservation reports answer different questions:

- `policy_selection/strict_reservation_policy_selection.md` shows how strict Class 1 reservation behaves under the normalized utilization and service-rate objectives.
- `assumption_diagnostics/assumption_report.md` checks whether the strict-reservation simulation and the main business hypotheses behave as expected under the broader historical behavior grid.

Together, they support using strict reservation as a candidate policy, but not as a final recommendation. The policy-selection report shows weighted objective gains, while the diagnostics report shows that those gains need to be read with Class 2 access, no-offer rates, and objective-definition checks.

## What We Know

- Strict reservation can improve the weighted objectives, especially when Class 1 has higher weight or demand is high.
- These improvements mainly come from moving service toward Class 1, not from improving both classes at the same time.
- `T_wait_offered` is not an access objective by itself. It can look better when some patients stop receiving offers.
- The `Q=0` sanity check passed, so the reservation implementation behaves like pooled FCFS when there are no reserved slots.
- Hard simulation checks passed across the standard diagnostic sweep.
- The utilization and service-rate objectives are not interchangeable in exact pairwise ordering, but clear objective conflicts are rare.

## Immediate Next Step

Run the same objective and diagnostic workflow for the historical booking-window policy.

Use the same:

- behavior regimes;
- arrival rates;
- weights;
- seeds;
- normalized utilization objective;
- normalized service-rate objective;
- offered-wait diagnostic with no-offer rates;
- hard checks and business-hypothesis diagnostics.

The comparison should answer whether booking windows, strict reservation, or neither performs better under the same conditions. A combined booking-window plus reservation policy should wait until this direct comparison is done.

## Comparison Outputs To Build

- One table comparing strict reservation and booking windows by scenario and objective.
- One compact best-range table for each policy under `Obj_util_norm` and `Obj_service_norm`.
- One class-tradeoff plot comparing how each policy moves Class 1 and Class 2 served rates.
- One offered-wait/no-offer diagnostic plot for both policies.
- One diagnostics summary showing hard-check status and business-hypothesis status for both policies.

## Interpretation Rules

- Do not treat a high weighted objective as a policy recommendation without an access requirement for Class 2.
- Do not treat lower offered wait as an improvement unless no-offer rates and service rates remain acceptable.
- Report near-tie ranges rather than forcing a unique best policy or unique best parameter.
- Keep strict reservation and booking windows separate first. Study a combined policy only after the direct comparison is clear.
