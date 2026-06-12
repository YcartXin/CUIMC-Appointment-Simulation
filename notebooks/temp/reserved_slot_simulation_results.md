# Reserved-Slot Simulation Results

**Stale after reservation-engine correction.** These tables and CSV artifacts were generated before the Class-1-first backfill logic was corrected to process Class 1 with day-by-day reserved-then-general search. Regenerate the notebook and CSVs before using the reported numbers.

Generated from the editable scenario currently defined in `notebooks/05_reserved_slot_strategy.ipynb`.

## Scenario

| Parameter | Value |
|:--|--:|
| slots per day `S` | 32 |
| horizon days `H` | 14 |
| burn-in days | 30 |
| measured days | 365 |
| cooldown days | 10 |
| reserved class | Class 1 |
| reserved slots per day `Q` | 10 |
| seeds | 5101-5130 |
| Class 1 arrival rate | 25/day |
| Class 2 arrival rate | 25/day |
| cancellation probability | 0.10 for both classes |
| balking rule | low 0.00 through delay 9, high 0.50 above delay 9 |
| no-show rule | low 0.00 through delay 6, high 0.30 above delay 6 |

Policies compared:

- `Pooled FCFS`: one shared pool of 32 slots.
- `Strict C1 reservation`: 10 slots are protected for Class 1 only; Class 2 cannot use unused protected slots.
- `Class-1-first backfill reservation`: Class 1 arrivals are handled first with reserved-then-general search on each appointment day; Class 2 can backfill unused protected slots afterward.

## Aggregate Means Across 30 Seeds

| policy | measured booked slot util. | measured served slot util. | served rate | booked arrival rate | balked rate | no-offer rate | mean offered delay | total served |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Pooled FCFS | 1.0000 | 0.9076 | 0.5801 | 1.0000 | 0.0000 | 0.0000 | 5.4615 | 10595.0 |
| Strict C1 reservation | 1.0000 | 0.9076 | 0.5801 | 1.0000 | 0.0000 | 0.0000 | 5.4615 | 10595.1 |
| Class-1-first backfill reservation | 1.0000 | 0.9502 | 0.6080 | 0.8737 | 0.1263 | 0.0000 | 5.1760 | 11101.6 |

## Aggregate Standard Deviations Across 30 Seeds

| policy | measured booked slot util. | measured served slot util. | served rate | booked arrival rate | balked rate | no-offer rate | mean offered delay | total served |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Pooled FCFS | 0.0000 | 0.0122 | 0.0114 | 0.0000 | 0.0000 | 0.0000 | 0.0894 | 147.0450 |
| Strict C1 reservation | 0.0000 | 0.0121 | 0.0113 | 0.0000 | 0.0000 | 0.0000 | 0.0894 | 146.9370 |
| Class-1-first backfill reservation | 0.0000 | 0.0019 | 0.0042 | 0.0049 | 0.0049 | 0.0000 | 0.0524 | 22.4033 |

## Deltas Relative To Pooled FCFS

| policy | measured booked slot util. | measured served slot util. | served rate | booked arrival rate | balked rate | mean offered delay | total served |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Pooled FCFS | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| Strict C1 reservation | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1 |
| Class-1-first backfill reservation | 0.0000 | 0.0426 | 0.0279 | -0.1263 | 0.1263 | -0.2856 | 506.5330 |

## Class-Level Means Across 30 Seeds

| policy, class | percent serviced | booked rate | balking rate | mean offered delay | mean accepted delay | arrivals | booked | balked | served |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Pooled FCFS, Class 1 | 0.5812 | 1.0000 | 0.0000 | 5.4596 | 5.4596 | 9149.07 | 9149.07 | 0.00 | 5316.63 |
| Pooled FCFS, Class 2 | 0.5791 | 1.0000 | 0.0000 | 5.4635 | 5.4635 | 9116.67 | 9116.67 | 0.00 | 5278.40 |
| Strict C1 reservation, Class 1 | 0.5879 | 1.0000 | 0.0000 | 5.3828 | 5.3828 | 9149.07 | 9149.07 | 0.00 | 5377.80 |
| Strict C1 reservation, Class 2 | 0.5724 | 1.0000 | 0.0000 | 5.5405 | 5.5405 | 9116.67 | 9116.67 | 0.00 | 5217.33 |
| Class-1-first backfill reservation, Class 1 | 0.3364 | 0.7471 | 0.2529 | 8.1118 | 7.4381 | 9119.93 | 6813.33 | 2306.60 | 3067.67 |
| Class-1-first backfill reservation, Class 2 | 0.8792 | 1.0000 | 0.0000 | 2.2455 | 2.2455 | 9138.90 | 9138.90 | 0.00 | 8033.90 |

## Class Access Gap

| policy | Class 1 served rate | Class 2 served rate | Class 1 minus Class 2 served rate | Class 1 balking rate | Class 2 balking rate | Class 1 offered delay | Class 2 offered delay |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Pooled FCFS | 0.5812 | 0.5791 | 0.0021 | 0.0000 | 0.0000 | 5.4596 | 5.4635 |
| Strict C1 reservation | 0.5879 | 0.5724 | 0.0155 | 0.0000 | 0.0000 | 5.3828 | 5.5405 |
| Class-1-first backfill reservation | 0.3364 | 0.8792 | -0.5428 | 0.2529 | 0.0000 | 8.1118 | 2.2455 |

## Generated CSV Artifacts

The same run data are saved in this directory:

- `reserved_slot_aggregate_runs.csv`: one row per policy/seed aggregate result.
- `reserved_slot_class_runs.csv`: one row per policy/seed/class result.
- `reserved_slot_aggregate_mean.csv`: aggregate means by policy.
- `reserved_slot_aggregate_std.csv`: aggregate standard deviations by policy.
- `reserved_slot_class_mean.csv`: class-level means by policy/class.
- `reserved_slot_class_gap.csv`: class access gap summary by policy.
