# Strict Class 1 Reservation: Policy-Selection Results

> **Conclusion.** Strict reservation raises both normalized weighted objectives in most tested cells, but the gains are produced by moving service toward Class 1. At 50 arrivals per class, the mathematical objective choice is always `Q=32`, which excludes Class 2 entirely. The lowest offered waiting time is therefore not an overall access improvement.

## 1. Purpose

This report identifies reservation quantities that perform similarly under two primary normalized objectives. It compares every Q with pooled FCFS (`Q=0`) and does not treat offered waiting time as an access objective by itself.

## 2. Experiment Grid

- 5 balking-threshold pairs and 3 common post-threshold balking rates.
- Equal Class 1 and Class 2 arrival rates of 25 or 50 per day.
- `Q = [0,2,4,6,8,10,12,16,20,24,28,32]`.
- Weights `(1,1)` and `(2,1)`; 20 seeds per simulation cell.
- Capacity 32/day, horizon 14, burn-in 30, measurement 365, cooldown 14.

## 3. Objective Definitions

With `S = 32 × 365` measured slots:

$$Obj_{util,raw}=w_1Y_1/S+w_2Y_2/S,\qquad Obj_{util,norm}=Obj_{util,raw}/(w_1+w_2).$$

$$Obj_{service,raw}=w_1Y_1/A_1+w_2Y_2/A_2,\qquad Obj_{service,norm}=Obj_{service,raw}/(w_1+w_2).$$

$$T_{wait,offered}=\frac{w_1\sum\tau_{offered,1}+w_2\sum\tau_{offered,2}}{w_1\,\mathrm{offered}_1+w_2\,\mathrm{offered}_2}.$$

The normalized utilization and service values are the primary comparison objectives. Raw values are retained. Offered waiting time is secondary and conditional on receiving an offer.

## 4. Main Findings

- A positive-Q policy exceeds FCFS in mean normalized utilization in 58 of 60 scenario-weight cells; a positive Q appears in the 1% near-tie set in 60 cells. The median best-strict delta is +0.080.
- A positive-Q policy exceeds FCFS in mean normalized service rate in 60 of 60 cells; a positive Q appears in the 1% near-tie set in 60 cells. The median best-strict delta is +0.062.
- Several Q values are effectively equivalent in 9 utilization cells and 9 service-rate cells.
- At 25 arrivals per class, increasing `w1` from 1 to 2 moves both primary objectives to `Q=24` in every behavior regime. With `w1=1`, the selected range is more behavior-dependent and sometimes includes FCFS.
- At 50 arrivals per class, both primary objectives select `Q=32` in every behavior and weight regime. This is full Class 1 protection and complete Class 2 exclusion.

## 5. Best-Q Recommendations By Regime

The cells below are 1% near-tie ranges, not forced unique optima. Bracketed lists show the tested Q values represented by each range.
These are objective-specific mathematical candidates, not access-constrained policy recommendations.

| behavior | util: 25/class, w1=1 | util: 25/class, w1=2 | util: 50/class, w1=1 | util: 50/class, w1=2 | service: 25/class, w1=1 | service: 25/class, w1=2 | service: 50/class, w1=1 | service: 50/class, w1=2 |
|---|---|---|---|---|---|---|---|---|
| (12,12), b=0.3 | 24 | 24 | 32 | 32 | 24 | 24 | 32 | 32 |
| (12,12), b=0.5 | 24 | 24 | 32 | 32 | 24 | 24 | 32 | 32 |
| (12,12), b=0.7 | 24 | 24 | 32 | 32 | 24 | 24 | 32 | 32 |
| (5,12), b=0.3 | 0-12 [0,2,4,6,8,10,12] | 24 | 32 | 32 | 0-12 [0,2,4,6,8,10,12] | 24 | 32 | 32 |
| (5,12), b=0.5 | 0-12 [0,2,4,6,8,10,12] | 24 | 32 | 32 | 0-12 [0,2,4,6,8,10,12] | 24 | 32 | 32 |
| (5,12), b=0.7 | 0-12 [0,2,4,6,8,10,12] | 24 | 32 | 32 | 0-12 [0,2,4,6,8,10,12] | 24 | 32 | 32 |
| (5,9), b=0.3 | 0-12 [0,2,4,6,8,10,12] | 24 | 32 | 32 | 0-12 [0,2,4,6,8,10,12] | 24 | 32 | 32 |
| (5,9), b=0.5 | 0-12 [0,2,4,6,8,10,12] | 24 | 32 | 32 | 0-12 [0,2,4,6,8,10,12] | 24 | 32 | 32 |
| (5,9), b=0.7 | 0-12 [0,2,4,6,8,10,12] | 24 | 32 | 32 | 0-12 [0,2,4,6,8,10,12] | 24 | 32 | 32 |
| (9,5), b=0.3 | 0-16 [0,2,4,6,8,10,12,16] | 24 | 32 | 32 | 0-16 [0,2,4,6,8,10,12,16] | 24 | 32 | 32 |
| (9,5), b=0.5 | 0-20 [0,2,4,6,8,10,12,16,20] | 24 | 32 | 32 | 0-20 [0,2,4,6,8,10,12,16,20] | 24 | 32 | 32 |
| (9,5), b=0.7 | 0-24 [0,2,4,6,8,10,12,16,20,24] | 24 | 32 | 32 | 0-24 [0,2,4,6,8,10,12,16,20,24] | 24 | 32 | 32 |
| (9,9), b=0.3 | 24 | 24 | 32 | 32 | 24 | 24 | 32 | 32 |
| (9,9), b=0.5 | 24 | 24 | 32 | 32 | 24 | 24 | 32 | 32 |
| (9,9), b=0.7 | 24 | 24 | 32 | 32 | 24 | 24 | 32 | 32 |

![Utilization best-Q ranges](figures/best_q_obj_util_norm.png)

The utilization map shows how weighting Class 1 and increasing demand move the practically equivalent reservation region.

![Service-rate best-Q ranges](figures/best_q_obj_service_norm.png)

The service-rate map separates access performance from capacity-based performance; the two objectives need not choose the same range.

## 6. FCFS Comparison

![Representative deltas](figures/representative_delta_vs_fcfs.png)

Positive values indicate improvement over matched-seed FCFS. The representative symmetric regime shows whether gains persist across arrival and weight settings rather than relying on a single curve. The improvement is weighted: it does not mean both classes improve.

## 7. Class Tradeoffs

![Class service tradeoff](figures/class_service_tradeoff.png)

Increasing Q generally moves service toward Class 1 and away from Class 2. An objective improvement should therefore be read as a weighted tradeoff, not as a simultaneous improvement for both classes.

## 8. Offered Waiting Time And No-Offer Composition Effects

![Wait and no-offer diagnostic](figures/offered_wait_no_offer_diagnostic.png)

Under the pre-specified strict flag, both class no-offer rates rise in 0 of 60 minimum-wait cells. However, all 60 minimum-wait cells select `Q=32`, where Class 2 receives no offers and its waiting-time line is undefined. The lower weighted offered wait is therefore an access-composition effect, not a true overall waiting-time improvement.

## 9. Limitations

- Same seeds provide matched labels, but policy-dependent RNG use means they are not exact common-random-number experiments.
- The utilization objective follows measured-arrival cohorts through cooldown; it is not identical to measured-service-day utilization.
- Results use a 1% practical-equivalence rule and 20 seeds.
- No reservation cost or external access constraint is imposed.
- The reported best Q values maximize the stated weighted objectives; they should not be adopted without a Class 2 access requirement.

## 10. Next Step: Comparison With Booking-Window Policy

Apply the same normalized objectives and access diagnostics to the historical booking-window policy, then compare each policy's best near-tie region under identical behavior, demand, weight, and seed settings. A combined policy should be studied only after this direct comparison.

Data tables: `tables/scenario_level_summary.csv` and `tables/best_q_summary.csv`; explicit near-tie members are in `tables/near_tie_q_ranges.csv`. Full run-level outputs remain under `outputs/strict_reservation_policy_selection/standard/`.
