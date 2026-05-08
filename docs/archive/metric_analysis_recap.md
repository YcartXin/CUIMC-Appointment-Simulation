# Recap: Metric Sensitivity Findings

This is a short recap of the most recent metric and sensitivity analysis. The detailed version is in:

- `outputs/reports/metric_analysis/rendered/metric_analysis.tex`
- `outputs/reports/metric_analysis/rendered/metric_analysis.pdf`

The goal of the recent work was to understand:

1. Which parameters matter most for clinic performance.
2. Whether one patient class is benefited over the other.
3. Whether effects come from absolute parameter levels or from differences between the two classes.

## Metrics Used

The final analysis focuses on relative or average metrics, not total counts.

| Metric | Meaning | Why it matters |
|---|---|---|
| `average_utilization` | Average daily share of slots that became completed visits | Measures capacity use |
| `overall_percent_serviced` | Served patients / arrivals | Measures access |
| `mean_accepted_booking_delay` | Delay among patients who accepted/booked | Measures wait among booked patients |
| `mean_offered_booking_delay` | Delay offered to all patients who got an offer, including those who balked | Better patient-facing delay metric |
| `percent_serviced_i` | Served / arrivals for class `i` | Measures access by class |

We also added two class-benefit metrics:

| Metric | Interpretation |
|---|---|
| `Delta_access = percent_serviced_1 - percent_serviced_2` | Positive means class 1 is served more often; negative means class 2 is served more often |
| `Delta_delay = mean_offered_delay_2 - mean_offered_delay_1` | Positive means class 1 gets shorter offered delays; negative means class 2 gets shorter offered delays |

## Main Findings

### 1. No-shows matter most for utilization

No-shows directly waste booked appointment slots. When no-show probability increases, `average_utilization` drops clearly.

This is the strongest lever if the question is:

> Are clinic slots being converted into completed visits?

### 2. Demand pressure matters most for access and delay

When total arrival rate increases, the system becomes more congested.

That causes:

- lower `overall_percent_serviced`
- higher offered booking delays
- high utilization even when many patients are not served

So utilization alone is not enough. A clinic can look busy while still providing poor access.

The new FCFS stress test makes this very clear. At the baseline arrival rate, utilization is about `0.839`, but only about `0.269` of arrivals are served. At `1.7x` the baseline arrival rate, utilization is still about `0.841`, but the served share falls to about `0.157`.

The main failure mode also changes with load:

- at baseline load, patients mostly leave through balking or after-booking losses
- at high overload, `no_offer` becomes a major failure mode because the FCFS horizon fills

### 3. Cancellation asymmetry creates the largest class advantage

The biggest class-benefit effect came from cancellation probability.

If class 1 cancels much less than class 2, class 1 is much more likely to be served. If class 2 cancels less, class 2 benefits.

This was the largest observed class-access difference in the fine-grid analysis.

### 4. Balking determines which class stays in the system

Balking affects whether patients accept long-delay offers.

The class that:

- has a lower balking step, or
- has a higher balking threshold

is more likely to remain in the system and be served.

So balking differences can create a real class advantage.

When both classes share the same balking threshold and the same jump level, the class advantage almost disappears. That sweep is useful for understanding the absolute access/wait tradeoff, not which class benefits.

### 5. No-show differences benefit one class in completed visits, not delay

No-show behavior happens after booking.

That means no-show parameters strongly affect:

- completed visits
- utilization
- percent serviced

But they barely affect offered booking delay, because the delay was already assigned before the no-show decision.

When both classes share the same no-show threshold and jump level, the main effect is aggregate utilization. Class advantage stays close to zero because the no-show rule is symmetric.

### 6. Arrival mix matters less when classes are otherwise identical

Changing the class mix with:

```text
lambda_1 = p * lambda_total
lambda_2 = (1 - p) * lambda_total
```

does not create a large class advantage in the current baseline.

Reason: the two classes have the same behavior rules and the same value. Since the booking policy is FCFS, class labels do not matter much unless the class parameters differ.

### 7. Own-class arrival rate mostly changes congestion

Increasing class 1 arrivals or class 2 arrivals increases congestion. But because classes are behaviorally symmetric, it does not strongly privilege either class.

Class differences would matter more if classes had different:

- cancellation probabilities
- no-show probabilities
- balking thresholds
- balking steps
- values or priorities

### 8. Regression screen confirms the main drivers

I added a randomized regression screen:

- 240 random FCFS parameter settings
- 2 simulation seeds per setting
- standardized linear regressions on the simulated metrics

Main regression findings:

| Target | Biggest drivers |
|---|---|
| `average_utilization` | no-show threshold, no-show step, cancellation probability |
| `overall_percent_serviced` | total arrival rate, no-show threshold, no-show step |
| `mean_offered_booking_delay` | total arrival rate, cancellation probability, balking threshold |
| `Delta_access` | cancellation gap, balking threshold gap, balking step gap |

This reinforces the heatmap conclusion: absolute demand and no-show behavior drive aggregate performance, while class-specific gaps drive who benefits.

## Impact Ranking

From strongest to weakest class-benefit effect in the recent fine-grid analysis:

| Rank | Parameter | Main effect |
|---:|---|---|
| 1 | Cancellation probability | Biggest class advantage |
| 2 | Balking threshold | Higher tolerance class benefits |
| 3 | Balking step | Lower balking class benefits |
| 4 | No-show step | Lower no-show class completes more visits |
| 5 | No-show threshold | Higher threshold class benefits |
| 6 | Arrival mix | Weak class effect under symmetric behavior |
| 7 | Own-class arrival rate | Mostly congestion, weak class advantage |

The additional common threshold/jump sweeps for balking and no-show are not ranked as class-benefit levers because both classes receive the same parameters. They show absolute system effects: balking changes the access/wait tradeoff, while no-show behavior changes utilization.

## Practical Interpretation

If the goal is better clinic performance:

- Reduce no-shows to improve utilization.
- Manage total demand pressure to improve access and wait times.
- Track cancellation behavior carefully because it can create major class differences.
- Use `overall_percent_serviced` with `average_utilization`; do not rely on utilization alone.
- Use `mean_offered_booking_delay` when thinking about patient experience, because it includes patients who balked after receiving long-delay offers.

## Bottom Line

The current baseline treats classes symmetrically, so class labels alone do not matter much.

Class advantage appears when class-specific behavior differs. The strongest observed class-benefit driver is cancellation probability, followed by balking behavior and no-show behavior. Arrival mix and class-specific arrival volume matter mainly through congestion, not through direct preferential treatment.
