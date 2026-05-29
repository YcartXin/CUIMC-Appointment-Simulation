# Audit: fcfs_conclusions_deck.tex

Date: May 2026

---

## Files Inspected

| File | Role |
|------|------|
| `configs/baseline.yaml` | Source for λ=100, S=32 stress-test parameters |
| `configs/realistic.yaml` | Source for realistic scenario parameters (λ₁=14, λ₂=10, S=20, H=28) |
| `docs/reports/metric_analysis/data/baseline_summary.csv` | Numerical source for baseline utilization and served rate (30 seeds) |
| `results/balking_effect_verification.csv` | Per-seed data for A_baseline, B_no_balking, C_half_balking, D_full_balking scenarios |
| `docs/reports/metric_analysis/research_style/final/simulation_realistic/fcfs_realistic_comparison.tex` | Source for realistic-scenario utilization, served rate, and class-level served rates |
| `docs/reports/conclusions/paper/fcfs_conclusion_sheet.tex` | Existing meeting-ready conclusion sheet (reference for conclusions) |
| `docs/reports/conclusions/paper/fcfs_conclusion_sheet_audit.md` | Prior audit documenting number sources and editorial decisions |
| `docs/reports/metric_analysis/research_style/final/note/fcfs_research_note_final.tex` | Full technical note with regression and sweep findings |
| `outputs/class1_balking/summary/aggregate_summary.csv` | Balking sweep summary (utilization flat, served rate flat, accepted wait drops from selection) |
| `outputs/class1_no_show/summary/aggregate_summary.csv` | No-show sweep summary (utilization drops monotonically with no-show rate) |
| `docs/reports/metric_analysis/data/regression_standardized_coefficients.csv` | Standardized OLS regression screen for behavioral drivers |

---

## Numbers Used and Sources

| Number on slide | Slide | Raw value | Source |
|-----------------|-------|-----------|--------|
| λ = 100 | 2 | — | `configs/baseline.yaml`, `lambda_per_day: 50` × 2 classes |
| S = 32 | 2 | — | `configs/baseline.yaml`, `slots_per_day: 32` |
| Utilization ≈ 0.841 | 2 | 0.840585... | `baseline_summary.csv`, column `average_utilization` |
| Served rate ≈ 0.269 | 2 | 0.268907... | `baseline_summary.csv`, column `overall_percent_serviced` |

All other slide content is qualitative and does not cite specific numbers. The qualitative claims are corroborated as follows:

| Qualitative claim | Slide | Evidence |
|-------------------|-------|----------|
| Balking shifts losses to no-offer and cancellation | 3 | `balking_effect_verification.csv`: B_no_balking mean no-offer ≈ 0.297 (vs 0 in baseline), mean cancellation ≈ 0.384 (vs 0.325 in baseline) |
| Cancellation releases slots; no-show does not | 4 | Simulation engine (`simulation/engine.py`): cancellations return slots to horizon; no-shows do not trigger rebooking |
| No-show is the primary utilization driver | 4 | Regression screen: no-show threshold β = +0.512, no-show step β = −0.423 (top-2 utilization drivers); `outputs/class1_no_show/summary/aggregate_summary.csv`: utilization monotonically declines as no-show probability increases |
| Symmetric parameters → near-zero class gap | 5 | `baseline_summary.csv`, `access_advantage_class_1 = −0.000354` |
| Asymmetric behavior → class gap under pooled FCFS | 5 | `fcfs_realistic_comparison.tex` abstract: Class 1 served rate 0.881, Class 2 0.661, gap 0.220 (22 percentage points) |

---

## Conclusions Included

1. **Slide 1 — Main message:** Utilization ≠ access; minimum reporting set of six metrics.
2. **Slide 2 — Capacity pressure:** Stress-test baseline decouples utilization from access; baseline caveat.
3. **Slide 3 — Balking:** Balking reallocates capacity under pooled FCFS; no-balking diagnostic shifts, not recovers, losses.
4. **Slide 4 — Cancellation vs. no-show:** Distinct behavioral channels; no-show is the direct slot-loss channel.
5. **Slide 5 — Class-level gaps and meeting decision:** Symmetric baseline hides asymmetric outcomes; framing choice for the meeting.

---

## Conclusions Omitted

| Omitted detail | Reason |
|----------------|--------|
| Specific regression coefficients (β = −0.774 for arrival rate, β = +0.429 for balking threshold gap, etc.) | Too granular for a slide; coefficients require model context to interpret correctly |
| Accepted wait values (8.34 d baseline, 9.45 d no-balking) | Selection-bias explanation requires more words than a bullet allows |
| Realistic scenario aggregate values (utilization 0.934, served rate 0.789) | Class gap (0.220) is the conclusion that matters; aggregates would dilute it |
| Full balking sweep range (served rate stable ~0.269 across all balking levels) | Captured qualitatively in Slide 3 bullet 1 |
| Full no-show sweep range (utilization 0.920 → 0.681) | Captured qualitatively in Slide 4 bullet 3 |
| Cancellation survival calculation (0.9^8.3 ≈ 0.42) | Proof-style; not a conclusion |
| No-offer share diagnostic values (0.297 in B_no_balking) | Captured qualitatively in Slide 3 bullet 3 |
| Demand-to-capacity ratio framing (3.1:1 vs 1.2:1) | Captured via λ and S values on Slide 2 |
| Class 1 vs Class 2 specific served rates in realistic scenario | Gap (0.220) cited qualitatively; exact rates omitted to stay ≤ 60 words/slide |

---

## Compile Command

```bash
cd docs/reports/conclusions/deck
pdflatex -interaction=nonstopmode fcfs_conclusions_deck.tex
pdflatex -interaction=nonstopmode fcfs_conclusions_deck.tex
```

---

## Verification

- LaTeX output (both passes): `Output written on fcfs_conclusions_deck.pdf (5 pages, 104987 bytes).`
- macOS metadata check: `kMDItemNumberOfPages = 5`
- Log check: no errors; `rerunfilecheck` warning resolved by second pass.
- Visual: each slide has one headline and 2–3 bullets; no tables; no paragraphs.

## Final Page Count

**5 pages.**
