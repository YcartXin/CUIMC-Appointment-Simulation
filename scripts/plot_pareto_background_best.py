"""Recreate slides 25-30 figures with one best Pareto case per background/policy.

Run: python scripts/plot_pareto_background_best.py
The original deck/images and all input CSVs remain unchanged.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ['horizon_only', 'reservation_only', 'both_flexible']
LABELS = {'horizon_only': 'Horizon only', 'reservation_only': 'Reservation only', 'both_flexible': 'Both flexible'}
LEVELS = ['low', 'medium', 'high']
COLORS = {'horizon_only': '#2385b3', 'reservation_only': '#ff7f0e', 'both_flexible': '#7843ad'}
KEYS = ['background_id', 'policy']


def flag(series):
    result = series.astype(str).str.lower().map({'true': True, 'false': False})
    if result.isna().any():
        raise ValueError(f'Invalid boolean values in {series.name}')
    return result


def select_best(cells, bank):
    if bank.background_id.duplicated().any():
        raise ValueError('Duplicate background IDs in bank')
    if cells.duplicated(KEYS + ['horizon_days', 'Q', 'window']).any():
        raise ValueError('Duplicate candidate settings')
    candidates = cells.loc[flag(cells.validated_meaningful_pareto)].copy()
    # Descriptive retrospective selection using evaluated mean Class 1 gain.
    # Secondary keys break exact ties deterministically and never mix cases.
    order = KEYS + ['eval_delta_sr1', 'eval_delta_sr2', 'eval_delta_utilization', 'horizon_days', 'Q', 'window']
    if not np.isfinite(candidates[order[2:]].to_numpy(dtype=float)).all():
        raise ValueError('Missing or nonfinite candidate metrics/settings')
    best = candidates.sort_values(order, ascending=[True, True, False, False, False, True, True, True], kind='stable').drop_duplicates(KEYS)
    design = bank[['background_id', 'noshow_level', 'balk_level', 'rho', 'class1_share', 'slots_per_day']]
    best = best.merge(design, on='background_id', how='left', validate='many_to_one', indicator=True)
    if not best._merge.eq('both').all():
        raise ValueError('Selected cases missing background metadata')
    best = best.drop(columns='_merge')
    maxima = candidates.groupby(KEYS).eval_delta_sr1.max().sort_index()
    pd.testing.assert_series_equal(best.set_index(KEYS).eval_delta_sr1.sort_index(), maxima)
    assert not best.duplicated(KEYS).any()
    assert len(best) == len(candidates[KEYS].drop_duplicates())
    best['reservation_pct'] = best.Q / best.slots_per_day * 100
    if not best.reservation_pct.between(0, 100).all():
        raise ValueError('Reservation count exceeds daily capacity')
    return best


def build(input_dir, combined_dir, output_dir, summary_dir):
    cells = pd.read_csv(input_dir / 'validated_meaningful_pareto_cells.csv')
    bank = pd.read_csv(input_dir / 'patient_behavior_factorial_bank.csv')
    universe = pd.read_csv(input_dir / 'validated_background_policy_universe.csv')
    combined = pd.read_csv(combined_dir / 'combined_background_policy_universe.csv')
    for table in [universe, combined]:
        if table.duplicated(KEYS).any():
            raise ValueError('Duplicate background/policy universe rows')
        table['validated_pareto_exists'] = flag(table.validated_pareto_exists)
    best = select_best(cells, bank)
    selected_keys = pd.MultiIndex.from_frame(best[KEYS])
    universe['selected_case_exists'] = pd.MultiIndex.from_frame(universe[KEYS]).isin(selected_keys)
    if not universe.selected_case_exists.equals(universe.validated_pareto_exists):
        raise ValueError('Candidate qualification disagrees with background universe')
    original = combined.loc[combined.data_source == 'original_experiment2']
    check = original.merge(universe, on=KEYS, validate='one_to_one', suffixes=('_combined', '_original'), indicator=True)
    if len(check) != len(universe) or not check._merge.eq('both').all() or not check.validated_pareto_exists_combined.equals(check.validated_pareto_exists_original):
        raise ValueError('Original and combined universe disagree')
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    best.to_csv(summary_dir / 'selected_best_pareto_cases.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14, 'axes.labelsize': 17,
                         'axes.titlesize': 23, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
                         'legend.fontsize': 14, 'svg.hashsalt': 'pareto-background-best'})

    def canvas(title, subtitle):
        fig, ax = plt.subplots(figsize=(16, 9))
        fig.subplots_adjust(left=.115, right=.97, top=.76, bottom=.22)
        fig.text(.115, .94, title, fontsize=24, weight='bold', color='#1c293b')
        fig.text(.115, .88, subtitle, fontsize=16, color='#596779')
        ax.grid(axis='y', color='#d9e0e8', alpha=.8)
        ax.set_axisbelow(True)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        return fig, ax

    def save(fig, stem, notes):
        for i, note in enumerate(notes):
            fig.text(.115, .11 - i * .035, note, fontsize=11, color='#596779')
        for ext in ['png', 'svg']:
            fig.savefig(output_dir / f'{stem}.{ext}', dpi=200, facecolor='white')
        plt.close(fig)
        print('Generated', stem)

    headline = universe.loc[universe.rho <= 2.5]
    rates = headline.groupby(['policy', 'noshow_level']).selected_case_exists.agg(['size', 'sum', 'mean']).reset_index()
    rates.to_csv(summary_dir / 'slide25_noshow_background_rates.csv', index=False)
    fig, ax = canvas('Pareto opportunities by no-show severity', 'Each background counts once for each policy, whether it has one qualifying case or many')
    x = np.arange(3)
    for i, policy in enumerate(POLICIES):
        g = rates[rates.policy == policy].set_index('noshow_level').reindex(LEVELS)
        bars = ax.bar(x + (i - 1)*.24, g['mean']*100, .23, color=COLORS[policy], label=LABELS[policy])
        for bar, (_, row) in zip(bars, g.iterrows()):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.45, f"{int(row['sum'])}/{int(row['size'])}", ha='center', fontsize=12)
    ax.set(xticks=x, xticklabels=['Low','Medium','High'], ylabel='Backgrounds with a qualifying Pareto case (%)', xlabel='Class 1 no-show severity', ylim=(0, 22))
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.legend(loc='upper left', frameon=False)
    save(fig, 'slide25_noshow_background_best', ['Original Experiment 2, demand / capacity <=2.5. Numerator / denominator shown above bars.', 'Qualification uses independently evaluated means. It does not require confidence-interval-supported nonharm.'])

    scale = combined.loc[(combined.policy == 'both_flexible') & combined.rho.isin([1.7, 2.0])]
    scale = scale.groupby(['class1_share', 'slots_per_day']).validated_pareto_exists.agg(['size','sum','mean']).reset_index()
    scale.to_csv(summary_dir / 'slide26_share_scale_background_rates.csv', index=False)
    fig, ax = canvas('Favorable system scale shifts as Class 1 share rises', 'Both flexible, demand / capacity = 1.7 or 2.0. Each point counts backgrounds once')
    for slots, color, marker in [(30,'#6545a8','o'),(50,'#687c88','s'),(40,'#d27c25','D')]:
        g = scale[scale.slots_per_day == slots].sort_values('class1_share')
        ax.plot(g.class1_share, g['mean']*100, color=color, marker=marker, lw=3, markersize=9, linestyle='-' if len(g)>1 else 'None', label=f'{slots} slots/day' + (' (80% follow-up)' if slots==40 else ''))
    ax.set(xlabel='Class 1 share of arrivals', ylabel='Backgrounds with a qualifying Pareto case (%)', xticks=sorted(scale.class1_share.unique()), ylim=(-3, 65))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.legend(loc='upper left', frameon=False)
    save(fig, 'slide26_scale_background_best', ['Each share and scale point uses 18 backgrounds. The 80% follow-up has success status only.', 'Case results for the follow-up are unavailable. These rates were already based on backgrounds and remain unchanged.'])

    h = best.loc[best.rho <= 2.5]
    util = h.groupby('policy').eval_delta_utilization.agg(['size','median','min','max']).reset_index()
    util['median_pp'] = util['median']*100
    util.to_csv(summary_dir / 'slide27_utilization_best_case.csv', index=False)
    fig, ax = canvas('Utilization change at each background\'s best Pareto case', 'Choose the largest Class 1 served-rate gain within each background and policy')
    shown = ['horizon_only','both_flexible']
    for i, policy in enumerate(shown):
        g = util.set_index('policy').loc[policy]
        value = g.median_pp
        ax.bar(i, value, width=.5, color=COLORS[policy])
        ax.text(i, value+.045, f"{value:.2f}pp\n(n={int(g['size'])} backgrounds)", ha='center', fontsize=15)
    ax.set(xticks=[0,1], xticklabels=[LABELS[p] for p in shown], ylabel='Median utilization change vs baseline (pp)', ylim=(0, 1.9))
    save(fig, 'slide27_utilization_background_best', ['Original Experiment 2, demand / capacity <=2.5. One selected case per qualifying background and policy.', 'Reservation only has no qualifying backgrounds. The median includes every selected case, including negative utilization changes.'])

    horizon = best.groupby(['policy','noshow_level']).agg(backgrounds=('background_id','size'), median_horizon_days=('horizon_days','median'), median_reservation_pct=('reservation_pct','median')).reset_index()
    horizon.to_csv(summary_dir / 'slides29_30_best_case_settings.csv', index=False)
    fig, ax = canvas('Horizon at each background\'s best Pareto case', 'Median of selected horizons, with one best case per background and policy')
    for i, policy in enumerate(shown):
        g = horizon[horizon.policy == policy].set_index('noshow_level').reindex(LEVELS)
        offsets = x + (i-.5)*.22
        ax.scatter(offsets, g.median_horizon_days, color=COLORS[policy], marker='o' if i==0 else 'D', s=115, label=LABELS[policy])
        for j, (_, row) in enumerate(g.iterrows()):
            if pd.isna(row.backgrounds):
                ax.text(offsets[j], 1.35, 'No qualifying\nbackgrounds', ha='center', fontsize=11, color=COLORS[policy])
            else:
                ax.text(offsets[j], row.median_horizon_days+.2, f"{row.median_horizon_days:g} days\n(n={int(row.backgrounds)})", ha='center', fontsize=13, color=COLORS[policy])
    ax.set(xticks=x, xticklabels=['Low','Medium','High'], xlabel='Class 1 no-show severity', ylabel='Median selected booking horizon (days)', ylim=(1,5), xlim=(-.5,2.5))
    ax.legend(loc='upper left', frameon=False)
    save(fig, 'slide29_horizon_background_best', ['Original Experiment 2, all original demand, share and capacity levels pooled. n = qualifying backgrounds.', 'Each horizon belongs to the selected case with the largest Class 1 gain. The 80% follow-up lacks case-level results.'])

    fig, ax = canvas('Reservation at each background\'s best Pareto case', 'Both flexible. Median of selected reservation percentages across qualifying backgrounds')
    g = horizon[horizon.policy == 'both_flexible'].set_index('noshow_level').reindex(LEVELS)
    ax.scatter(x, g.median_reservation_pct, color=COLORS['both_flexible'], marker='D', s=130)
    for j, (_, row) in enumerate(g.iterrows()):
        if pd.notna(row.backgrounds):
            ax.text(j, row.median_reservation_pct+3, f"{row.median_reservation_pct:.1f}%\n(n={int(row.backgrounds)})", ha='center', fontsize=15, color=COLORS['both_flexible'])
    ax.set(xticks=x, xticklabels=['Low','Medium','High'], xlabel='Class 1 no-show severity', ylabel='Median daily capacity reserved (%)', ylim=(0,55), xlim=(-.5,2.5))
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    save(fig, 'slide30_reservation_background_best', ['Reservation percentage = selected reserved slots / daily capacity x 100. Every background has equal weight.', 'Original Experiment 2, all original demand, share and capacity levels pooled. The 80% follow-up lacks case-level results.'])
    print(f'Selected {len(best)} cases from {len(cells.loc[flag(cells.validated_meaningful_pareto)])} qualifying cases.')
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT/'outputs/pareto_presentation')
    parser.add_argument('--combined-dir', type=Path, default=ROOT/'outputs/pareto_combined')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'figures/pareto_background_best')
    parser.add_argument('--summary-dir', type=Path, default=ROOT/'outputs/pareto_background_best')
    a = parser.parse_args()
    build(a.input_dir, a.combined_dir, a.output_dir, a.summary_dir)


if __name__ == '__main__':
    main()

