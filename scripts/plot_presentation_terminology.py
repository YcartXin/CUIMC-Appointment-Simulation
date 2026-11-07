"""Regenerate the Policy and Pareto charts whose source lived outside the repo.
Run: python scripts/plot_presentation_terminology.py
Uses the original source tables and selections, with updated display wording.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def save(fig, section, stem):
    dest = ROOT / 'figures' / section
    dest.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ['png', 'svg']:
        fig.savefig(dest / f'{stem}.{ext}', dpi=200, facecolor='white')
    plt.close(fig)
    print('Generated', section, stem)

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.titlesize':18,'axes.labelsize':15})
    # Same aggregation as average_utilization_3x3_findings_led.qmd:
    # utilization objective, rho <= 2.5, median across clinic contexts/balking.
    pair = pd.read_csv(ROOT/'full_run_summaries/patient_behavior_factorial_3_5/release/pairwise_policy_deltas.csv')
    pair = pair[(pair.selection_objective == 'average_utilization') & (pair.rho <= 2.5)]
    labels = {'horizon_only_vs_baseline':'Horizon only', 'reservation_only_vs_baseline':'Reservation only', 'both_flexible_vs_baseline':'Both flexible'}
    levels = ['low','medium','high']
    summary = pair.groupby(['comparison','noshow_level']).delta_average_utilization.median()
    fig, ax = plt.subplots(figsize=(10.5,6))
    for comp, label in labels.items():
        ax.plot(range(3), summary.loc[comp].reindex(levels)*100, marker='o', lw=2, label=label)
    ax.set(title='Policy value rises with no-show post probability', xlabel='Class 1 no-show post probability', ylabel='Median utilization gain vs matched baseline (pp)', xticks=range(3), xticklabels=['5%','15%','25%'])
    ax.axhline(0,lw=.8); ax.legend(frameon=False); ax.grid(axis='y',alpha=.2)
    save(fig,'Policy','noshow_post_probability_policy_value')
    cells = pd.read_csv(ROOT/'outputs/pareto_presentation/validated_meaningful_pareto_cells.csv')
    valid = cells.validated_meaningful_pareto.astype(str).str.lower().map({'true':True,'false':False})
    if valid.isna().any(): raise ValueError('Invalid Pareto validation flag')
    cells = cells[valid]
    fig, ax = plt.subplots(figsize=(12.8,7.2))
    for policy, label, color, marker in [('horizon_only','Horizon only','#1f77b4','o'),('both_flexible','Both flexible','#ff7f0e','^')]:
        g = cells[cells.policy == policy]
        ax.scatter(g.eval_delta_sr1*100,g.eval_delta_sr2*100,color=color,marker=marker,alpha=.75,label=f'{label}: {len(g)} policies / {g.background_id.nunique()} scenarios')
    ax.axvline(.5,ls='--',lw=1,color='#6aa5ce'); ax.axhline(0,lw=.8); ax.axvline(0,lw=.8)
    ax.text(.52,.365,'0.5 pp meaningful-gain threshold',fontsize=10)
    ax.text(.98,.07,'Reservation-only: 0 validated policies',transform=ax.transAxes,ha='right',fontsize=11,bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
    ax.set(title='Validated Pareto served-rate improvements vs matched baseline',xlabel='Change in Class 1 served rate, ΔSR1 (percentage points)',ylabel='Change in Class 2 served rate, ΔSR2 (percentage points)')
    ax.legend(frameon=False); ax.grid(alpha=.15)
    save(fig,'Pareto','validated_scatter_paretos')

if __name__ == '__main__': main()
