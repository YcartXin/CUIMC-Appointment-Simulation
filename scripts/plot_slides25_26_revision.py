"""Slide 25 legacy Pareto plus 80% follow-up; slide 26 scale comparison.
Run: python scripts/plot_slides25_26_revision.py
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT/'figures/Pareto/slides25_26_revision'
OUT = ROOT/'outputs/slides25_26_revision'
COLORS = {30:'#0072B2',40:'#009E73',50:'#D55E00'}


def boolean(s):
    b=s.astype(str).str.lower().map({'true':True,'false':False})
    if b.isna().any(): raise ValueError(f'Invalid flag: {s.name}')
    return b


def canvas(title, ylabel):
    fig,ax=plt.subplots(figsize=(12.8,7.2),layout='constrained')
    ax.set(title=title,ylabel=ylabel)
    ax.grid(axis='y',alpha=.22)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    return fig,ax


def save(fig,name):
    for ext in ['png']:
        destination = ROOT/'figures/Trade-off/scale' if 'frontier' in name else FIG
        destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination/f'{name}.{ext}',dpi=200,facecolor='white')
    plt.close(fig)
    print('Generated',name)


def scale_plot(summary,title,name):
    fig,ax=canvas(title,'Successful scenarios (%)')
    for slots,marker in [(30,'o'),(40,'D'),(50,'s')]:
        g=summary[summary.slots_per_day==slots].sort_values('class1_share')
        ax.plot(g.class1_share,g.rate*100,marker=marker,color=COLORS[slots],linewidth=4 if slots==30 else 2.6,markersize=11 if slots==30 else 8,markerfacecolor='white' if slots==30 else COLORS[slots],
                linestyle=({30:'-',40:':',50:'--'}[slots]) if len(g)>1 else 'None',label=f'{slots} slots/day')
    ax.set(xlabel='Class 1 share of arrivals',xticks=sorted(summary.class1_share.unique()),ylim=(-3,105 if summary.rate.max()>.7 else 65))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.legend(frameon=False)
    save(fig,name)


def main():
    FIG.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.titlesize':19,'axes.labelsize':16,'svg.hashsalt':'slides25-26-revision'})
    c=pd.read_csv(ROOT/'outputs/pareto_combined/combined_background_policy_universe.csv')
    if c.duplicated(['background_id','policy']).any(): raise ValueError('Duplicate scenario-policy rows')
    c['success']=boolean(c.validated_pareto_exists)
    # Only the legacy pipeline feeds the Pareto figures. Keep original headline
    # rho<=2.5 and pool original/follow-up backgrounds with equal weight.
    old=c[(c.policy=='both_flexible') & (c.rho<=2.5)].copy()
    groups=old.groupby(['data_source','noshow_level']).success.agg(backgrounds='size',successful='sum',rate='mean').reset_index()
    groups.to_csv(OUT/'slide25_source_noshow_rates.csv',index=False)
    sources=sorted(old.data_source.unique())
    if len(sources)!=2 or 'original_experiment2' not in sources: raise ValueError('Expected original and follow-up sources')
    follow=next(s for s in sources if s!='original_experiment2')
    if len(old[old.data_source==follow])!=54: raise ValueError('Expected 54 legacy follow-up scenarios')
    if old.background_id.duplicated().any(): raise ValueError('Repeated slide 25 scenario')
    pooled=old.groupby('noshow_level').success.agg(backgrounds='size',successful='sum',rate='mean').reindex(['low','medium','high'])
    if not pooled.backgrounds.eq(168).all(): raise ValueError('Expected 168 scenarios per post probability')
    pooled.to_csv(OUT/'slide25_combined_noshow_rates.csv')
    fig,ax=canvas('Pareto scenarios by Class 1 no-show post probability','Scenarios with a Pareto improvement (%)')
    x=np.arange(3)
    bars=ax.bar(x,pooled.rate*100,width=.55,color='#0072B2')
    for bar,(_,row) in zip(bars,pooled.iterrows()):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.5,f"{row.rate*100:.1f}%",ha='center',fontsize=15)
    ax.set(xticks=x,xticklabels=['Low','Medium','High'],xlabel='Class 1 no-show post probability',ylim=(0,25))
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    save(fig,'slide25_legacy_pareto_with_80pct')
    comparable=c[(c.policy=='both_flexible') & c.rho.isin([1.7,2.0])]
    legacy=comparable.groupby(['class1_share','slots_per_day']).success.agg(backgrounds='size',successful='sum',rate='mean').reset_index()
    assert legacy.backgrounds.eq(18).all()
    legacy.to_csv(OUT/'slide26_legacy_pareto_scale.csv',index=False)
    scale_plot(legacy,'Pareto scenarios by Class 1 share and system size','slide26_pareto_scale')
    t=pd.read_csv(ROOT/'outputs/experiment2_frontier/background_policy_targets.csv')
    if t.duplicated(['background_id','policy']).any(): raise ValueError('Duplicate frontier targets')
    t=t[(t.policy=='both_flexible') & t.rho.isin([1.7,2.0])].copy()
    t['success']=boolean(t.achieves_5pp)
    if not (t.success==t.min_budget_5pp.notna()).all(): raise ValueError('Budget and success mismatch')
    if not t.loc[t.success,'min_budget_5pp'].between(0,.1+1e-9).all(): raise ValueError('Budget exceeds 10pp')
    frontier=t.groupby(['class1_share','slots_per_day']).success.agg(backgrounds='size',successful='sum',rate='mean').reset_index()
    assert len(frontier)==18 and frontier.backgrounds.eq(18).all()
    frontier.to_csv(OUT/'slide26_frontier_5pp_scale.csv',index=False)
    # Audit actual harm separately from permitted harm. A background gets one vote
    # even when several tested budgets qualify.
    d=pd.read_csv(ROOT/'outputs/experiment2_frontier/evaluated_selected_cells.csv')
    d=d[(d.policy=='both_flexible') & d.rho.isin([1.7,2.0])].copy()
    d['harm_success']=boolean(d.validated) & (d.delta_sr1>=.05-1e-9) & (d.delta_sr2<0) & (d.loss_budget<=.1+1e-9)
    actual=d.groupby('background_id').harm_success.max().rename('actual_harm_success')
    audit=t.merge(actual,on='background_id',validate='one_to_one')
    if audit.actual_harm_success.isna().any(): raise ValueError('Missing evaluated scenario')
    if not audit.actual_harm_success.equals(audit.success): raise ValueError('Actual harm and allowed-harm success differ; revise report')
    audit.to_csv(OUT/'slide26_frontier_background_audit.csv',index=False)
    scale_plot(frontier,'Class 1 gain ≥5pp within a 10pp Class 2 loss budget','slide26_frontier_allowed_loss_scale')
    print(groups.to_string(index=False))
    print(frontier.to_string(index=False))


if __name__=='__main__': main()

