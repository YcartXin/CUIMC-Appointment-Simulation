"""Class 1 served-rate gains versus matched baseline, grouped by rho.
Run: python scripts/plot_class1_gains_by_rho.py
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_pareto_background_best import select_best

ROOT=Path(__file__).resolve().parents[1]
SERIES=[('Pareto','#0072B2','o',-.10),('Class 2 loss allowed','#D55E00','D',.10)]


def main():
    figures=ROOT/'figures/policy_settings_comparison';out=ROOT/'outputs/policy_settings_comparison'
    figures.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    p=select_best(pd.read_csv(ROOT/'outputs/pareto_presentation/validated_meaningful_pareto_cells.csv'),pd.read_csv(ROOT/'outputs/pareto_presentation/patient_behavior_factorial_bank.csv'))
    p=p[p.policy=='both_flexible'].copy();p['gain_pp']=p.eval_delta_sr1*100;p['group']='Pareto'
    d=pd.read_csv(ROOT/'outputs/experiment2_frontier/evaluated_selected_cells.csv')
    flags=d.validated.astype(str).str.lower().map({'true':True,'false':False})
    if flags.isna().any():raise ValueError('Invalid validation flags')
    candidates=d[(d.policy=='both_flexible') & flags & (d.loss_budget<=.1+1e-9)&(d.delta_sr1>=.05-1e-9)]
    l=candidates.sort_values(['background_id','delta_sr1','loss_budget','delta_sr2','horizon_days','Q','window'],ascending=[True,False,True,False,True,True,True],kind='stable').drop_duplicates('background_id').copy()
    l['gain_pp']=l.delta_sr1*100;l['group']='Class 2 loss allowed'
    assert len(p)==33 and len(l)==519
    np.testing.assert_allclose(l.set_index('background_id').delta_sr1.sort_index(),candidates.groupby('background_id').delta_sr1.max().sort_index())
    all_data=pd.concat([p[['background_id','group','rho','gain_pp']],l[['background_id','group','rho','gain_pp']]],ignore_index=True)
    assert not all_data.duplicated(['group','background_id']).any()
    if not np.isfinite(all_data[['rho','gain_pp']]).all().all():raise ValueError('Nonfinite gain/rho')
    all_data.to_csv(out/'class1_gains_selected_backgrounds.csv',index=False)
    summary=all_data.groupby(['group','rho']).gain_pp.agg(['count','min','median','max']).reset_index()
    summary.to_csv(out/'class1_gains_by_rho_summary.csv',index=False)
    rhos=sorted(d.rho.unique());x=np.arange(len(rhos))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.labelsize':16,'axes.titlesize':19,'svg.hashsalt':'class1-gains-rho'})
    fig,ax=plt.subplots(figsize=(12.8,7.2),layout='constrained')
    for group,color,marker,offset in SERIES:
        g=summary[summary.group==group].set_index('rho').reindex(rhos)
        valid=g['count'].notna();positions=x[valid]+offset
        med=g.loc[valid,'median'].to_numpy();lo=g.loc[valid,'min'].to_numpy();hi=g.loc[valid,'max'].to_numpy()
        bars=ax.errorbar(positions,med,yerr=np.vstack([med-lo,hi-med]),fmt='none',ecolor=matplotlib.colors.to_rgba(color,.25),elinewidth=.9,capsize=4,capthick=.9,zorder=1)
        for line in bars[2]:line.set_linestyle('--')
        ax.scatter(positions,med,color=color,marker=marker,s=85,label=group,zorder=3)
        for xx,v in zip(positions,med):
            ax.annotate(f'{v:.1f}',(xx,v),xytext=(10,0),textcoords='offset points',va='center',fontsize=13,color=color)
    ax.set(title='Class 1 served-rate gains by demand level',xlabel=r'Demand / capacity ($\rho$)',ylabel='Class 1 served-rate gain vs baseline (pp)',xticks=x,xticklabels=[f'{v:g}' for v in rhos],xlim=(-.45,len(rhos)-.45),ylim=(-1,55))
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,loc='upper left')
    for ext in ['png','svg']:fig.savefig(figures/f'class1_served_rate_gain_by_rho.{ext}',dpi=200,facecolor='white')
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__=='__main__':main()
