"""Compare both-flexible Pareto and Class 2 loss-allowing settings for slides 33/34.
Run: python scripts/plot_policy_settings_comparison.py
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from plot_pareto_background_best import select_best

ROOT=Path(__file__).resolve().parents[1]
LEVELS=['low','medium','high']
SERIES=[('Pareto','#0072B2','o',-.10),('Class 2 loss allowed','#D55E00','D',.10)]


def main():
    figdir=ROOT/'figures/policy_settings_comparison';outdir=ROOT/'outputs/policy_settings_comparison'
    figdir.mkdir(parents=True,exist_ok=True);outdir.mkdir(parents=True,exist_ok=True)
    cells=pd.read_csv(ROOT/'outputs/pareto_presentation/validated_meaningful_pareto_cells.csv')
    bank=pd.read_csv(ROOT/'outputs/pareto_presentation/patient_behavior_factorial_bank.csv')
    pareto=select_best(cells,bank)
    pareto=pareto[pareto.policy=='both_flexible'].copy()
    pareto['result_group']='Pareto'
    d=pd.read_csv(ROOT/'outputs/experiment2_frontier/evaluated_selected_cells.csv')
    # One retrospective best evaluated Class 1 gain per background. Repeated
    # settings across budgets never contribute extra observations.
    flag=d.validated.astype(str).str.lower().map({'true':True,'false':False})
    if flag.isna().any():raise ValueError('Invalid validation flags')
    candidates=d[(d.policy=='both_flexible') & flag & (d.loss_budget<=.1+1e-9) & (d.delta_sr1>=.05-1e-9)].copy()
    needed=['delta_sr1','delta_sr2','horizon_days','Q','window','slots_per_day','loss_budget']
    if not np.isfinite(candidates[needed]).all().all():raise ValueError('Nonfinite selected metrics/settings')
    loss=candidates.sort_values(['background_id','delta_sr1','loss_budget','delta_sr2','horizon_days','Q','window'],ascending=[True,False,True,False,True,True,True],kind='stable').drop_duplicates('background_id').copy()
    assert not loss.background_id.duplicated().any()
    np.testing.assert_allclose(loss.set_index('background_id').delta_sr1.sort_index(),candidates.groupby('background_id').delta_sr1.max().sort_index())
    assert (loss.delta_sr2<0).all(), 'Some selected cases have no actual harm; revise explanation'
    loss['reservation_pct']=loss.Q/loss.slots_per_day*100
    loss['result_group']='Class 2 loss allowed'
    targets=pd.read_csv(ROOT/'outputs/experiment2_frontier/background_policy_targets.csv')
    expected=targets[(targets.policy=='both_flexible') & targets.achieves_5pp].background_id
    assert set(loss.background_id)==set(expected)
    assert len(pareto)==33 and len(loss)==519
    cols=['background_id','result_group','policy','noshow_level','rho','class1_share','slots_per_day','horizon_days','Q','window','reservation_pct']
    selected=pd.concat([pareto[cols],loss[cols]],ignore_index=True)
    assert not selected.duplicated(['result_group','background_id']).any()
    selected.to_csv(outdir/'selected_background_settings.csv',index=False)
    loss.to_csv(outdir/'selected_loss_allowing_cases.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.labelsize':16,'axes.titlesize':19,'svg.hashsalt':'policy-settings-comparison'})
    summary=[]
    for metric,title,ylabel,stem,ylim in [
      ('horizon_days','Selected booking horizons','Booking horizon (days)','slide33_horizon_pareto_vs_allowed_loss',(0,34)),
      ('reservation_pct','Selected reservation percentages','Daily capacity reserved (%)','slide34_reservation_pareto_vs_allowed_loss',(0,113))]:
        fig,ax=plt.subplots(figsize=(12.8,7.2),layout='constrained')
        for group,color,marker,offset in SERIES:
            g=selected[selected.result_group==group].groupby('noshow_level')[metric].agg(['count','min','median','max']).reindex(LEVELS)
            assert (g['min']<=g['median']).all() and (g['median']<=g['max']).all()
            for level,row in g.iterrows():summary.append({'metric':metric,'result_group':group,'noshow_level':level,**row.to_dict()})
            x=np.arange(3)+offset;med=g['median'].to_numpy()
            bars=ax.errorbar(x,med,yerr=np.vstack([med-g['min'],g['max']-med]),fmt='none',ecolor=matplotlib.colors.to_rgba(color,.25),elinewidth=.9,capsize=4,capthick=.9,zorder=1)
            for line in bars[2]:line.set_linestyle('--')
            ax.scatter(x,med,color=color,marker=marker,s=100,label=group,zorder=3)
            for xx,v in zip(x,med):
                ax.annotate(f'{v:g}'+('%' if metric=='reservation_pct' else ''),(xx,v),xytext=(-12 if offset<0 else 12,0),textcoords='offset points',ha='right' if offset<0 else 'left',va='center',fontsize=14,color=color)
        ax.set(title=title,xlabel='Class 1 no-show severity',ylabel=ylabel,xticks=np.arange(3),xticklabels=['Low','Medium','High'],xlim=(-.55,2.55),ylim=ylim)
        if metric=='reservation_pct':ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='upper left',frameon=False)
        for ext in ['png','svg']:fig.savefig(figdir/f'{stem}.{ext}',dpi=200,facecolor='white')
        plt.close(fig);print('Generated',stem)
    pd.DataFrame(summary).to_csv(outdir/'settings_summary.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__=='__main__':main()
