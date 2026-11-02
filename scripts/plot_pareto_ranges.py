"""Subtle observed min-max ranges for slide 27, 29 and 30 median figures.
Run: python scripts/plot_pareto_ranges.py
Uses one best Class 1 Pareto case per background, both-flexible only.
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
FIG=ROOT/'figures/pareto_background_best'
OUT=ROOT/'outputs/pareto_background_best'
BLUE='#0072B2'


def plot_range(frame,title,ylabel,stem,percent=False):
    fig,ax=plt.subplots(figsize=(12.8,7.2),layout='constrained')
    x=np.arange(len(frame)); med=frame['median'].to_numpy()
    assert ((frame['min']<=frame['median']) & (frame['median']<=frame['max'])).all()
    # Draw observed extrema behind the median markers, using faint dashed lines.
    result=ax.errorbar(x,med,yerr=np.vstack([med-frame['min'],frame['max']-med]),fmt='none',
        ecolor=matplotlib.colors.to_rgba(BLUE,.30),elinewidth=.9,capsize=5,capthick=.9,zorder=1)
    for col in result[2]: col.set_linestyle('--')
    ax.scatter(x,med,s=100,color=BLUE,zorder=3)
    for i,v in enumerate(med):
        ax.annotate(f'{v:.1f}'+('%' if percent else ''),(i,v),xytext=(12,0),textcoords='offset points',va='center',color=BLUE,fontsize=14)
    ax.set(title=title,ylabel=ylabel,xticks=x,xticklabels=frame.index,xlim=(-.5,len(frame)-.5))
    if len(frame)>1: ax.set_xlabel('Class 1 no-show severity')
    if percent: ax.yaxis.set_major_formatter(PercentFormatter(100))
    low=min(0,frame['min'].min()); high=frame['max'].max(); span=max(1,high-low)
    ax.set_ylim(low-.08*span,high+.12*span)
    ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    for ext in ['png','svg']: fig.savefig(FIG/f'{stem}.{ext}',dpi=200,facecolor='white')
    plt.close(fig)
    print('Generated',stem)


def main():
    FIG.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.labelsize':16,'axes.titlesize':19,'svg.hashsalt':'pareto-ranges'})
    cells=pd.read_csv(ROOT/'outputs/pareto_presentation/validated_meaningful_pareto_cells.csv')
    bank=pd.read_csv(ROOT/'outputs/pareto_presentation/patient_behavior_factorial_bank.csv')
    best=select_best(cells,bank)
    best=best[best.policy=='both_flexible'].copy()
    assert not best.background_id.duplicated().any()
    best.to_csv(OUT/'selected_best_both_flexible_range_cases.csv',index=False)
    util=best[best.rho<=2.5].eval_delta_utilization.mul(100).agg(['count','min','median','max']).to_frame().T
    util.index=['Both flexible']
    util.to_csv(OUT/'slide27_utilization_range.csv')
    plot_range(util,'Utilization change at the best Class 1 Pareto case','Utilization change vs baseline (pp)','slide27_utilization_background_best')
    for col,label,title,ylabel,stem in [
        ('horizon_days','horizon','Booking horizon at the best Class 1 Pareto case','Booking horizon (days)','slide29_horizon_background_best'),
        ('reservation_pct','reservation','Reservation at the best Class 1 Pareto case','Daily capacity reserved (%)','slide30_reservation_background_best')]:
        summary=best.groupby('noshow_level')[col].agg(['count','min','median','max']).reindex(['low','medium','high'])
        summary.index=['Low','Medium','High']
        summary.to_csv(OUT/f'{stem}_range.csv')
        plot_range(summary,title,ylabel,stem,percent=col=='reservation_pct')


if __name__=='__main__':main()
