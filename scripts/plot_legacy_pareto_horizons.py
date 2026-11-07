from pathlib import Path
import shutil
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
repo=Path(__file__).resolve().parents[1]
p=repo/'outputs/pareto_presentation'
v=pd.read_csv(p/'validated_meaningful_pareto_cells.csv')
b=pd.read_csv(p/'patient_behavior_factorial_bank.csv')
v=v[v.validated_meaningful_pareto & v.policy.isin(['horizon_only','both_flexible'])].merge(b[['background_id','noshow_level']],on='background_id',validate='many_to_one')
t=v.groupby(['policy','background_id','noshow_level']).horizon_days.median().reset_index()
s=t.groupby(['noshow_level','policy']).agg(median_horizon_days=('horizon_days','median'),backgrounds=('background_id','size')).reset_index()
print(s.to_string(index=False))
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':13})
fig,ax=plt.subplots(figsize=(12,6.75),facecolor='white')
fig.subplots_adjust(left=.105,right=.96,bottom=.24,top=.76)
fig.text(.105,.93,'Median validated horizon by no-show post probability',fontsize=23,weight='bold',color='#192638')
fig.text(.105,.867,'Horizon only and both flexible | Original Experiment 2',fontsize=14,color='#596574')
levels=['low','medium','high']
for policy,offset,color,marker,label in [('horizon_only',-.12,'#257EAD','o','Horizon only'),('both_flexible',.12,'#7544A4','D','Both flexible')]:
    for i,lev in enumerate(levels):
        r=s[(s.policy==policy)&(s.noshow_level==lev)]
        if r.empty:
            ax.text(i+offset,1.75,'No validated\nscenarios',ha='center',va='center',fontsize=10,color=color)
            continue
        y=float(r.median_horizon_days.iloc[0]);n=int(r.backgrounds.iloc[0])
        ax.scatter(i+offset,y,s=115,color=color,marker=marker,label=label if i==0 or (policy=='horizon_only' and i==1) else None,zorder=3)
        ax.annotate(f'{y:g} days\n(n = {n})',(i+offset,y),xytext=(-13 if offset<0 else 13,18),textcoords='offset points',ha='right' if offset<0 else 'left',fontsize=12,color=color)
ax.set_xlim(-.5,2.5);ax.set_ylim(1.4,4.6)
ax.set_xticks(range(3),[x.title() for x in levels]);ax.set_yticks([2,2.5,3,3.5,4])
ax.set_ylabel('Median booking horizon (days)',labelpad=12,color='#253448')
ax.set_xlabel('Class 1 no-show post probability',labelpad=12,color='#253448')
ax.set_axisbelow(True);ax.grid(axis='y',color='#E3E7EB')
for sp in ['top','right']:ax.spines[sp].set_visible(False)
for sp in ['bottom','left']:ax.spines[sp].set_color('#C7CED5')
ax.tick_params(length=0,pad=8,colors='#4C5968')
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([],[],marker=m,color=c,linestyle='None',markersize=8,label=l) for m,c,l in [('o','#257EAD','Horizon only'),('D','#7544A4','Both flexible')]],loc='lower center',bbox_to_anchor=(.5,.09),frameon=False,ncol=2,fontsize=13)
fig.text(.105,.045,'Each scenario contributes its median horizon across validated settings; the plotted value is the median across scenarios.',fontsize=10,color='#596574')
fig.text(.105,.018,'n = qualifying scenarios. All original demand, share and capacity levels are pooled. The 80% follow-up lacks horizon-setting data.',fontsize=10,color='#596574')
out=repo/'figures/Pareto';out.mkdir(parents=True,exist_ok=True)
for ext in ['png']:
    target=out/f'pareto_median_horizon_by_noshow_severity.{ext}'
    fig.savefig(target,dpi=300,facecolor='white')
s.to_csv(repo/'outputs/pareto_background_best'/'pareto_median_horizon_by_noshow_severity.csv',index=False)
plt.close(fig)
print('Saved replacement PNG and SVG.')
