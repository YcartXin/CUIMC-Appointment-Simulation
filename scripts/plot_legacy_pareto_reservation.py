from pathlib import Path
import shutil
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
repo=Path(__file__).resolve().parents[1]
p=repo/'outputs/pareto_presentation'
v=pd.read_csv(p/'validated_meaningful_pareto_cells.csv')
b=pd.read_csv(p/'patient_behavior_factorial_bank.csv')
v=v[v.validated_meaningful_pareto & v.policy.eq('both_flexible')].merge(b[['background_id','noshow_level','slots_per_day']],on='background_id',validate='many_to_one')
v['reservation_percentage']=100*v.Q/v.slots_per_day
bg=v.groupby(['background_id','noshow_level']).reservation_percentage.median().reset_index()
s=bg.groupby('noshow_level').agg(median_reservation_percentage=('reservation_percentage','median'),backgrounds=('background_id','size')).reindex(['low','medium','high']).reset_index()
assert s.backgrounds.sum()==33 and len(v)==139
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':13})
fig,ax=plt.subplots(figsize=(12,6.75),facecolor='white')
fig.subplots_adjust(left=.105,right=.96,bottom=.24,top=.76)
fig.text(.105,.93,'Median reservation percentage by no-show post probability',fontsize=22,weight='bold',color='#192638')
fig.text(.105,.867,'Both flexible | Original Experiment 2',fontsize=14,color='#596574')
x=range(3); y=s.median_reservation_percentage
ax.scatter(x,y,s=130,color='#7544A4',marker='D',zorder=3)
for i,r in s.iterrows():
    ax.annotate(f'{r.median_reservation_percentage:.1f}%\n(n = {int(r.backgrounds)})',(i,r.median_reservation_percentage),xytext=(0,17),textcoords='offset points',ha='center',fontsize=13,color='#7544A4')
ax.set_xlim(-.5,2.5);ax.set_ylim(0,max(50,float(y.max())+15))
ax.set_xticks(range(3),['Low','Medium','High'])
ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
ax.set_ylabel('Median daily capacity reserved',labelpad=12,color='#253448')
ax.set_xlabel('Class 1 no-show post probability',labelpad=12,color='#253448')
ax.set_axisbelow(True);ax.grid(axis='y',color='#E3E7EB')
for sp in ['top','right']:ax.spines[sp].set_visible(False)
for sp in ['bottom','left']:ax.spines[sp].set_color('#C7CED5')
ax.tick_params(length=0,pad=8,colors='#4C5968')
fig.text(.105,.11,'Reservation percentage = reserved slots / daily capacity × 100. Each scenario receives equal weight.',fontsize=11,color='#596574')
fig.text(.105,.073,'Each scenario contributes its median across validated settings; points show the median across scenarios. n = scenarios.',fontsize=10,color='#596574')
fig.text(.105,.036,'All original demand, share and capacity levels are pooled. The 80% follow-up lacks reservation-setting data.',fontsize=10,color='#596574')
out=repo/'figures/Pareto';out.mkdir(parents=True,exist_ok=True)
for ext in ['png']:
    f=out/f'pareto_median_reservation_percentage_by_noshow_severity.{ext}'
    fig.savefig(f,dpi=300,facecolor='white')
s.to_csv(repo/'outputs/pareto_background_best'/'pareto_median_reservation_percentage_by_noshow_severity.csv',index=False)
plt.close(fig)
print(s.to_string(index=False))
