"""Current presentation figures. PNG only; supply replacement datasets via flags.
python scripts/update_presentation_figures.py --help
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_pareto_background_best import select_best
ROOT=Path(__file__).resolve().parents[1]
COLORS=['#0072B2','#D55E00','#009E73']
LEVELS=['low','medium','high']
rng=np.random.default_rng(20261103)

def flag(s):
 v=s.astype(str).str.lower().map({'true':True,'false':False})
 if v.isna().any():raise ValueError('Invalid validation flag')
 return v

def save(fig,section,stem):
 dest=ROOT/'figures'/section;dest.mkdir(parents=True,exist_ok=True)
 fig.tight_layout();fig.savefig(dest/(stem+'.png'),dpi=200,facecolor='white');plt.close(fig)
 (dest/(stem+'.svg')).unlink(missing_ok=True)
 print('Generated',section,stem)

def interval(x,stat=np.mean):
 x=np.asarray(x);return np.quantile([stat(rng.choice(x,len(x),replace=True)) for _ in range(1500)],[.025,.975])

def main():
 import sys
 completed=ROOT/'outputs/experiment2_ratio_horizon/tradeoff/freeze.json'
 if len(sys.argv)==1 and completed.exists():
  from plot_completed_experiment2 import main as completed_main
  completed_main();return
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--pareto-dir',type=Path,default=ROOT/'outputs/pareto_presentation')
 ap.add_argument('--frontier-dir',type=Path,default=ROOT/'outputs/experiment2_frontier')
 ap.add_argument('--policy-release',type=Path,default=ROOT/'full_run_summaries/patient_behavior_factorial_3_5/release')
 a=ap.parse_args();plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':17,'axes.labelsize':14})
 bank=pd.read_csv(a.pareto_dir/'patient_behavior_factorial_bank.csv');cells=pd.read_csv(a.pareto_dir/'validated_meaningful_pareto_cells.csv');valid=cells[flag(cells.validated_meaningful_pareto)]
 frontier=pd.read_csv(a.frontier_dir/'evaluated_selected_cells.csv')
 # Slide 17: table only, source probabilities explicit, design coverage data-driven.
 design=pd.read_csv(a.frontier_dir/'background_policy_targets.csv').drop_duplicates('background_id')
 fig,ax=plt.subplots(figsize=(12.8,7.2));ax.axis('off')
 rows=[['No-show pre probability','5% (both classes)'],['Balking pre probability','5% (both classes)'],['Class 1 no-show post probability','5%, 15%, 25%'],['Class 1 balking post probability','10%, 20%, 30%'],['Demand / capacity (ρ)',', '.join(f'{v:g}' for v in sorted(design.rho.unique()))],['Class 1 arrival share',', '.join(f'{v:.0%}' for v in sorted(set(design.class1_share.unique()) | {.6}))],['System size (slots/day)',', '.join(str(v) for v in sorted(design.slots_per_day.unique()))]]
 table=ax.table(cellText=rows,colLabels=['Varying parameters and fixed pre probabilities','Values'],cellLoc='left',colLoc='left',bbox=[0,.18,1,.78],colWidths=[.52,.48]);table.auto_set_font_size(False);table.set_fontsize(15)
 ax.text(0,.09,'9 Class 1 behavior profiles = 3 no-show post probabilities × 3 balking post probabilities',fontsize=15,transform=ax.transAxes)
 ax.text(0,.02,f'Planned design: {9 * design.rho.nunique() * len(set(design.class1_share.unique()) | {.6}) * design.slots_per_day.nunique():,} scenarios; currently {len(design):,} in the frontier dataset',fontsize=17,weight='bold',transform=ax.transAxes)
 save(fig,'Policy','slide17_design_parameters')
 # Slide 19: paired seed bootstrap of medians over fixed clinic scenarios and balking.
 seed=pd.read_csv(a.policy_release/'pairwise_policy_deltas_seed.csv');seed=seed[seed.selection_objective.eq('average_utilization')]
 comparisons=['horizon_only_vs_baseline','reservation_only_vs_baseline','both_flexible_vs_baseline'];labels=['Horizon only','Reservation only','Both flexible']
 fig,ax=plt.subplots(figsize=(12.8,7.2))
 for comp,label,color in zip(comparisons,labels,COLORS):
  xs=[];ys=[];los=[];his=[]
  for rho,g in seed[seed.comparison.eq(comp)].groupby('rho'):
   m=g.pivot(index=['noshow_level','background_id'],columns='seed',values='delta_average_utilization');levels=m.index.get_level_values(0)
   def contrast(vals):return 100*(np.median(vals[levels=='high'])-np.median(vals[levels=='low']))
   point=contrast(m.mean(axis=1).to_numpy());boots=[contrast(m.iloc[:,rng.integers(0,m.shape[1],m.shape[1])].mean(axis=1).to_numpy()) for _ in range(1000)];lo,hi=np.quantile(boots,[.025,.975]);xs.append(rho);ys.append(point);los.append(min(lo,point));his.append(max(hi,point))
  ax.errorbar(xs,ys,yerr=[np.array(ys)-los,np.array(his)-ys],marker='o',capsize=4,label=label,color=color)
 ax.set(title='Demand activates the no-show post probability effect',xlabel='Demand / capacity (ρ)',ylabel='High minus low no-show post probability\nMedian utilization gain across balking and clinic scenarios (pp)');ax.legend();ax.grid(alpha=.2)
 fig.text(.5,.005,'95% paired-seed bootstrap intervals; fixed scenario design',ha='center',fontsize=10);fig.subplots_adjust(bottom=.2)
 save(fig,'Policy','slide19_demand_activation')
 # Slide 20: bootstrap clinic contexts, retaining all three balking settings together.
 interval_rows=[]
 sel=pd.read_csv(a.policy_release/'selected_policy_outcomes.csv');sel=sel[(sel.selection_objective=='average_utilization') & (sel.rho<=2.5) & (sel.policy!='baseline')]
 fig,axes=plt.subplots(1,2,figsize=(14,6))
 for policy,label,color in zip(['horizon_only','reservation_only','both_flexible'],labels,COLORS):
  for ax,col,ylabel in zip(axes,['selected_horizon_days','average_utilization'],['Mean selected booking horizon (days)','Mean utilization (%)']):
   if col=='selected_horizon_days' and policy=='reservation_only': continue
   means=[];bounds=[]
   for level in LEVELS:
    g=sel[(sel.policy==policy)&(sel.noshow_level==level)];vals=g.groupby('clinic_context_id')[col].mean().to_numpy();vals=vals*(100 if col=='average_utilization' else 1);means.append(vals.mean());bounds.append(interval(vals));interval_rows.append({'policy':policy,'post_probability':{'low':.05,'medium':.15,'high':.25}[level],'metric':col,'mean':means[-1],'ci_low':bounds[-1][0],'ci_high':bounds[-1][1]})
   bounds=np.array(bounds);ax.errorbar(np.arange(3)+{'horizon_only':-.04,'reservation_only':0,'both_flexible':.04}[policy],means,yerr=[np.array(means)-bounds[:,0],bounds[:,1]-means],marker='o',markersize=4,capsize=7,elinewidth=2,capthick=2,color=color,label=label);ax.set(xticks=range(3),xticklabels=['5%','15%','25%'],xlabel='Class 1 no-show post probability',ylabel=ylabel);ax.grid(alpha=.2)
 axes[0].legend();axes[1].legend();fig.suptitle('Selected horizon and utilization by no-show post probability');fig.text(.5,.005,'95% clinic-context bootstrap intervals; balking levels retained together. Horizons fixed after selection.',ha='center',fontsize=10)
 save(fig,'Policy','slide20_horizon_utilization')
 interval_dir=ROOT/'outputs/presentation_figures';interval_dir.mkdir(parents=True,exist_ok=True);pd.DataFrame(interval_rows).to_csv(interval_dir/'slide20_intervals.csv',index=False)
 # Slide 24: all supplied validated policy settings, with dynamic counts.
 fig,ax=plt.subplots(figsize=(12.8,7.2))
 for policy,label,color,marker in zip(['horizon_only','reservation_only','both_flexible'],labels,['#0072B2','#777777','#D55E00'],['o','s','^']):
  g=valid[valid.policy==policy];ax.scatter(g.eval_delta_sr1*100,g.eval_delta_sr2*100,alpha=.8,color=color,marker=marker,edgecolors='white',linewidths=.35,label=f'{label}: {len(g)} policies / {g.background_id.nunique()} scenarios')
 ax.axvline(.5,ls='--',color='grey');ax.axhline(0,color='grey');ax.set(title='Validated Pareto served-rate improvements',xlabel='Class 1 served-rate gain (pp)',ylabel='Class 2 served-rate gain (pp)');ax.legend();ax.grid(alpha=.15)
 save(fig,'Pareto','validated_scatter_paretos')
 # Slide 25: common full denominator from supplied universe, not only successes.
 u=pd.read_csv(a.pareto_dir/'validated_background_policy_universe.csv');u=u[u.rho<=2.5].copy();u['success']=flag(u.validated_pareto_exists)
 fig,ax=plt.subplots(figsize=(12.8,7.2))
 for offset,policy,label,color in zip([-.2,0,.2],['horizon_only','reservation_only','both_flexible'],labels,COLORS):
  rates=u[u.policy==policy].groupby('noshow_level').success.mean().reindex(LEVELS)*100;ax.bar(np.arange(3)+offset,rates,.2,label=label,color=color)
 ax.set(xticks=range(3),xticklabels=['5%','15%','25%'],xlabel='Class 1 no-show post probability',ylabel='Scenarios with a Pareto improvement (%)',title='Pareto success by no-show post probability');ax.legend();ax.grid(axis='y',alpha=.15)
 fig.text(.5,.005,f'Common original scenario coverage; ρ ≤ 2.5. {u.background_id.nunique()} scenarios per policy. Partial 80% follow-up excluded.',ha='center',fontsize=10)
 save(fig,'Pareto','slide25_pareto_success')
 best=select_best(cells,bank);p=best[best.policy=='both_flexible'].copy();p['gain_pp']=100*p.eval_delta_sr1;p['group']='Pareto'
 candidates=frontier[(frontier.policy=='both_flexible') & flag(frontier.validated) & (frontier.loss_budget<=.1+1e-9)&(frontier.delta_sr1>=.05-1e-9)]
 loss=candidates.sort_values(['background_id','delta_sr1','loss_budget','delta_sr2','horizon_days','Q','window'],ascending=[True,False,True,False,True,True,True],kind='stable').drop_duplicates('background_id').copy();loss['gain_pp']=loss.delta_sr1*100;loss['group']='Class 2 loss allowed'
 # Slide 30: distributions of one selected policy per scenario.
 fig,ax=plt.subplots(figsize=(12.8,7.2));rhos=sorted(frontier.rho.unique())
 for frame,off,color in [(p,-.17,COLORS[0]),(loss,.17,COLORS[1])]:
  data=[];pos=[]
  for i,rho in enumerate(rhos):
   vals=frame[frame.rho==rho].gain_pp
   if len(vals):data.append(vals);pos.append(i+off)
  boxes=ax.boxplot(data,positions=pos,widths=.28,patch_artist=True,manage_ticks=False,medianprops={'color':'black','linewidth':1.6})
  for box in boxes['boxes']:box.set_facecolor(color);box.set_alpha(.65)
 
 for i,rho in enumerate(rhos):
  if p[p.rho==rho].empty:ax.annotate('No qualifying\nPareto scenarios',(i-.17,0),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8,color=COLORS[0])
 ax.plot([],[],color=COLORS[0],lw=8,label='Pareto');ax.plot([],[],color=COLORS[1],lw=8,label='Class 2 loss allowed');ax.legend();ax.set(xticks=range(len(rhos)),xticklabels=rhos,xlabel='Demand / capacity (ρ)',ylabel='Class 1 served-rate gain (pp)',title='Class 1 gains across scenarios');save(fig,'Trade-off','slide30_class1_gain_boxplots')
 # Slide 32: minimum observed actual Class 2 loss to reach 5pp, once per scenario/policy.
 f=frontier[flag(frontier.validated)&(frontier.loss_budget<=.1+1e-9)&(frontier.delta_sr1>=.05-1e-9)].copy();f['actual_loss']=np.maximum(0,-100*f.delta_sr2)
 chosen=f.sort_values('actual_loss').drop_duplicates(['background_id','policy']);bins=[-1e-9,0,1,2,5,10+1e-7];names=['0','>0–1','>1–2','>2–5','>5–10'];chosen['loss_bin']=pd.cut(chosen.actual_loss,bins,labels=names)
 fig,ax=plt.subplots(figsize=(12.8,7.2))
 for j,policy in enumerate(['reservation_only','both_flexible']):
  g=chosen[chosen.policy==policy];counts=g.loss_bin.value_counts().reindex(names,fill_value=0);ax.bar(np.arange(5)+(j-.5)*.35,counts,.35,color=COLORS[j],label=policy.replace('_',' ').title())
 ax.set(xticks=range(5),xticklabels=names,xlabel='Minimum observed actual Class 2 served-rate loss to reach ≥5pp Class 1 gain (pp)',ylabel='Successful scenarios',title='Policy success by Class 2 trade-off');ax.legend();save(fig,'Trade-off','slide32_success_by_class2_loss')
 # Companion to slide 35: same selected settings as reservation percentages.
 p['reservation_pct']=p.Q/p.slots_per_day*100;loss['reservation_pct']=loss.Q/loss.slots_per_day*100
 for metric,stem,ylabel in [('horizon_days','slide33_horizon_pareto_vs_allowed_loss','Booking horizon (days)'),('reservation_pct','slide34_reservation_pareto_vs_allowed_loss','Daily capacity reserved (%)')]:
  fig,ax=plt.subplots(figsize=(12.8,7.2))
  for frame,off,color,label in [(p,-.1,COLORS[0],'Pareto'),(loss,.1,COLORS[1],'Class 2 loss allowed')]:
   g=frame.groupby('noshow_level')[metric].agg(['median','min','max']).reindex(LEVELS);med=g['median'].to_numpy()
   ax.errorbar(np.arange(3)+off,med,yerr=[med-g['min'],g['max']-med],fmt='o',color=color,capsize=4,label=label)
  ax.set(xticks=range(3),xticklabels=['5%','15%','25%'],xlabel='Class 1 no-show post probability',ylabel=ylabel,title='Selected policies: '+ylabel.lower());ax.legend()
  fig.text(.5,.005,'Markers: medians. Bars: observed minimum–maximum across selected scenarios (not confidence intervals).',ha='center',fontsize=10)
  save(fig,'Trade-off/policy_settings_comparison',stem)
 fig,ax=plt.subplots(figsize=(12.8,7.2))
 for frame,off,color,label in [(p,-.17,COLORS[0],'Pareto'),(loss,.17,COLORS[1],'Class 2 loss allowed')]:
  data=[frame[frame.noshow_level==level].window for level in LEVELS];boxes=ax.boxplot(data,positions=np.arange(3)+off,widths=.28,patch_artist=True,manage_ticks=False,medianprops={'color':'black','linewidth':1.6})
  for box in boxes['boxes']:box.set_facecolor(color);box.set_alpha(.65)
  ax.plot([],[],color=color,lw=8,label=label)
 ax.set(xticks=range(3),xticklabels=['5%','15%','25%'],xlabel='Class 1 no-show post probability',ylabel='Reservation period (days)',title='Reservation periods of selected policies');ax.legend();save(fig,'Trade-off','slide35_reservation_periods')
 assert len(chosen)==chosen.groupby(['policy','loss_bin'],observed=False).size().sum(), 'Unbinned successful scenarios'
 print('Pareto coverage:',len(bank),'scenarios. Supply --pareto-dir with completed replacement tables for added simulations.')

if __name__=='__main__':main()
