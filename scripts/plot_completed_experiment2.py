"""Validate completed ratio/horizon outputs and regenerate current study figures.
Run: python scripts/plot_completed_experiment2.py
No policy reselection from independent trade-off evaluation outcomes.
"""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from plot_pareto_background_best import select_best
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/experiment2_ratio_horizon'
OUT=DATA/'summaries'
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73'
LEVELS=['low','medium','high'];KEY=['background_id','policy']


def flag(s):
    v=s.astype(str).str.lower().map({'true':True,'false':False})
    if (v.isna() & s.notna()).any():raise ValueError(f'Invalid flag {s.name}')
    return v.fillna(False)


def load():
    t=DATA/'tradeoff';p=DATA/'pareto'
    for folder,meta_name,items in [(t,'freeze.json',{'bank.csv':'bank_sha256','selected_discovery.csv':'selection_sha256'}),(p,'pareto_freeze.json',{'patient_behavior_factorial_bank.csv':'bank_sha256','selected_pareto_discovery.csv':'selection_sha256'})]:
        meta=json.loads((folder/meta_name).read_text())
        assert meta['allow_horizon_only'] is True
        for name,key in items.items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==meta[key]
    meta=json.loads((t/'freeze.json').read_text());assert meta['selection_rule']=='min_loss_per_gain' and meta['minimum_c1_gain']==.05
    bank=pd.read_csv(t/'bank.csv');pb=pd.read_csv(p/'patient_behavior_factorial_bank.csv')
    pd.testing.assert_frame_equal(bank,pb,check_dtype=False,rtol=1e-10,atol=1e-12)
    assert len(bank)==1134 and not bank.background_id.duplicated().any()
    assert set(bank.class1_share)=={.1,.3,.5,.6,.7,.8,.9} and set(bank.slots_per_day)=={30,40,50}
    for folder in [t,p]:
        for name in ['evaluation_coverage.csv']+(['search_coverage.csv'] if folder==t else []):
            c=pd.read_csv(folder/'inventory'/name);assert len(c)==1134 and c.pending.sum()==0
    f=pd.read_csv(t/'evaluated_selected_cells.csv');sel=pd.read_csv(t/'selected_discovery.csv')
    assert len(f)==20412 and not f.duplicated(KEY+['loss_budget']).any()
    assert set(f.background_id)==set(bank.background_id)
    matched=f.merge(sel,on=KEY+['loss_budget'],validate='one_to_one',suffixes=('_eval','_frozen'))
    for col in ['horizon_days','Q','window']:
        np.testing.assert_allclose(matched[col+'_eval'],matched[col+'_frozen'],equal_nan=True)
    feasible=flag(f.discovery_feasible);assert flag(f.loc[feasible,'evaluation_complete']).all()
    assert f.loc[feasible,'discovery_delta_sr1'].ge(.05-1e-12).all()
    np.testing.assert_allclose(f.loc[feasible,'discovery_loss_per_gain'],-f.loc[feasible,'discovery_delta_sr2']/f.loc[feasible,'discovery_delta_sr1'])
    f['success']=flag(f.validated)&(f.delta_sr1>=.05)&(f.delta_sr2>=-f.loss_budget-1e-12)
    f['gain_pp']=100*f.delta_sr1;f['loss_pp']=-100*f.delta_sr2;f['reservation_pct']=100*f.Q/f.slots_per_day
    # No reservation is displayed as zero days, not the engine sentinel -1.
    f['reservation_days']=np.where(f.Q==0,0,f.window)
    f['group']='Class 2 loss allowed'
    pc=pd.read_csv(p/'validated_meaningful_pareto_cells.csv');u=pd.read_csv(p/'validated_background_policy_universe.csv')
    assert len(pc)==3115 and not pc.duplicated(KEY+['horizon_days','Q','window']).any()
    valid=(pc.eval_delta_sr1>=-1e-12)&(pc.eval_delta_sr2>=-1e-12)&(pc[['eval_delta_sr1','eval_delta_sr2']].max(axis=1)>=.005)
    assert valid.equals(flag(pc.validated_meaningful_pareto))
    # Check independent seed coverage and reported means for both experiments.
    for folder,name,keys,metrics,reported in [(t,'evaluated_seed_deltas.csv',KEY+['loss_budget'],['delta_sr1','delta_sr2','delta_u'],f.loc[feasible]),(p,'validated_meaningful_pareto_seed_deltas.csv',KEY+['horizon_days','Q','window'],['delta_sr1','delta_sr2','delta_utilization'],pc)]:
        seed=pd.read_csv(folder/name);assert not seed.duplicated(keys+['seed']).any()
        assert set(seed.seed)==set(range(2000,2020))
        assert seed.groupby(keys).seed.nunique().eq(20).all()
        means=seed.groupby(keys)[metrics].mean().sort_index()
        renamed=reported.rename(columns={'eval_delta_sr1':'delta_sr1','eval_delta_sr2':'delta_sr2','eval_delta_utilization':'delta_utilization'})
        check=renamed.set_index(keys)[metrics].sort_index()
        pd.testing.assert_frame_equal(means,check,check_dtype=False,check_names=False,rtol=1e-9,atol=1e-12)
    best=select_best(pc,pb);best=best[best.policy=='both_flexible'].copy()
    best['gain_pp']=100*best.eval_delta_sr1;best['loss_pp']=-100*best.eval_delta_sr2;best['group']='Pareto'
    best['reservation_days']=np.where(best.Q==0,0,best.window)
    assert len(best)==117
    assert len(u)==2268 and not u.duplicated(KEY).any()
    expected=pc.loc[valid].groupby(KEY).size().rename('n_validated').reset_index()
    check=u[KEY+['n_validated']].merge(expected,on=KEY,how='left',suffixes=('_file','_check')).fillna(0)
    assert check.n_validated_file.eq(check.n_validated_check).all()
    return bank,f,pc,u,best


def canvas(title,ylabel):
    fig,ax=plt.subplots(figsize=(12.8,7.2),layout='constrained');ax.set(title=title,ylabel=ylabel)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    return fig,ax


def save(fig,path):
    dest=ROOT/'figures'/f'{path}.png';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        archive=ROOT/'archive/pre_completed_ratio_figures_20261109'/dest.relative_to(ROOT/'figures')
        if not archive.exists():archive.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,archive)
    fig.savefig(dest,dpi=200,facecolor='white');plt.close(fig);print('Generated',dest.relative_to(ROOT))


def ranges(ax,x,g,color,marker,label=None):
    y=g['median'].to_numpy();v=np.isfinite(y)
    bars=ax.errorbar(np.asarray(x)[v],y[v],yerr=np.vstack([y[v]-g['min'].to_numpy()[v],g['max'].to_numpy()[v]-y[v]]),fmt='none',ecolor=matplotlib.colors.to_rgba(color,.24),elinewidth=.8,capsize=3,zorder=1)
    for line in bars[2]:line.set_linestyle('--')
    ax.scatter(np.asarray(x)[v],y[v],s=65,color=color,marker=marker,label=label,zorder=3)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.labelsize':15,'axes.titlesize':18})
    bank,f,pc,u,p=load();bf=f[f.policy=='both_flexible'].copy()
    primary=bf[bf.loss_budget==.1].copy();loss=primary[primary.success].copy()
    assert not loss.background_id.duplicated().any()
    p.to_csv(OUT/'pareto_one_policy_per_scenario.csv',index=False)
    primary.to_csv(OUT/'tradeoff_frozen_10pp_all_scenarios.csv',index=False)
    loss.to_csv(OUT/'tradeoff_10pp_successful_scenarios.csv',index=False)
    first=bf[bf.success].sort_values(['background_id','loss_budget']).drop_duplicates('background_id').copy()
    first['budget_pp']=100*first.loss_budget;first.to_csv(OUT/'first_successful_tested_budget.csv',index=False)
    summaries=[]
    # One policy per scenario, including horizon-only choices under both-flexible.
    for metric,title,ylabel,path in [('horizon_days','Selected booking horizons','Booking horizon (days)','Trade-off/policy_settings_comparison/slide33_horizon_pareto_vs_allowed_loss'),('reservation_pct','Selected reservation percentages','Daily capacity reserved (%)','Trade-off/policy_settings_comparison/slide34_reservation_pareto_vs_allowed_loss'),('reservation_days','Selected reservation periods','Reservation period (days)','Trade-off/slide35_reservation_periods')]:
        fig,ax=canvas(title,ylabel)
        for frame,color,marker,off,label in [(p,BLUE,'o',-.1,'Pareto'),(loss,ORANGE,'D',.1,'Class 2 loss allowed')]:
            g=frame.groupby('noshow_level')[metric].agg(['count','min','median','max']).reindex(LEVELS)
            ranges(ax,np.arange(3)+off,g,color,marker,label)
            for i,v in enumerate(g['median']):
                if np.isfinite(v):ax.annotate(f'{v:g}'+('%' if metric=='reservation_pct' else ''),(i+off,v),xytext=(-10 if off<0 else 10,0),textcoords='offset points',ha='right' if off<0 else 'left',va='center',color=color,fontsize=12)
            summaries.append(g.reset_index().assign(metric=metric,group=label))
        ax.set(xticks=range(3),xticklabels=['5%','15%','25%'],xlabel='Class 1 no-show post probability',xlim=(-.5,2.5));ax.set_ylim(bottom=-1);ax.margins(y=.18);ax.legend(frameon=False)
        if metric=='reservation_pct':ax.yaxis.set_major_formatter(PercentFormatter(100));ax.set_ylim(-3,112)
        save(fig,path)
    pd.concat(summaries).to_csv(OUT/'policy_setting_summaries.csv',index=False)
    # Gain distributions/ranges by rho; empty groups are gaps, not zeros.
    rhos=sorted(bank.rho.unique())
    fig,ax=canvas('Class 1 gains across scenarios','Class 1 served-rate gain vs baseline (pp)')
    for frame,color,off,label in [(p,BLUE,-.16,'Pareto'),(loss,ORANGE,.16,'Class 2 loss allowed')]:
        vals=[frame.loc[frame.rho==rho,'gain_pp'].to_numpy() for rho in rhos];pos=[i+off for i,v in enumerate(vals) if len(v)]
        boxes=ax.boxplot([v for v in vals if len(v)],positions=pos,widths=.27,patch_artist=True,manage_ticks=False,medianprops={'color':'black'})
        for box in boxes['boxes']:box.set_facecolor(color);box.set_alpha(.65)
        ax.plot([],[],color=color,lw=7,label=label)
    ax.set(xticks=range(len(rhos)),xticklabels=rhos,xlabel='Demand / capacity (ρ)');ax.legend(frameon=False);save(fig,'Trade-off/slide30_class1_gain_boxplots')
    fig,ax=canvas('Class 1 served-rate gains by demand level','Class 1 served-rate gain vs baseline (pp)')
    gain=[]
    for frame,color,marker,off,label in [(p,BLUE,'o',-.10,'Pareto'),(loss,ORANGE,'D',.10,'Class 2 loss allowed')]:
        g=frame.groupby('rho').gain_pp.agg(['count','min','median','max']).reindex(rhos);ranges(ax,np.arange(len(rhos))+off,g,color,marker,label);gain.append(g.reset_index().assign(group=label))
    pd.concat(gain).to_csv(OUT/'gain_by_rho_summary.csv',index=False)
    ax.set(xticks=range(len(rhos)),xticklabels=rhos,xlabel='Demand / capacity (ρ)');ax.legend(frameon=False);save(fig,'Trade-off/policy_settings_comparison/class1_served_rate_gain_by_rho')
    # Pareto: common full denominator, only both-flexible, one scenario vote.
    up=u[(u.policy=='both_flexible')&(u.rho<=2.5)].copy();up['success']=flag(up.validated_pareto_exists)
    g=up.groupby('noshow_level').success.agg(['sum','size','mean']).reindex(LEVELS);g.to_csv(OUT/'pareto_success_by_noshow.csv')
    fig,ax=canvas('Pareto success by no-show post probability','Scenarios with a Pareto improvement (%)');ax.bar(range(3),100*g['mean'],color=BLUE,width=.55);ax.set(xticks=range(3),xticklabels=['5%','15%','25%'],xlabel='Class 1 no-show post probability');ax.yaxis.set_major_formatter(PercentFormatter(100));save(fig,'Pareto/slide25_pareto_success')
    fig,ax=canvas('Selected Pareto served-rate improvements','Class 2 served-rate gain (pp)');ax.scatter(p.gain_pp,-p.loss_pp,color=BLUE,s=24,alpha=.7);ax.set_xlabel('Class 1 served-rate gain (pp)');save(fig,'Pareto/validated_scatter_paretos')
    fig,ax=canvas('Utilization at selected Pareto policies','Utilization gain vs baseline (pp)');g=(100*p.eval_delta_utilization).agg(['min','median','max']).to_frame().T;ranges(ax,[0],g,BLUE,'o');ax.set(xticks=[0],xticklabels=['Both flexible'],xlim=(-.5,.5));save(fig,'Pareto/best_cases/slide27_utilization_background_best')
    # Same rho range for Pareto/trade-off system-size comparisons.
    for frame,value,path,title in [(u[u.policy=='both_flexible'].assign(success=lambda x:flag(x.validated_pareto_exists)),'success','Pareto/slide26_pareto_scale','Pareto success by patient share and system size'),(primary,'success','Trade-off/scale/slide26_frontier_allowed_loss_scale','Trade-off success by patient share and system size')]:
        g=frame[frame.rho.isin([1.7,2.0])].groupby(['class1_share','slots_per_day'])[value].agg(['size','sum','mean']).reset_index();g.to_csv(OUT/(Path(path).stem+'.csv'),index=False)
        fig,ax=canvas(title,'Successful scenarios (%)')
        for slots,color,marker,style in [(30,BLUE,'o','-'),(40,GREEN,'D',':'),(50,ORANGE,'s','--')]:
            z=g[g.slots_per_day==slots];ax.plot(z.class1_share,100*z['mean'],color=color,marker=marker,ls=style,label=f'{slots} slots/day')
        ax.set(xlabel='Class 1 arrival share',xticks=sorted(bank.class1_share.unique()),ylim=(-3,103));ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(100));ax.legend(frameon=False);save(fig,path)
    # Trade-off success by tested budget, fixed frozen choices and full denominator.
    g=bf.groupby('loss_budget').success.agg(['sum','size','mean']);g.to_csv(OUT/'tradeoff_success_by_budget.csv')
    fig,ax=canvas('Success by allowed Class 2 loss budget','Successful scenarios (%)');ax.plot(g.index*100,g['mean']*100,color=ORANGE,marker='o');ax.set_xlabel('Class 2 loss budget (pp)');ax.yaxis.set_major_formatter(PercentFormatter(100));save(fig,'Trade-off/success_rate_by_tested_budget')
    observed=bf[bf.success].copy();observed['nonnegative_loss_pp']=observed.loss_pp.clip(lower=0)
    observed=observed.sort_values(['background_id','nonnegative_loss_pp','loss_budget']).drop_duplicates('background_id')
    labels=['0','>0–1','>1–2','>2–5','>5–10'];observed['loss_bin']=pd.cut(observed.nonnegative_loss_pp,[-1e-9,0,1,2,5,10+1e-9],labels=labels)
    counts=observed.loss_bin.value_counts().reindex(labels,fill_value=0);assert counts.sum()==len(observed)
    observed.to_csv(OUT/'minimum_observed_loss_among_tested_choices.csv',index=False)
    fig,ax=canvas('Success by observed Class 2 loss','Successful scenarios');ax.bar(range(5),counts,color=ORANGE,width=.6);ax.set(xticks=range(5),xticklabels=labels,xlabel='Minimum observed Class 2 loss to reach ≥5pp Class 1 gain (pp)');save(fig,'Trade-off/slide32_success_by_class2_loss')
    shares=sorted(bank.class1_share.unique())
    fig,ax=canvas('Trade-off success by patient mix and demand','Successful scenarios (%)')
    for rho,z in primary.groupby('rho'):
        g=z.groupby('class1_share').success.mean().reindex(shares);ax.plot(shares,100*g,marker='o',label=f'ρ={rho:g}')
    ax.set(xlabel='Class 1 arrival share',xticks=shares,ylim=(-3,103));ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(100));ax.legend(ncol=2,frameon=False);save(fig,'Trade-off/experiment2_frontier/fig1_bf_success_rate_by_share_and_rho')
    fig,ax=canvas('Minimum tested Class 2 budget for success','Minimum tested budget (pp)')
    for i,rho in enumerate(rhos):
        g=first[first.rho==rho].groupby('class1_share').budget_pp.agg(['min','median','max']).reindex(shares);color=plt.get_cmap('tab10')(i);x=np.asarray(shares)+(i-2.5)*.004;ax.plot(x,g['median'],color=color,lw=1.3);ranges(ax,x,g,color,'o',f'ρ={rho:g}')
    ax.set(xlabel='Class 1 arrival share',xticks=shares,ylim=(-.3,10.7));ax.xaxis.set_major_formatter(PercentFormatter(1));ax.legend(ncol=2,frameon=False);save(fig,'Trade-off/experiment2_frontier/fig2_bf_min_budget_by_share_and_rho')
    failure=primary.groupby(['rho','class1_share']).success.mean().unstack().reindex(index=rhos,columns=shares)
    fig,ax=canvas('Trade-off failure within the 10pp ceiling','Demand / capacity (ρ)');values=100*(1-failure.to_numpy());im=ax.imshow(values,aspect='auto',vmin=0,vmax=100,cmap='Blues')
    for i in range(len(rhos)):
        for j in range(len(shares)):ax.text(j,i,f'{values[i,j]:.0f}%',ha='center',va='center',color='white' if values[i,j]>60 else 'black',fontsize=12)
    ax.set(xticks=range(len(shares)),xticklabels=[f'{v:.0%}' for v in shares],yticks=range(len(rhos)),yticklabels=rhos,xlabel='Class 1 arrival share');fig.colorbar(im,ax=ax,label='Failed scenarios (%)');save(fig,'Trade-off/experiment2_frontier/fig3_bf_failure_heatmap_under_10pp_ceiling')
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    for ax,col,title in zip(axes,['budget_pp','loss_pp'],['Minimum tested budget','Actual Class 2 loss at that budget']):
        v=np.sort(first[col].to_numpy());ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',color=ORANGE);ax.set(title=title,xlabel='Percentage points',ylabel='Cumulative share of successful scenarios');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.15)
    save(fig,'Trade-off/experiment2_frontier/fig5_required_budget_distribution_by_policy')
    # Refresh remaining Pareto rate figure and the current presentation aliases.
    pu=u[u.policy=='both_flexible'].copy();pu['success']=flag(pu.validated_pareto_exists)
    pr=pu.groupby('rho').success.mean().reindex(rhos)*100
    fig,ax=canvas('Pareto success by demand level','Successful scenarios (%)');ax.bar(np.arange(len(rhos)),pr,color=BLUE,width=.55);ax.set(xticks=range(len(rhos)),xticklabels=rhos,xlabel='Demand / capacity (ρ)');ax.yaxis.set_major_formatter(PercentFormatter(100));save(fig,'Pareto/pareto_by_rho')
    for alias in ['Pareto/pareto_scale_by_class1_share.png','Pareto/best_cases/slide26_scale_background_best.png']:
        dest=ROOT/'figures'/alias
        if dest.exists():
            old=ROOT/'archive/pre_completed_ratio_figures_20261109'/alias
            if not old.exists():old.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,old)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'figures/Pareto/slide26_pareto_scale.png',dest)
    # The old policy-exclusive plot combines policy families and is no longer
    # part of the requested both-flexible-only presentation.
    stale=ROOT/'figures/Trade-off/experiment2_frontier/fig6_bf_only_backgrounds_breakdown.png'
    if stale.exists():
        old=ROOT/'archive/pre_completed_ratio_figures_20261109'/stale.relative_to(ROOT/'figures')
        old.parent.mkdir(parents=True,exist_ok=True)
        if not old.exists():shutil.copy2(stale,old)
        stale.unlink()
    fig,ax=plt.subplots(figsize=(12.8,7.2),layout='constrained');ax.axis('off')
    rows=[['Class 1 no-show post probability','5%, 15%, 25%','3'],['Class 1 balking post probability','10%, 20%, 30%','3'],['Demand / capacity (ρ)',', '.join(f'{v:g}' for v in rhos),'6'],['Class 1 arrival share',', '.join(f'{v:.0%}' for v in shares),'7'],['System size (slots/day)','30, 40, 50','3']]
    table=ax.table(cellText=rows,colLabels=['Varying parameter','Values','Levels'],colWidths=[.42,.49,.09],cellLoc='left',colLoc='left',bbox=[0,.22,1,.68]);table.auto_set_font_size(False);table.set_fontsize(15)
    ax.set_title('The 3 × 3 behavior design and clinic scenarios',fontsize=20,pad=15)
    ax.text(0,.09,'3 × 3 × 6 × 7 × 3 = 1,134 scenarios',fontsize=23,weight='bold',transform=ax.transAxes)
    save(fig,'Policy/slide17_design_parameters')
    report=dict(scenarios=len(bank),pareto_successful_scenarios=len(p),pareto_selected_horizon_only=int(p.Q.eq(0).sum()),tradeoff_10pp_successful_scenarios=len(loss),tradeoff_10pp_discovery_feasible=int(flag(primary.discovery_feasible).sum()),tradeoff_10pp_selected_horizon_only=int(loss.Q.eq(0).sum()),first_successful_budget_scenarios=len(first))
    (OUT/'validation_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
