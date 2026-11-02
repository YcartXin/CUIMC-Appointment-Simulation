#!/usr/bin/env python3
"""Complete Pareto evaluation from the expanded frontier's shared discovery bank.

No horizon-only runs. Every discovery-qualifying genuine-reservation cell is
frozen and evaluated; independent evaluation never changes selection.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis import experiment2_frontier as f

COLUMNS=['background_id','policy']+f.CELL+['discovery_feasible','search_delta_sr1','search_delta_sr2']


def freeze_read(root):
    meta=json.loads((root/'pareto_freeze.json').read_text())
    for name,key in [('bank.csv','bank_sha256'),('selected_pareto_discovery.csv','selection_sha256')]:
        if f.digest(root/name)!=meta[key]:raise ValueError('Frozen Pareto bank/selection changed')
    return pd.read_csv(root/'selected_pareto_discovery.csv')


def catalog(a,bank,sources):
    if (a.root/'pareto_freeze.json').exists() or (a.root/'selected_pareto_discovery.csv').exists():
        raise ValueError('Pareto selection is immutable; use a new root')
    if (a.root/'bank.csv').exists():
        pd.testing.assert_frame_equal(pd.read_csv(a.root/'bank.csv'),bank,check_dtype=False)
    rows=[]
    for _,row in bank.iterrows():
        raw=f.raw_for(row,'search',sources,a.frontier_root)
        for policy in f.POLICIES:
            grid=f.cells(row)[policy]
            x=f.paired(raw,grid,f.SEARCH)
            g=x.groupby(f.CELL,as_index=False)[f.DELTA].mean()
            g=g[(g.delta_sr1>=0)&(g.delta_sr2>=0)&(g[['delta_sr1','delta_sr2']].max(axis=1)>=.005)]
            for _,c in g.iterrows():
                rows.append(dict(background_id=row.background_id,policy=policy,
                    **{k:int(c[k]) for k in f.CELL},discovery_feasible=True,
                    search_delta_sr1=c.delta_sr1,search_delta_sr2=c.delta_sr2))
    f.write_csv(bank,a.root/'bank.csv')
    f.write_csv(pd.DataFrame(rows,columns=COLUMNS),a.root/'selected_pareto_discovery.csv')
    meta=dict(bank_sha256=f.digest(a.root/'bank.csv'),selection_sha256=f.digest(a.root/'selected_pareto_discovery.csv'),
        source_manifest=json.loads(a.sources.read_text()),frontier_root=str(a.frontier_root.resolve()),
        search_seeds=f.SEARCH,evaluation_seeds=f.EVAL,
        selection='all discovery cells with dSR1>=0,dSR2>=0,max gain>=0.005; genuine Q>0,W>=2',
        grid='same systematic grid as expanded frontier; no adaptive reselection')
    (a.root/'pareto_freeze.json').write_text(json.dumps(meta,indent=2))
    print(f'Frozen {len(rows)} Pareto candidate cells across {len(bank)} scenarios')


def postprocess(a,bank,sources,selected):
    rows=[];seeds=[];rng=np.random.default_rng(20260928)
    for _,row in bank.iterrows():
        chosen=selected[selected.background_id==row.background_id]
        if chosen.empty:continue
        cs=list(chosen[f.CELL].drop_duplicates().itertuples(index=False,name=None))
        raw=f.raw_for(row,'evaluation',sources,a.root)
        x=f.paired(raw,cs,f.EVAL)
        for _,c in chosen.iterrows():
            z=x[(x.horizon_days==c.horizon_days)&(x.Q==c.Q)&(x.window==c.window)].sort_values('seed')
            if len(z)!=len(f.EVAL):raise ValueError('Incomplete Pareto evaluation')
            means=z[f.DELTA].mean();d1,d2,du=means
            result=c.to_dict()
            names={'delta_sr1':'eval_delta_sr1','delta_sr2':'eval_delta_sr2','delta_u':'eval_delta_utilization'}
            for metric,name in names.items():
                values=z[metric].to_numpy()
                draws=rng.integers(0,len(values),size=(2000,len(values)))
                lo,hi=np.quantile(values[draws].mean(axis=1),[.025,.975])
                result.update({name:means[metric],name+'_ci_low':lo,name+'_ci_high':hi})
            valid=d1>=-f.EPS and d2>=-f.EPS and max(d1,d2)>=.005
            nonharm=result['eval_delta_sr1_ci_low']>=-f.EPS and result['eval_delta_sr2_ci_low']>=-f.EPS
            positive=result['eval_delta_sr1_ci_low']>f.EPS or result['eval_delta_sr2_ci_low']>f.EPS
            result.update(validated_meaningful_pareto=valid,validated_strict_both_improve=d1>f.EPS and d2>f.EPS and max(d1,d2)>=.005,
                ci_supported_nonharm_both=nonharm,ci_supported_positive_gain=positive,strongly_supported_pareto=valid and nonharm and positive)
            rows.append(result)
            for _,v in z.iterrows():
                seeds.append(dict(background_id=row.background_id,policy=c.policy,
                    **{k:int(c[k]) for k in f.CELL},seed=int(v.seed),delta_sr1=v.delta_sr1,delta_sr2=v.delta_sr2,delta_utilization=v.delta_u))
    result=pd.DataFrame(rows,columns=COLUMNS+['eval_delta_sr1','eval_delta_sr1_ci_low','eval_delta_sr1_ci_high',
        'eval_delta_sr2','eval_delta_sr2_ci_low','eval_delta_sr2_ci_high','eval_delta_utilization','eval_delta_utilization_ci_low',
        'eval_delta_utilization_ci_high','validated_meaningful_pareto','validated_strict_both_improve',
        'ci_supported_nonharm_both','ci_supported_positive_gain','strongly_supported_pareto'])
    f.write_csv(result,a.root/'validated_meaningful_pareto_cells.csv')
    f.write_csv(pd.DataFrame(seeds,columns=['background_id','policy']+f.CELL+['seed','delta_sr1','delta_sr2','delta_utilization']),a.root/'validated_meaningful_pareto_seed_deltas.csv')
    universe=bank.merge(pd.DataFrame({'policy':f.POLICIES}),how='cross')
    counts=result.groupby(['background_id','policy']).agg(n_candidates=('background_id','size'),n_validated=('validated_meaningful_pareto','sum')).reset_index()
    universe=universe.merge(counts,on=['background_id','policy'],how='left',validate='one_to_one')
    for col in ['n_candidates','n_validated']:universe[col]=universe[col].fillna(0).astype(int)
    universe['validated_pareto_exists']=universe.n_validated>0
    f.write_csv(universe,a.root/'validated_background_policy_universe.csv')
    f.write_csv(bank,a.root/'patient_behavior_factorial_bank.csv')
    print(universe.groupby('policy').validated_pareto_exists.agg(['sum','count']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['catalog','inventory','evaluate','postprocess'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--frontier-root',type=Path,required=True)
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--shard',type=int)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.root.resolve()==a.frontier_root.resolve():p.error('Pareto and frontier output roots must differ')
    bank=pd.read_csv(a.frontier_root/'bank.csv')
    pd.testing.assert_frame_equal(bank,f.expanded_bank(include_share60=True),check_dtype=False)
    if a.shard is not None and not 1<=a.shard<=len(bank):p.error('Invalid shard')
    sources=f.source_catalog(a.sources)
    # Reuse the expanded frontier's newly completed search and evaluation too.
    mapping={f.signature(row):[str(row.background_id)] for _,row in bank.iterrows()}
    sources.append((mapping,{stage:[a.frontier_root/stage/'raw'] for stage in ['search','evaluation']}))
    if a.stage=='catalog':catalog(a,bank,sources);return
    meta=json.loads((a.root/'pareto_freeze.json').read_text())
    if meta['source_manifest']!=json.loads(a.sources.read_text()):raise ValueError('Source manifest changed after freeze')
    selected=freeze_read(a.root)
    pd.testing.assert_frame_equal(pd.read_csv(a.root/'bank.csv'),bank,check_dtype=False)
    if a.stage=='postprocess':postprocess(a,bank,sources,selected);return
    if a.stage=='evaluate':a.stage='evaluation'
    target=bank if a.shard is None else bank.iloc[a.shard-1:a.shard]
    f.plans(a,target,sources,'evaluation',selected)


if __name__=='__main__':main()
