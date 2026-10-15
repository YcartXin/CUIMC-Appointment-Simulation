#!/usr/bin/env python3
"""Full-background, discovery-frozen Experiment 2 access frontiers.

Reuse sources are explicit bank/raw-directory pairs in JSON. No source is changed.
Selection reads discovery seeds only; evaluation never changes the frozen choices.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.patient_behavior_factorial import (
    _row_config_kwargs, _task_grid, q_coarse_grid, window_coarse_grid, run_tasks,
)
from experiments.patient_behavior_factorial_bank import generate_bank, _profiles
from analysis.experiment2_share80_followup import bank_frame as followup_bank

BUDGETS = (0., .005, .01, .02, .03, .04, .05, .075, .10)
POLICIES = ('reservation_only', 'both_flexible')
CELL = ['horizon_days', 'Q', 'window']
KEY = CELL + ['seed']
METRICS = ['class_1_percent_serviced', 'class_2_percent_serviced', 'average_utilization']
DELTA = ['delta_sr1', 'delta_sr2', 'delta_u']
SEARCH = tuple(range(1000, 1005))
EVAL = tuple(range(2000, 2020))
EPS = 1e-12

def signature(row):
    """Match simulation inputs, never rely on context IDs across banks."""
    d = _row_config_kwargs(row)
    return tuple((k, round(float(v), 12)) for k, v in sorted(d.items()))

def expanded_bank():
    old = pd.concat([generate_bank(), followup_bank()], ignore_index=True)
    lookup = {signature(r): r.to_dict() for _, r in old.iterrows()}
    rows = []
    for p, rho, share, cap in product(_profiles(), (1.2,1.4,1.7,2.,2.5,3.),
                                      (.1,.3,.5,.7,.8,.9), (30,40,50)):
        context = f'E2X_R{rho:g}_S{round(share*100)}_C{cap}'
        r = dict(**p, background_id=f"PBF_{p['profile_id']}_{context}",
                 clinic_context_id=context, horizon_days=100, rho=rho,
                 class1_share=share, slots_per_day=cap, lambda_1=rho*cap*share,
                 lambda_2=rho*cap*(1-share), cap_thresholds_to_horizon=False,
                 design_note='experiment2_frontier_expansion',
                 patient_characteristic='balk_noshow_factorial',
                 class2_reference='fixed_late_thresholds')
        rows.append(lookup.get(signature(pd.Series(r)), r))
    bank = pd.DataFrame(rows).sort_values('background_id').reset_index(drop=True)
    assert len(bank) == 972 and bank.background_id.nunique() == 972
    return bank

def cells(row):
    q = q_coarse_grid(int(row.slots_per_day))
    # Historical genuine-reservation catalog excludes W=1 in both families.
    return {'baseline': [(100,0,-1)],
            'reservation_only': [(100,x,w) for x in q for w in window_coarse_grid(100) if w>=2],
            'both_flexible': [(h,x,w) for h in range(2,31) for x in q
                              for w in window_coarse_grid(h) if w>=2]}

def source_catalog(path):
    specs = json.loads(path.read_text())
    sources = []
    for s in specs:
        if not s.get('simulation_convention_verified', False):
            raise ValueError('Verify burn-in=30, measure=365, cooldown=max(14,H), release semantics in source manifest')
        bank = pd.read_csv(s['bank'])
        mapping = {}
        for _, r in bank.iterrows():
            mapping.setdefault(signature(r), []).append(str(r.background_id))
        dirs = {stage: [Path(x) for x in s.get(stage, [])] for stage in ('search','evaluation')}
        for paths in dirs.values():
            for p in paths:
                if not p.is_dir(): raise FileNotFoundError(p)
        sources.append((mapping, dirs))
    return sources

def read_raw(paths, seeds):
    frames = []
    for p in dict.fromkeys(paths):
        if not p.exists(): continue
        x = pd.read_csv(p)
        if not set(KEY+METRICS) <= set(x): raise ValueError(f'Invalid raw schema: {p}')
        x = x[x.seed.isin(seeds)].copy()
        if x.empty: continue
        x = x[KEY+METRICS]
        if not np.isfinite(x.to_numpy(dtype=float)).all():
            raise ValueError(f'Nonfinite/incomplete raw rows: {p}')
        if not (x[KEY].to_numpy() == x[KEY].to_numpy().astype(int)).all():
            raise ValueError(f'Noninteger cell/seed: {p}')
        frames.append(x)
    if not frames: return pd.DataFrame(columns=KEY+METRICS)
    x = pd.concat(frames, ignore_index=True)
    grouped = x.groupby(KEY)[METRICS]
    spread = grouped.max()-grouped.min()
    if (spread > EPS).any().any():
        raise ValueError('Conflicting duplicate simulations; inspect source conventions before reuse')
    return x.drop_duplicates(KEY).sort_values(KEY).reset_index(drop=True)

def raw_for(row, stage, sources, root):
    paths = []
    for mapping, dirs in sources:
        for bg in mapping.get(signature(row), []):
            paths += [d/f'{bg}.csv' for d in dirs[stage]]
    paths.append(root/stage/'raw'/f'{row.background_id}.csv')
    return read_raw(paths, SEARCH if stage=='search' else EVAL)

def expected(c, seeds):
    return {(h,q,w,s) for h,q,w in c for s in seeds}

def present(raw):
    return set(raw[KEY].itertuples(index=False, name=None))

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def frozen(root, bank):
    p = root/'selected_discovery.csv'
    meta = json.loads((root/'freeze.json').read_text())
    if digest(p)!=meta['selection_sha256'] or digest(root/'bank.csv')!=meta['bank_sha256']:
        raise ValueError('Frozen selection or bank changed; use a new output root')
    return pd.read_csv(p)

def write_csv(x, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    x.to_csv(tmp, index=False)
    tmp.replace(path)

def bank_stage(a):
    bank = expanded_bank()
    if (a.root/'freeze.json').exists(): raise ValueError('Frozen root: do not rewrite bank')
    write_csv(bank, a.root/'bank.csv')
    old = set(pd.concat([generate_bank(),followup_bank()]).background_id)
    write_csv(bank[~bank.background_id.isin(old)], a.root/'new_bank_entries.csv')
    print('972 backgrounds; 594 existing bank identities; 378 new entries (raw coverage is separate)')

def plans(a, bank, sources, stage, selection=None):
    report = []
    missing_dir = a.root/'inventory'/f'{stage}_missing'
    if a.stage=='inventory': missing_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in bank.iterrows():
        if stage=='search': grid = cells(row)
        else:
            chosen = selection[(selection.background_id==row.background_id) & selection.discovery_feasible]
            cs = sorted(set(chosen[CELL].itertuples(index=False, name=None)))
            grid = {'evaluation': cs + ([(100,0,-1)] if cs else [])}
        raw = raw_for(row, stage, sources, a.root)
        have = present(raw)
        missing, planned = [], 0
        for policy, cs in grid.items():
            want = expected(cs, SEARCH if stage=='search' else EVAL)
            pending = sorted(want-have)
            planned += len(want)
            missing += [dict(policy=policy, **dict(zip(KEY,k))) for k in pending]
        report.append(dict(background_id=row.background_id, shard=idx+1,
                           planned=planned, reused=planned-len(missing), pending=len(missing)))
        if a.stage=='inventory':
            write_csv(pd.DataFrame(missing, columns=['policy']+KEY), missing_dir/f'{row.background_id}.csv')
        if a.stage==stage and not a.dry_run and (a.shard is None or a.shard==idx+1):
            tasks = []
            for m in missing:
                tasks += _task_grid(row, stage=m['policy'], phase='frontier_systematic' if stage=='search' else 'frontier_frozen_evaluation',
                                    cells=[tuple(m[k] for k in CELL)], seeds=[m['seed']], smoke=False)
            if tasks:
                print(f'{row.background_id}: running {len(tasks)} missing rows', flush=True)
                run_tasks(tasks, a.root/stage/'raw'/f'{row.background_id}.csv', 1, flush_every=20)
    out = pd.DataFrame(report)
    if a.stage=='inventory':
        write_csv(out, a.root/'inventory'/f'{stage}_coverage.csv')
        (a.root/'inventory'/f'{stage}_shards.txt').write_text(''.join(f'{s}\n' for s in out.loc[out.pending>0,'shard']))
    print(stage, out[['planned','reused','pending']].sum().to_dict(), flush=True)
    return out

def paired(raw, cs, seeds):
    if not expected(cs+[(100,0,-1)],seeds)<=present(raw):
        raise ValueError('Incomplete policy/baseline seed coverage')
    b = raw[(raw.horizon_days==100)&(raw.Q==0)&(raw.window==-1)]
    x = raw.merge(b[['seed']+METRICS], on='seed', suffixes=('','_baseline'), validate='many_to_one')
    for m,d in zip(METRICS,DELTA): x[d] = x[m]-x[m+'_baseline']
    keys = set(cs)
    return x[[tuple(r) in keys for r in x[CELL].itertuples(index=False,name=None)]]

def choose(g, budget):
    z = g[g.delta_sr2 >= -budget-EPS]
    if z.empty: return z
    # Final window ordering resolves all remaining ties deterministically.
    return z.sort_values(['delta_sr1','delta_sr2','average_utilization','q_fraction','horizon_days','window'],
                         ascending=[False,False,False,True,False,True],kind='stable').head(1)

def select_stage(a, bank, sources):
    if (a.root/'selected_discovery.csv').exists() or (a.root/'freeze.json').exists():
        raise ValueError('Selection is immutable; choose a new root to select again')
    selected, all_frontier, old = [], [], []
    for _, row in bank.iterrows():
        raw = raw_for(row,'search',sources,a.root)
        grid = cells(row)
        for policy in POLICIES:
            x = paired(raw,grid[policy],SEARCH)
            g = x.groupby(CELL,as_index=False)[METRICS+DELTA].mean()
            g['q_fraction'] = g.Q/row.slots_per_day
            # Exact discovery nondominated curve, with the entire C2-gain region kept.
            curve = g[g.delta_sr2>=-.10-EPS].sort_values(['delta_sr2','delta_sr1'],ascending=False)
            best = -np.inf
            for _, c in curve.iterrows():
                if c.delta_sr1 > best:
                    all_frontier.append(dict(background_id=row.background_id,policy=policy,**c.to_dict()))
                    best = c.delta_sr1
            for loss in BUDGETS:
                z = choose(g,loss)
                d = dict(background_id=row.background_id,policy=policy,loss_budget=loss,discovery_feasible=not z.empty)
                d.update(z.iloc[0].to_dict() if not z.empty else {c:np.nan for c in CELL+METRICS+DELTA+['q_fraction']})
                selected.append(d)
            strict = g[(g.delta_sr1>=-EPS)&(g.delta_sr2>=-EPS)&((g.delta_sr1>=.005-EPS)|(g.delta_sr2>=.005-EPS))]
            z = choose(strict,0.)
            old.append(dict(background_id=row.background_id,policy=policy,legacy_meaningful_discovery=not z.empty,
                            **(z.iloc[0].to_dict() if not z.empty else {})))
    write_csv(pd.DataFrame(selected),a.root/'selected_discovery.csv')
    write_csv(pd.DataFrame(all_frontier),a.root/'discovery_nondominated.csv')
    write_csv(pd.DataFrame(old),a.root/'legacy_rule_discovery_best.csv')
    (a.root/'freeze.json').write_text(json.dumps(dict(selection_sha256=digest(a.root/'selected_discovery.csv'),
        bank_sha256=digest(a.root/'bank.csv'), search_seeds=SEARCH,evaluation_seeds=EVAL,
        source_manifest=json.loads(a.sources.read_text()),
        budgets=BUDGETS,selection='max dSR1; dSR2; utilization; lower Q/slots; longer H; lower W',
        grid='systematic coarse; H=100 reservation; H=2..30 both; Q step5; W historical >=2'),indent=2))

def postprocess(a,bank,sources):
    sel = frozen(a.root,bank)
    rows, seed_rows = [], []
    for _, row in bank.iterrows():
        choices = sel[sel.background_id==row.background_id]
        cs = sorted(set(choices.loc[choices.discovery_feasible,CELL].itertuples(index=False,name=None)))
        raw = raw_for(row,'evaluation',sources,a.root)
        x = paired(raw,cs,EVAL) if cs else pd.DataFrame()
        for _, c in choices.iterrows():
            r = c.to_dict()
            for d in DELTA: r['discovery_'+d]=r.pop(d)
            for m in METRICS: r['discovery_'+m]=r.pop(m)
            r.update(evaluation_complete=False,evaluation_feasible=False,validated=False,
                     strict_pareto=False,meaningful_pareto=False,practical_frontier=False)
            for d in DELTA: r[d]=np.nan
            if c.discovery_feasible:
                z = x[(x.horizon_days==c.horizon_days)&(x.Q==c.Q)&(x.window==c.window)]
                r.update(z[DELTA].mean().to_dict())
                r.update(z[METRICS].mean().to_dict())
                r['evaluation_complete']=len(z)==20
                r['evaluation_feasible']=r['delta_sr2']>=-c.loss_budget-EPS
                r['validated']=r['evaluation_complete'] and r['evaluation_feasible']
                r['strict_pareto']=r['delta_sr1']>=-EPS and r['delta_sr2']>=-EPS
                r['meaningful_pareto']=r['strict_pareto'] and max(r['delta_sr1'],r['delta_sr2'])>=.005-EPS
                r['practical_frontier']=r['validated'] and r['delta_u']>=-.005-EPS
                rng = np.random.default_rng(20260928)
                draws = rng.integers(0,20,size=(2000,20))
                boot = z[DELTA].to_numpy()[draws].mean(axis=1)
                for j,d in enumerate(DELTA):
                    r[d+'_ci_low'],r[d+'_ci_high']=np.quantile(boot[:,j],[.025,.975])
                r['statistical_c2_support']=r['delta_sr2_ci_low']>=-c.loss_budget
                for _, v in z.iterrows():
                    seed_rows.append(dict(background_id=row.background_id,policy=c.policy,loss_budget=c.loss_budget,
                                          seed=int(v.seed),**{d:v[d] for d in DELTA}))
            rows.append(r)
    result = pd.DataFrame(rows).merge(bank,on='background_id',suffixes=('','_bank'),validate='many_to_one')
    result['H']=result.horizon_days; result['W']=result.window
    result['q_fraction']=result.Q/result.slots_per_day
    write_csv(result,a.root/'evaluated_selected_cells.csv')
    write_csv(pd.DataFrame(seed_rows),a.root/'evaluated_seed_deltas.csv')
    for dims in ([],['rho'],['class1_share'],['slots_per_day'],['noshow_level'],['balk_level'],
                 ['rho','class1_share','slots_per_day','noshow_level','balk_level']):
        keys=['policy','loss_budget']+dims
        summary=result.groupby(keys,dropna=False).agg(backgrounds=('background_id','size'),
            discovered=('discovery_feasible','sum'),validated=('validated','sum'),practical=('practical_frontier','sum'),
            mean_delta_sr1=('delta_sr1','mean'),median_delta_sr1=('delta_sr1','median'),
            mean_delta_sr2=('delta_sr2','mean'),median_delta_sr2=('delta_sr2','median'),
            mean_delta_u=('delta_u','mean')).reset_index()
        write_csv(summary,a.root/'summaries'/('by_'+('_'.join(dims) or 'policy')+'.csv'))
        valid=result[result.validated].groupby(keys,dropna=False)[DELTA].agg(['mean','median']).reset_index()
        valid.columns=['_'.join(x).rstrip('_') for x in valid.columns]
        write_csv(valid,a.root/'summaries'/('validated_by_'+('_'.join(dims) or 'policy')+'.csv'))
        practical=result[result.practical_frontier].groupby(keys,dropna=False)[DELTA].agg(['mean','median']).reset_index()
        practical.columns=['_'.join(x).rstrip('_') for x in practical.columns]
        write_csv(practical,a.root/'summaries'/('practical_by_'+('_'.join(dims) or 'policy')+'.csv'))
    curve=result.sort_values(['background_id','policy','loss_budget']).copy()
    for prefix in ('discovery_',''):
        curve[prefix+'gain_increment']=curve.groupby(['background_id','policy'])[prefix+'delta_sr1'].diff()
        curve[prefix+'gain_per_budget_increment']=curve[prefix+'gain_increment']/curve.groupby(['background_id','policy']).loss_budget.diff()
    write_csv(curve,a.root/'budget_curve_knee_inputs.csv')
    write_csv(result[result.loss_budget==0],a.root/'strict_endpoint_comparison.csv')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['bank','inventory','search','select','evaluate','postprocess'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--shard',type=int,help='Stable one-based background index, 1..972')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.shard is not None and not 1<=a.shard<=972: p.error('shard must be 1..972')
    if a.stage=='bank': bank_stage(a); return
    bank=pd.read_csv(a.root/'bank.csv')
    pd.testing.assert_frame_equal(bank,expanded_bank(),check_dtype=False)
    sources=source_catalog(a.sources)
    if a.stage=='select': select_stage(a,bank,sources)
    elif a.stage=='postprocess': postprocess(a,bank,sources)
    else:
        stage='evaluation' if a.stage=='evaluate' else 'search'
        # CLI spelling differs from the on-disk convention.
        if a.stage=='evaluate': a.stage='evaluation'
        selection=frozen(a.root,bank) if stage=='evaluation' else None
        target=bank if a.shard is None else bank.iloc[a.shard-1:a.shard]
        plans(a,target,sources,stage,selection)
        if a.stage=='inventory' and (a.root/'freeze.json').exists():
            plans(a,bank,sources,'evaluation',frozen(a.root,bank))

if __name__=='__main__': main()
