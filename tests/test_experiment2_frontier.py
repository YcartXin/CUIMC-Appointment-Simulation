import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
from analysis import experiment2_frontier as f

class FrontierTests(unittest.TestCase):
    def test_design_and_grid(self):
        b=f.expanded_bank()
        self.assertEqual(len(b),972)
        self.assertEqual(set(b.class1_share),{.1,.3,.5,.7,.8,.9})
        self.assertEqual(set(b.slots_per_day),{30,40,50})
        self.assertEqual(len(set(b.background_id)&set(f.generate_bank().background_id)),540)
        self.assertEqual(len(set(b.background_id)&set(f.followup_bank().background_id)),54)
        for cap in (30,40,50):
            r=b[b.slots_per_day==cap].iloc[0]
            g=f.cells(r)
            self.assertEqual({h for h,q,w in g['reservation_only']},{100})
            self.assertEqual({h for h,q,w in g['both_flexible']},set(range(2,31)))
            self.assertTrue(all(q>0 and w>=2 for h,q,w in g['both_flexible']))

    def test_ties_and_budget(self):
        g=pd.DataFrame([
            [2,10,3,.1,-.02,.8,.25], [4,5,3,.1,-.02,.8,.125],
            [8,5,5,.1,-.02,.8,.125], [8,5,3,.1,-.02,.8,.125],
            [9,5,3,.09,0,.9,.125]],
            columns=f.CELL+['delta_sr1','delta_sr2','average_utilization','q_fraction'])
        self.assertEqual(f.choose(g,0).iloc[0].delta_sr1,.09)
        z=f.choose(g,.02).iloc[0]
        self.assertEqual(tuple(z[f.CELL]),(8,5,3))
        g.loc[0,'delta_sr2']=-.019
        self.assertEqual(f.choose(g,.02).iloc[0].Q,10)
        g.loc[1,'delta_sr2']=-.019
        g.loc[1,'average_utilization']=.81
        self.assertEqual(f.choose(g,.02).iloc[0].horizon_days,4)

    def test_reuse_freeze_independent_and_postprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            b=f.expanded_bank().iloc[:1]
            r=b.iloc[0]
            f.write_csv(b,root/'bank.csv')
            grid={'baseline':[(100,0,-1)],'reservation_only':[(100,5,3)],'both_flexible':[(2,5,2),(4,5,3)]}
            data=[]
            for policy,cs in grid.items():
                for h,q,w in cs:
                    for s in f.SEARCH:
                        gain=0 if not q else (.04 if h==4 else .01)
                        loss=0 if h!=4 else -.02
                        data.append(dict(zip(f.KEY+f.METRICS,[h,q,w,s,.5+gain,.5+loss,.8])))
            f.write_csv(pd.DataFrame(data),root/'search/raw'/f'{r.background_id}.csv')
            (root/'sources.json').write_text('[]')
            args=SimpleNamespace(root=root,stage='inventory',shard=None,dry_run=True,sources=root/'sources.json')
            with patch.object(f,'cells',return_value=grid):
                report=f.plans(args,b,[],'search')
                self.assertEqual(report.pending.sum(),0)
                f.select_stage(args,b,[])
                selected=f.frozen(root,b)
                self.assertEqual(len(selected),18)
                bf=selected[selected.policy=='both_flexible']
                self.assertEqual(bf[bf.loss_budget==0].iloc[0].horizon_days,2)
                self.assertEqual(bf[bf.loss_budget==.02].iloc[0].horizon_days,4)
                with self.assertRaises(ValueError): f.select_stage(args,b,[])
            ev=[]
            for h,q,w in [(100,0,-1),(100,5,3),(2,5,2),(4,5,3)]:
                for s in f.EVAL:
                    ev.append(dict(zip(f.KEY+f.METRICS,[h,q,w,s,.5+(.01 if q else 0),.5-(.03 if h==4 else 0),.8])))
            f.write_csv(pd.DataFrame(ev),root/'evaluation/raw'/f'{r.background_id}.csv')
            before=(root/'selected_discovery.csv').read_bytes()
            f.postprocess(args,b,[])
            result=pd.read_csv(root/'evaluated_selected_cells.csv')
            self.assertEqual(len(result),18)
            z=result[(result.policy=='both_flexible')&(result.loss_budget==.02)].iloc[0]
            self.assertEqual(z.H,4)  # failed validation never replaced by H=2
            self.assertFalse(z.validated)
            self.assertEqual(before,(root/'selected_discovery.csv').read_bytes())
            # Partial discovery fails closed.
            with self.assertRaises(ValueError): f.paired(pd.DataFrame(data[:-1]),[(4,5,3)],f.SEARCH)
            # Frozen tampering is rejected.
            (root/'selected_discovery.csv').write_text('changed')
            with self.assertRaises(ValueError): f.frozen(root,b)

    def test_cross_bank_alias_and_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'old').mkdir()
            row=f.expanded_bank().iloc[0].copy()
            alias=row.copy(); alias['background_id']='DIFFERENT_CONTEXT_ID'
            pd.DataFrame([alias]).to_csv(root/'old_bank.csv',index=False)
            raw=pd.DataFrame([[100,0,-1,1000,.5,.6,.8]],columns=f.KEY+f.METRICS)
            raw.to_csv(root/'old/DIFFERENT_CONTEXT_ID.csv',index=False)
            manifest=[dict(bank=str(root/'old_bank.csv'),search=[str(root/'old')],simulation_convention_verified=True)]
            (root/'sources.json').write_text(json.dumps(manifest))
            src=f.source_catalog(root/'sources.json')
            got=f.raw_for(row,'search',src,root)
            self.assertEqual(len(got),1)
            raw['average_utilization']=.9
            f.write_csv(raw,root/'search/raw'/f'{row.background_id}.csv')
            with self.assertRaises(ValueError): f.raw_for(row,'search',src,root)

    def test_only_missing_frozen_tasks_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            b=f.expanded_bank().iloc[:1]; row=b.iloc[0]
            selected=pd.DataFrame([dict(background_id=row.background_id,discovery_feasible=True,
                                        horizon_days=2,Q=5,window=2)]*2)
            args=SimpleNamespace(root=root,stage='evaluation',shard=None,dry_run=False)
            captured=[]
            def fake_run(tasks,path,workers,flush_every):
                captured.extend(tasks)
                vals=[]
                for t in tasks:
                    v={k:t['extra_cols'][k] for k in f.CELL}
                    v.update(seed=t['seed'],**dict(zip(f.METRICS,[.5,.5,.8])))
                    vals.append(v)
                f.write_csv(pd.DataFrame(vals),path)
            with patch.object(f,'run_tasks',side_effect=fake_run):
                report=f.plans(args,b,[],'evaluation',selected)
                self.assertEqual(report.pending.sum(),40) # one winner plus baseline, not two copies
                self.assertEqual({t['seed'] for t in captured},set(f.EVAL))
                self.assertEqual({tuple(t['extra_cols'][k] for k in f.CELL) for t in captured},
                                 {(2,5,2),(100,0,-1)})
                captured.clear()
                report=f.plans(args,b,[],'evaluation',selected)
                self.assertEqual(report.pending.sum(),0)
                self.assertEqual(captured,[])
            self.assertTrue(f.choose(pd.DataFrame([dict(delta_sr2=-.2)]),.1).empty)

if __name__=='__main__': unittest.main()
