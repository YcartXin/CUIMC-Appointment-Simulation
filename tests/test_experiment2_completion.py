import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
from analysis import experiment2_frontier as f
from analysis import experiment2_pareto_completion as p
from scripts.prepare_experiment2_completion_sources import prepare


class CompletionTests(unittest.TestCase):
    def test_pareto_all_candidates_independent_validation_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);front=root/'frontier';pareto=root/'pareto'
            bank=f.expanded_bank(True).iloc[:1];row=bank.iloc[0]
            f.write_csv(bank,front/'bank.csv')
            sources=root/'sources.json';sources.write_text('[]')
            grid={'baseline':[(100,0,-1)],'reservation_only':[(100,5,3)],'both_flexible':[(2,5,2),(4,5,3)]}
            search=[]
            for policy,cells in grid.items():
                for h,q,w in cells:
                    for seed in f.SEARCH:
                        gain=0 if q==0 else (.01 if h==2 else .02)
                        loss=-.02 if h==100 and q else 0
                        search.append(dict(zip(f.KEY+f.METRICS,[h,q,w,seed,.5+gain,.5+loss,.8])))
            f.write_csv(pd.DataFrame(search),front/'search/raw'/f'{row.background_id}.csv')
            a=SimpleNamespace(root=pareto,frontier_root=front,sources=sources,stage='catalog',dry_run=True,shard=None)
            with patch.object(f,'cells',return_value=grid):
                p.catalog(a,bank,[])
                chosen=p.freeze_read(pareto)
                self.assertEqual(len(chosen),2) # preserve both meaningful Pareto cells, not just max gain
                with self.assertRaises(ValueError):p.catalog(a,bank,[])
            # Independent evaluation overturns one discovery-qualified cell.
            evaluation=[]
            for h,q,w in [(100,0,-1),(2,5,2),(4,5,3)]:
                for seed in f.EVAL:
                    d1=0 if q==0 else .015
                    d2=-.001 if h==4 else 0
                    evaluation.append(dict(zip(f.KEY+f.METRICS,[h,q,w,seed,.5+d1,.5+d2,.81])))
            f.write_csv(pd.DataFrame(evaluation),front/'evaluation/raw'/f'{row.background_id}.csv')
            reused=[({f.signature(row):[str(row.background_id)]},{'search':[front/'search/raw'],'evaluation':[front/'evaluation/raw']})]
            a.stage='inventory'
            report=f.plans(a,bank,reused,'evaluation',chosen)
            self.assertEqual(report.pending.sum(),0)
            p.postprocess(a,bank,reused,chosen)
            out=pd.read_csv(pareto/'validated_meaningful_pareto_cells.csv')
            self.assertEqual(out.validated_meaningful_pareto.sum(),1)
            self.assertEqual(len(out),2)
            universe=pd.read_csv(pareto/'validated_background_policy_universe.csv')
            self.assertEqual(len(universe),2)
            self.assertEqual(universe.validated_pareto_exists.sum(),1)
            # A missing independent seed must prevent publication.
            (front/'evaluation/raw'/f'{row.background_id}.csv').write_text(pd.DataFrame(evaluation).iloc[:-1].to_csv(index=False))
            with self.assertRaises(ValueError):p.postprocess(a,bank,reused,chosen)
            # Frozen metadata detects tampering.
            with (pareto/'selected_pareto_discovery.csv').open('a') as out:out.write('\n')
            with self.assertRaises(ValueError):p.freeze_read(pareto)

    def test_incomplete_discovery_aborts_without_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bank=f.expanded_bank(True).iloc[:1]
            a=SimpleNamespace(root=root/'p',frontier_root=root/'f',sources=root/'sources.json')
            a.sources.write_text('[]')
            with patch.object(f,'cells',return_value={'baseline':[(100,0,-1)],'reservation_only':[(100,5,3)],'both_flexible':[(2,5,2)]}):
                with self.assertRaises(ValueError):p.catalog(a,bank,[])
            self.assertFalse((a.root/'pareto_freeze.json').exists())

    def test_manifest_reuses_previous_raw_and_refuses_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';new=root/'new'
            f.write_csv(f.expanded_bank().iloc[:1],old/'bank.csv')
            for stage in ['search','evaluation']:(old/stage/'raw').mkdir(parents=True)
            (old/'freeze.json').write_text(json.dumps({'source_manifest':[]}))
            path=prepare(old,new)
            self.assertEqual(prepare(old,new),path)
            self.assertEqual(len(json.loads(path.read_text())),1)
            path.write_text('[]')
            with self.assertRaises(ValueError):prepare(old,new)
            with self.assertRaises(ValueError):prepare(old,old)


if __name__=='__main__':unittest.main()
