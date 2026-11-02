"""Prepare a new completion reuse manifest from the previous frozen frontier."""
import argparse
import json
from pathlib import Path
from analysis.experiment2_frontier import source_catalog


def prepare(previous_root, new_root):
    if previous_root.resolve()==new_root.resolve():raise ValueError('New output root required')
    meta=json.loads((previous_root/'freeze.json').read_text())
    specs=meta['source_manifest']
    if not isinstance(specs,list):raise ValueError('Invalid historical source manifest')
    spec=dict(bank=str((previous_root/'bank.csv').resolve()),
        search=[str((previous_root/'search/raw').resolve())],
        evaluation=[str((previous_root/'evaluation/raw').resolve())],simulation_convention_verified=True)
    specs=specs+[spec]
    # Same inspected pipeline conventions as the original frontier. Fail on
    # missing historical sources rather than silently scheduling duplicate work.
    new_root.mkdir(parents=True,exist_ok=True)
    path=new_root/'reuse_sources.json'
    if path.exists():
        if json.loads(path.read_text())!=specs:raise ValueError('Existing source manifest differs; preserve it and inspect')
    else:
        # Validate bank schemas, directories and verified assertions first.
        for s in specs:
            if not Path(s['bank']).is_file():raise FileNotFoundError(s['bank'])
            for stage in ['search','evaluation']:
                for raw in s.get(stage,[]):
                    if not Path(raw).is_dir():raise FileNotFoundError(raw)
        path.write_text(json.dumps(specs,indent=2))
    source_catalog(path)
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-root',type=Path,required=True)
    p.add_argument('--new-root',type=Path,required=True)
    a=p.parse_args();print(prepare(a.previous_root,a.new_root))
