"""Resolve a validation precision check using H2 tolerances 1e-9 and 1e-11.

The initial 1e-7 to 1e-9 check preserved the winner but changed one RMS by
1.44 percent. Retain that failure; compare the refined pair before production.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import numpy as np

from reproduction.noise_selection import VALIDATION_SEEDS
from reproduction.run_matched_crossover import invoke
from reproduction.run_matched_lengthscales import digest


def worker():
    import geospsr.hierarchical as hierarchy
    from geospsr.hierarchical_gpu import GPUHierarchicalSystem
    from reproduction.noise_selection_control import solve_with_margin
    original = hierarchy.HierarchicalMaternSystem

    def precise(*args, **kwargs):
        kwargs['tolerance'] = 1e-11
        return original(*args, **kwargs)

    hierarchy.HierarchicalMaternSystem = precise
    GPUHierarchicalSystem.solve = solve_with_margin
    sys.argv.remove('--worker')
    out = Path(sys.argv[sys.argv.index('--output')+1])
    runpy.run_module('reproduction.noise_selection', run_name='__main__')
    d = json.loads(out.read_text())
    d['backend'] = 'H2 tolerance 1e-11, solve tolerance 1e-10; quarter-tolerance recurrence stopping'
    for source in [__file__, 'reproduction/noise_selection_control.py']:
        d['source_sha256'][source] = digest(source)
    out.write_text(json.dumps(d, indent=2)+'\n')


def check(root):
    root = Path(root)
    first = root/'precision-primary.json'
    if not first.exists() and (root/'precision.json').exists():
        first.write_bytes((root/'precision.json').read_bytes())
    selection = json.loads((root/'selection.json').read_text())
    largest = max(r['n'] for r in selection)
    rows = []
    for cell in selection:
        if cell['n'] != largest:
            continue
        checked = []
        for candidate in sorted(cell['curves'], key=lambda r:r['score'])[:2]:
            levels = []
            for folder, module in [('precision','check_noise_selection'),('precision_refined','refine_noise_selection')]:
                scores, records = [], []
                for seed in VALIDATION_SEEDS:
                    ell, sigma = candidate['lengthscale'], cell['normal_noise']
                    path = root/f'{folder}/n{largest}-s{sigma:g}-ell{ell:g}-seed{seed}.json'
                    path.parent.mkdir(exist_ok=True)
                    if not path.exists():
                        status = invoke([sys.executable, '-m', 'reproduction.'+module, '--worker', '--n', str(largest),
                            '--sigma', str(sigma), '--ell', str(ell), '--seed', str(seed), '--output', str(path)],
                            path.with_suffix('.log'))
                        path.with_suffix('.status.json').write_text(json.dumps(status, indent=2)+'\n')
                        if status.get('returncode') != 0:
                            raise RuntimeError('Refined selection fit failed '+str(path))
                    d = json.loads(path.read_text())
                    for source, checksum in d['source_sha256'].items():
                        if digest(source) != checksum:
                            raise ValueError('Stale precision source '+source)
                    scores.extend(m['penalized_surface_rms']**2 for m in d['noisy_metrics'])
                    records.append(dict(path=str(path), sha256=digest(path)))
                levels.append(dict(score=float(np.sqrt(np.mean(scores))), records=records))
            checked.append(dict(lengthscale=ell, original_score=candidate['score'], primary_score=levels[0]['score'],
                tighter_score=levels[1]['score'], relative_score_change=abs(levels[1]['score']/levels[0]['score']-1),
                records=levels[0]['records']+levels[1]['records']))
        winner = min(checked, key=lambda r:r['tighter_score'])['lengthscale']
        row = dict(n=largest, normal_noise=cell['normal_noise'], selected=cell['lengthscale'], tighter_winner=winner,
                   compared_hierarchy_tolerances=[1e-9,1e-11], candidates=checked,
                   passes=winner==cell['lengthscale'] and max(r['relative_score_change'] for r in checked)<.01)
        rows.append(row)
        (root/'precision.json').write_text(json.dumps(rows, indent=2)+'\n')
        print('precision refined', row, flush=True)
        if not row['passes']:
            raise RuntimeError('Selection still unstable under hierarchy refinement')


if __name__=='__main__':
    if '--worker' in sys.argv:
        worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='reproduction/results/noise_selected')
        check(p.parse_args().root)
