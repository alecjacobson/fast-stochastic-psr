"""Check large-N lengthscale choices with a tighter hierarchy, before timings."""
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
        kwargs['tolerance'] = 1e-9
        return original(*args, **kwargs)

    hierarchy.HierarchicalMaternSystem = precise
    GPUHierarchicalSystem.solve = solve_with_margin
    sys.argv.remove('--worker')
    out = Path(sys.argv[sys.argv.index('--output')+1])
    runpy.run_module('reproduction.noise_selection', run_name='__main__')
    record = json.loads(out.read_text())
    record.update(backend='H2 tolerance 1e-9, solve tolerance 1e-10; quarter-tolerance recurrence stopping')
    for source in [__file__, 'reproduction/noise_selection_control.py']:
        record['source_sha256'][source] = digest(source)
    out.write_text(json.dumps(record, indent=2)+'\n')


def check(root):
    root = Path(root)
    selection = json.loads((root/'selection.json').read_text())
    largest = max(r['n'] for r in selection)
    rows = []
    for cell in selection:
        if cell['n'] != largest:
            continue
        candidates = sorted(cell['curves'], key=lambda r:r['score'])[:2]
        checked = []
        for candidate in candidates:
            scores = []
            records = []
            for seed in VALIDATION_SEEDS:
                ell, sigma = candidate['lengthscale'], cell['normal_noise']
                path = root/f'precision/n{largest}-s{sigma:g}-ell{ell:g}-seed{seed}.json'
                path.parent.mkdir(exist_ok=True)
                if not path.exists():
                    status = invoke([sys.executable, '-m', 'reproduction.check_noise_selection', '--worker',
                        '--n', str(largest), '--sigma', str(sigma), '--ell', str(ell), '--seed', str(seed),
                        '--output', str(path)], path.with_suffix('.log'))
                    path.with_suffix('.status.json').write_text(json.dumps(status, indent=2)+'\n')
                    if status.get('returncode') != 0:
                        raise RuntimeError('Tight hierarchy check failed '+str(path))
                d = json.loads(path.read_text())
                for source, checksum in d['source_sha256'].items():
                    if digest(source) != checksum:
                        raise ValueError('Stale precision source '+source)
                scores.extend(m['penalized_surface_rms']**2 for m in d['noisy_metrics'])
                records.append(dict(path=str(path), sha256=digest(path)))
            score = float(np.sqrt(np.mean(scores)))
            checked.append(dict(lengthscale=ell, primary_score=candidate['score'], tighter_score=score,
                                relative_score_change=abs(score/candidate['score']-1), records=records))
        winner = min(checked, key=lambda r:r['tighter_score'])['lengthscale']
        row = dict(n=largest, normal_noise=cell['normal_noise'], selected=cell['lengthscale'],
                   tighter_winner=winner, candidates=checked,
                   passes=winner == cell['lengthscale'] and max(r['relative_score_change'] for r in checked) < .01)
        rows.append(row)
        (root/'precision.json').write_text(json.dumps(rows, indent=2)+'\n')
        print('precision', row, flush=True)
        if not row['passes']:
            raise RuntimeError('Selection changes under hierarchy refinement; refine before timing')


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker()
    else:
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument('--root', default='reproduction/results/noise_selected')
        check(p.parse_args().root)
