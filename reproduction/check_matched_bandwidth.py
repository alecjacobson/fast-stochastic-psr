"""Do smoother matched-width periodic models converge at cutoff 50?"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.special import ndtr


def main(root):
    root = Path(root); results = []
    for ell in [.12, .06]:
        previous = root/f'ell{ell:g}/dense-8192.json'
        for cutoff in [75, 100]:
            out = root/f'bandwidth/ell{ell:g}-cutoff{cutoff}.json'; out.parent.mkdir(exist_ok=True)
            if not out.exists():
                command = [sys.executable, '-m', 'reproduction.benchmark_eagle_comparison',
                           '--method', 'dense', '--n', '8192', '--ell', str(ell/(2*np.pi)),
                           '--cutoff', str(cutoff), '--output', str(out)]
                with out.with_suffix('.log').open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
            old, new = [json.loads(p.read_text()) for p in [previous, out]]
            for key in ['point_sha256', 'normal_sha256']:
                if old[key] != new[key]: raise ValueError('Mismatched '+key)
            if old['probes']['query'] != new['probes']['query']:
                raise ValueError('Different probes')
            values = []
            for d in [old, new]:
                m, v = [np.array(d['probes'][key]) for key in ['reference_mean', 'reference_variance']]
                if not np.isfinite(m).all() or not np.isfinite(v).all() or np.any(v <= 0):
                    raise ValueError('Invalid reference')
                values.append((ndtr((m[-1]-m)/np.sqrt(v)), v))
            pe = float(np.max(abs(values[0][0]-values[1][0])))
            ve = float(np.max(abs(values[0][1]-values[1][1]))/new['probes']['prior'])
            row = dict(lengthscale=ell, cutoff=old['arguments']['cutoff'], reference_cutoff=cutoff,
                       source=str(previous), reference=str(out),
                       source_sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),
                       reference_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                       probability_max_difference=pe, variance_max_difference_over_prior=ve,
                       passes=bool(pe <= 1e-3 and ve <= 1e-4),
                       scope='Direct uncompressed moments, 73 probes; successive finite bands, not infinite-band certification')
            results.append(row); print(row, flush=True); previous = out
            (root/'bandwidth_checks.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='reproduction/results/matched_lengthscales')
    main(p.parse_args().root)
