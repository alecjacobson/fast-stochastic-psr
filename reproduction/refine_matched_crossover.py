"""Retain failed settings; select variance/hierarchy budgets using offline audits.

This is benchmark tuning, not a production output-error stopping certificate.
"""
import argparse
import json
from pathlib import Path
import sys

from reproduction.run_matched_crossover import invoke
from reproduction.run_matched_lengthscales import flags, digest


def main(root, n):
    root = Path(root)
    baseline = json.loads((root/f'h2-streamed-{n}.json').read_text())
    if baseline['independent_audit']['passes']:
        print('Fixed budget already passes; no tuning performed.'); return
    target = root/'refined'; target.mkdir(exist_ok=True)
    for iterations, tolerance in [(16, 1e-5), (32, 1e-5), (16, 1e-6), (32, 1e-6)]:
        suffix = '' if tolerance == 1e-5 else f'-tol{tolerance:g}'
        out = target/f'h2-streamed-{n}-iterations{iterations}{suffix}.json'
        if not out.exists():
            a = baseline['arguments'].copy()
            a.update(variance_iterations=iterations, h2_tolerance=tolerance, output=str(out))
            result = invoke([sys.executable, '-m', 'reproduction.benchmark_hierarchical_variance', *flags(a)],
                            out.with_suffix('.log'))
            result['benchmark_sha256'] = digest('reproduction/benchmark_hierarchical_variance.py')
            out.with_suffix('.status.json').write_text(json.dumps(result, indent=2)+'\n')
            if result.get('returncode') != 0: raise RuntimeError('Refined benchmark failed')
        status = json.loads(out.with_suffix('.status.json').read_text())
        if status['benchmark_sha256'] != digest('reproduction/benchmark_hierarchical_variance.py'):
            raise ValueError('Stale benchmark source '+str(out))
        record = json.loads(out.read_text())
        for path, checksum in record['source_sha256'].items():
            if digest(path) != checksum: raise ValueError('Stale dependency '+path)
        if 'independent_audit' not in json.loads(out.read_text()):
            result = invoke([sys.executable, '-m', 'reproduction.audit_hierarchical_variance', str(out),
                             '--reference', str(root/f'reference-{n}.json'), '--random-queries', '8',
                             '--reference-preconditioner', 'prefix-kd'], out.with_suffix('.audit.log'))
            if result.get('returncode') != 0: raise RuntimeError('Refined audit failed')
        d = json.loads(out.read_text())
        print(iterations, tolerance, d['total_seconds'], d['independent_audit'], flush=True)
        if d['independent_audit']['passes']: return
    raise RuntimeError('No passing tested accuracy settings')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='reproduction/results/matched_crossover')
    p.add_argument('--n', type=int, default=196608)
    a = p.parse_args(); main(a.root, a.n)
