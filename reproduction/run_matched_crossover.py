"""Extend the fixed-detail Eagle sweep through the dense memory boundary."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

from reproduction.run_matched_lengthscales import digest, flags

BASE = Path('reproduction/results/matched_lengthscales')


def invoke(command, log, cap=False):
    start = perf_counter()
    def limit(): resource.setrlimit(resource.RLIMIT_AS, (12*2**30, 12*2**30))
    with log.open('w') as stream:
        try:
            r = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                               timeout=600, preexec_fn=limit if cap else None)
            return dict(command=command, returncode=r.returncode, wall_seconds=perf_counter()-start)
        except subprocess.TimeoutExpired:
            return dict(command=command, timeout_seconds=600, wall_seconds=perf_counter()-start)


def run(root, sizes, methods):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    source = {p: digest(p) for p in ['reproduction/benchmark_eagle_comparison.py',
              'reproduction/benchmark_hierarchical_variance.py', 'reproduction/matched_lengthscale.py',
              'reproduction/audit_periodic_scaling.py', 'reproduction/audit_hierarchical_variance.py']}
    for n in sizes:
        for method in methods:
            out = root/f'{method}-{n}.json'; status_path = out.with_suffix('.status.json')
            if not status_path.exists():
                template = json.loads((BASE/f'ell0.12/{method}-32768.json').read_text())
                a = template['arguments'].copy(); a.update(n=n, output=str(out))
                module = ('reproduction.benchmark_hierarchical_variance' if method.endswith('streamed')
                          else 'reproduction.benchmark_eagle_comparison')
                status = invoke([sys.executable, '-m', module, *flags(a)], out.with_suffix('.log'), method == 'original')
                status['source_sha256'] = source
                status_path.write_text(json.dumps(status, indent=2)+'\n')
                print(method, n, status.get('returncode', 'timeout'), status['wall_seconds'], flush=True)
            status = json.loads(status_path.read_text())
            if status['source_sha256'] != source: raise ValueError('Stale source '+str(out))
            if not out.exists(): continue
            record = json.loads(out.read_text())
            if 'total_seconds' not in record: continue
            if method == 'dense':
                audit_path = out.with_name(out.stem+'.audit.json')
                if not audit_path.exists():
                    result = invoke([sys.executable, '-m', 'reproduction.audit_periodic_scaling', str(out)],
                                    out.with_suffix('.audit.log'))
                    if result.get('returncode') != 0: raise RuntimeError('Dense audit failed '+str(out))
            elif method.endswith('streamed') and 'independent_audit' not in record:
                reference = root/f'reference-{n}.json'
                result = invoke([sys.executable, '-m', 'reproduction.audit_hierarchical_variance', str(out),
                                 '--reference', str(reference), '--random-queries', '8',
                                 '--reference-preconditioner', 'prefix-kd'], out.with_suffix('.audit.log'))
                if result.get('returncode') != 0: raise RuntimeError('Free-space audit failed '+str(out))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='reproduction/results/matched_crossover')
    p.add_argument('--sizes', type=int, nargs='+', default=[65536, 77824, 81920])
    p.add_argument('--methods', nargs='+', choices=['h2-streamed', 'sparse-streamed', 'dense', 'original'],
                   default=['h2-streamed', 'dense'])
    a = p.parse_args(); run(a.root, a.sizes, a.methods)
