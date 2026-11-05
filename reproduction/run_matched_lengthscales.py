"""Eagle half-correlation sweep; all times include mean and marginal variance."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

import numpy as np
from reproduction.matched_lengthscale import matched_grid

OLD = Path('reproduction/results/asymptotic_regimes')
CLOUD = '/home/horde/.cache/fast-stochastic-psr/poissonrecon/eagle.npy'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flags(arguments):
    result = []
    for key, value in arguments.items():
        if value is None or value is False or key == 'point_cloud_sha256':
            continue
        result.append('--'+key.replace('_', '-'))
        if value is not True:
            result.append(str(value))
    return result


def jobs():
    methods = ['h2-streamed', 'sparse-streamed', 'dense', 'original', 'sdd', 'nufft-cg', 'nufft-hybrid']
    for ell in [.12, .06, .03]:
        for method in methods:
            yield ell, 8192, method
    # A separate N sweep at a feasible original resolution. Competitors already
    # slower at 8k need not extend; no extrapolated completion times are used.
    for n in [2048, 32768]:
        for method in ['h2-streamed', 'sparse-streamed', 'dense', 'original']:
            yield .12, n, method


def run(root, only=None):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    cloud = np.load(CLOUD, mmap_mode='r')
    lo, hi = cloud[:, :3].min(0)-.1, cloud[:, :3].max(0)+.1
    manifest = dict(protocol='physical vector-kernel axis half-correlation matching',
                    paper='https://arxiv.org/pdf/2503.19136#page=15',
                    matching_formula_is_ours=True, memory_cap_bytes=12*2**30,
                    source_sha256={p: digest(p) for p in [
                        'reproduction/matched_lengthscale.py', 'reproduction/run_matched_lengthscales.py',
                        'reproduction/benchmark_eagle_comparison.py', 'reproduction/benchmark_hierarchical_variance.py',
                        'reproduction/variance_series.py', 'reproduction/scaling_reference.py']},
                    grids={str(ell): matched_grid(lo, hi, ell) for ell in [.12, .06, .03]}, rows=[])
    for ell, n, method in jobs():
        name = f'ell{ell:g}/{method}-{n}'
        out = root/(name+'.json'); out.parent.mkdir(exist_ok=True)
        row = dict(lengthscale=ell, n=n, method=method, path=str(out), reused=False)
        if ell == .03 and method != 'original':
            group = 'free-space' if method.endswith('streamed') else 'periodic'
            source = OLD/f'{group}/{method}-{n}.json'
            row.update(path=str(source), reused=True, sha256=digest(source),
                       reuse_reason='Identical physical lengthscale, data, workload and production settings; original-grid option does not execute on this route')
        elif method == 'original' and manifest['grids'][str(ell)]['one_dense_covariance_bytes'] > manifest['memory_cap_bytes']:
            row.update(status='not attempted: one dense grid covariance exceeds address-space cap',
                       lower_bound_bytes=manifest['grids'][str(ell)]['one_dense_covariance_bytes'])
        elif only is None or only in name:
            status_path = out.with_suffix('.status.json')
            if not status_path.exists():
                if method.endswith('streamed'):
                    a = json.loads((OLD/f'free-space/{method}-8192.json').read_text())['arguments']
                    a.update(n=n, lengthscale=ell, output=str(out))
                    module = 'reproduction.benchmark_hierarchical_variance'
                else:
                    a = dict(method=method, n=n, output=str(out), ell=ell/(2*np.pi))
                    if method == 'original': a['original_lengthscale'] = ell
                    if method == 'sdd': a['eta'] = .1
                    if method == 'nufft-cg': a['rhs_block_size'] = 4
                    if method == 'nufft-hybrid': a['moment_dtype'] = 'float64'
                    module = 'reproduction.benchmark_eagle_comparison'
                command = [sys.executable, '-m', module, *flags(a)]
                start = perf_counter()
                status = dict(command=command, source_sha256=manifest['source_sha256'])
                def cap(): resource.setrlimit(resource.RLIMIT_AS, (12*2**30, 12*2**30))
                with out.with_suffix('.log').open('w') as log:
                    try:
                        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                                timeout=600, preexec_fn=cap if method == 'original' else None)
                        status['returncode'] = result.returncode
                    except subprocess.TimeoutExpired:
                        status['timeout_seconds'] = 600
                status['wall_seconds'] = perf_counter()-start
                status_path.write_text(json.dumps(status, indent=2)+'\n')
                print(name, status.get('returncode', 'timeout'), round(status['wall_seconds'], 2), flush=True)
            status = json.loads(status_path.read_text())
            if status['source_sha256'] != manifest['source_sha256']:
                raise ValueError('Stale run sources: '+str(status_path))
            if method.endswith('streamed') and status.get('returncode') == 0:
                record = json.loads(out.read_text())
                if 'independent_audit' not in record:
                    reference = root/f'ell{ell:g}/reference-{n}.json'
                    with out.with_suffix('.audit.log').open('w') as log:
                        subprocess.run([sys.executable, '-m', 'reproduction.audit_hierarchical_variance',
                                        str(out), '--reference', str(reference), '--random-queries', '8'],
                                       stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
        if Path(row['path']).exists():
            row['sha256'] = digest(row['path'])
        manifest['rows'].append(row)
        (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='reproduction/results/matched_lengthscales')
    p.add_argument('--only', help='Optional job-name substring; omitted jobs remain planned')
    a = p.parse_args(); run(a.root, a.only)
