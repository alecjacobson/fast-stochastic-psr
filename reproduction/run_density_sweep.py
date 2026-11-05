"""Density-dependent Eagle workload with fixed physical amplitude and noise."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from reproduction.density_protocol import lengthscale, original_parameters
from reproduction.matched_lengthscale import matched_grid
from reproduction.run_matched_crossover import invoke
from reproduction.run_matched_lengthscales import flags, digest, CLOUD

BASE = Path('reproduction/results/matched_lengthscales/ell0.12')
METHODS = ['h2-streamed','sparse-streamed','original','dense','sdd','nufft-cg','nufft-hybrid']
SIZES = [2048,8192,16384,32768,65536,131072,262144]


def run(root, sizes):
    root = Path(root); root.mkdir(parents=True,exist_ok=True)
    manifest_path = root/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(
        law='ell(N)=0.12*sqrt(8192/N)', vector_prior_variance=.0025, observation_noise_variance=.000025,
        queries=1000000, batches=5, queries_per_batch=200000,
        periodic_cutoff='ceil(50*0.12/ell); bandwidth convergence tested separately',
        gates=dict(cdf=.001, variance_over_prior=.0001), rows=[])
    x=np.load(CLOUD,mmap_mode='r')[:,:3]; lo=x.min(0)-.1; hi=x.max(0)+.1
    sources={p:digest(p) for p in ['reproduction/benchmark_eagle_comparison.py',
        'reproduction/benchmark_hierarchical_variance.py','reproduction/density_protocol.py',
        'reproduction/matched_lengthscale.py','reproduction/gpu_audit.py']}
    for n in sizes:
        ell=float(lengthscale(n)); cutoff=int(np.ceil(50*.12/ell))
        grid=matched_grid(lo,hi,ell)
        for method in METHODS:
            out=root/f'{method}-{n}.json'; status_path=out.with_suffix('.status.json')
            row=dict(n=n,lengthscale=ell,cutoff=cutoff,method=method,path=str(out))
            if method=='original':
                row.update(grid=grid,parameter_control=original_parameters(grid['spacing'],.0025,.000025))
            if method=='nufft-hybrid' and n>8192:
                row['status']='not attempted: hybrid already slower than H2 at both measured sizes'
            elif method in ['sdd','nufft-cg'] and n>32768:
                row['status']='not attempted: stop after slower accepted overlap and the 32768 trial'
            elif method=='dense' and n>65536:
                row['status']='not attempted in primary sweep: dense capacity extension recorded separately'
            elif method=='original' and grid['one_dense_covariance_bytes']>12*2**30:
                row['status']='not attempted: one dense grid covariance exceeds 12 GiB address-space cap'
            else:
                if not status_path.exists():
                    a=json.loads((BASE/f'{method}-8192.json').read_text())['arguments'].copy()
                    a.update(n=n,queries=200000,batches=5,output=str(out),amplitude=.0025,relative_noise=.01)
                    if method.endswith('streamed'):
                        a.update(lengthscale=ell)
                        module='reproduction.benchmark_hierarchical_variance'
                    else:
                        a.update(ell=ell/(2*np.pi),cutoff=cutoff,fit_cutoff=cutoff)
                        if method=='original':a.update(original_lengthscale=ell,original_normalization='controlled')
                        module='reproduction.benchmark_eagle_comparison'
                    status=invoke([sys.executable,'-m',module,*flags(a)],out.with_suffix('.log'),method=='original')
                    status.update(source_sha256=sources)
                    status_path.write_text(json.dumps(status,indent=2)+'\n')
                    print(method,n,status.get('returncode','timeout'),round(status['wall_seconds'],2),flush=True)
                status=json.loads(status_path.read_text())
                if status['source_sha256']!=sources:raise ValueError('Stale source '+str(out))
                row['execution']=status
                if out.exists():
                    d=json.loads(out.read_text())
                    if method.endswith('streamed') and 'total_seconds' in d and 'independent_audit' not in d:
                        reference=root/f'reference-{n}.json'
                        result=invoke([sys.executable,'-m','reproduction.audit_hierarchical_variance',str(out),
                            '--reference',str(reference),'--random-queries','8','--reference-preconditioner','prefix-kd'],out.with_suffix('.audit.log'))
                        if result.get('returncode')!=0:raise RuntimeError('Independent audit failed '+str(out))
                    elif method=='dense' and 'total_seconds' in d and not d.get('quality',{}).get('passes'):
                        audit=out.with_name(out.stem+'.audit.json')
                        if not audit.exists():
                            invoke([sys.executable,'-m','reproduction.audit_periodic_scaling',str(out)],out.with_suffix('.audit.log'))
                    row['sha256']=digest(out)
            manifest['rows']=[r for r in manifest['rows'] if (r['n'],r['method'])!=(n,method)]+[row]
            manifest['source_sha256']=sources
            manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='reproduction/results/density_sweep')
    p.add_argument('--sizes',type=int,nargs='+',default=SIZES)
    a=p.parse_args();run(a.root,a.sizes)
