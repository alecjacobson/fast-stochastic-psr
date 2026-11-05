"""Measured query/reuse sweeps and bandwidth/eigenspace output comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.special import ndtr

from reproduction.benchmark_hierarchical_variance import read_field,query_batches


def main(root):
    import cupy as cp
    from cupyx.scipy.special import ndtr as cdf
    root=Path(root);records=[]
    candidates=[root/'periodic/dense-32768.json',root/'free-space/h2-streamed-32768.json',
                root/'free-space/sparse-streamed-32768.json']
    for path in candidates:
        d=json.loads(path.read_text());f=read_field(d['field']);f.prepare_gpu()
        a=d['arguments'];f.predict_gpu([[1,0,0]]);cp.cuda.get_current_stream().synchronize()
        for count,batches in [(1024,1),(4096,1),(16384,1),(65536,1),(262144,1),(1048576,1),(65536,4),(65536,16),(65536,64)]:
            runs=[]
            for rep in range(3):
                elapsed=0.;checks=[]
                for q in query_batches(SimpleNamespace(**dict(a,queries=count,batches=batches))):
                    cp.cuda.get_current_stream().synchronize();start=perf_counter()
                    m,v=f.predict_gpu(q);p=cdf((d['level']-m)/cp.sqrt(v))
                    winner=int(cp.argmax((4*p*(1-p)).reshape(-1,64).mean(1)));checksum=float(p.sum())
                    cp.cuda.get_current_stream().synchronize();elapsed+=perf_counter()-start
                    checks.append([winner,checksum])
                runs.append(elapsed)
            records.append(dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                method=a['method'],n=a['n'],queries=count,batches=batches,m=count*batches,
                seconds=runs,median_query_seconds=float(np.median(runs)),
                reused_preparation_seconds=d['preparation_seconds'],
                composed_total_seconds=d['preparation_seconds']+float(np.median(runs)),checksums=checks))
    (root/'query_sweep.json').write_text(json.dumps(records,indent=2)+'\n')
    # Compare direct, uncompressed posterior probes as bandwidth changes.
    paths=[root/'periodic/dense-2048.json',*sorted((root/'bandwidth').glob('dense-*.json'))]
    paths=[p for p in paths if not p.name.endswith('.status.json')]
    values=sorted([(json.loads(p.read_text()),p) for p in paths],key=lambda t:t[0]['arguments']['cutoff'])
    values=[(d,p) for d,p in values if d.get('complete') and d.get('quality',{}).get('passes')]
    checks=[]
    for (d,p),(ref,rp) in zip(values[:-1],values[1:]):
        m=np.array(d['probes']['reference_mean']);v=np.array(d['probes']['reference_variance'])
        rm=np.array(ref['probes']['reference_mean']);rv=np.array(ref['probes']['reference_variance'])
        pe=float(np.max(abs(ndtr((m[-1]-m)/np.sqrt(v))-ndtr((rm[-1]-rm)/np.sqrt(rv)))))
        ve=float(np.max(abs(v-rv))/ref['probes']['prior'])
        checks.append(dict(source=str(p),reference=str(rp),cutoff=d['arguments']['cutoff'],
            reference_cutoff=ref['arguments']['cutoff'],probability_error=pe,variance_error_over_prior=ve,
            passes=bool(pe<=1e-3 and ve<=1e-4),scope='Successive finite-band models, not an infinite-band certificate'))
    (root/'bandwidth_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    import gpytoolbox as gpy
    base=json.loads((root/'original/full-grid16-8192.json').read_text());ref=np.load(base['original_grid_file'])
    gs=base['grid']['shape'];h=base['grid']['spacing'];corner=base['grid']['corner']
    contact=gpy.fd_interpolate(np.array([[1.,0,0]]),gs=gs,h=h,corner=corner)
    rl=float((contact@ref['mean'])[0]);checks=[]
    for path in sorted((root/'original').glob('subspace*-8192.json')):
        if path.name.endswith('.status.json'):continue
        d=json.loads(path.read_text())
        if not d.get('complete'):continue
        field=np.load(d['original_grid_file']);level=float((contact@field['mean'])[0]);pe=ve=0.
        for q in query_batches(SimpleNamespace(**d['arguments'])):
            w=gpy.fd_interpolate(q,gs=gs,h=h,corner=corner)
            m,v=w@field['mean'],w@field['variance'];rm,rv=w@ref['mean'],w@ref['variance']
            pe=max(pe,float(np.max(abs(ndtr((level-m)/np.sqrt(v))-ndtr((rl-rm)/np.sqrt(rv))))))
            ve=max(ve,float(np.max(abs(v-rv))/np.max(ref['variance'])))
        checks.append(dict(source=str(path),rank=d['arguments']['subspace'],probability_error=pe,
            variance_error_over_max_grid_variance=ve,passes=bool(pe<=1e-3 and ve<=1e-4),
            scope='Same-grid full covariance reference, one million queries; no cross-grid or continuum claim'))
    (root/'subspace_checks.json').write_text(json.dumps(checks,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='reproduction/results/asymptotic_regimes')
    main(p.parse_args().root)
