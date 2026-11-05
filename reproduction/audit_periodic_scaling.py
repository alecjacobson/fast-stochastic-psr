"""Validate large dense production records with bounded-workspace full kernels.

A production fit can succeed close to device capacity while the convenient
NUFFT/leaf-cached offline reference fails. This independent audit constructs
cross kernels before the factor, then uses tiled full analytic residuals.
No reference work is included in the production timing.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from reproduction.benchmark_eagle_comparison import cholesky_in_place,assess
from reproduction.benchmark_hierarchical_variance import read_field
from reproduction.poissonrecon_data import load_cloud
from reproduction.scaling_reference import refined_moments


def audit(path):
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular
    from geospsr.spectral import FourierPoisson
    from geospsr.spectral_tail import analytic_covariance_rows64
    from reproduction.dense_covariance import analytic_covariance_rows32
    path=Path(path);d=json.loads(path.read_text());a=d['arguments'];t=perf_counter()
    if 'field' not in d or 'total_seconds' not in d:raise ValueError('No completed production fit/query work')
    x,y,_=load_cloud(a['point_cloud'],a['n']);n=len(x);axis=a['amplitude']**(1/3);noise=a['amplitude']*a['relative_noise']
    q=np.vstack(([1,0,0]+a['halfwidth']*np.random.default_rng(731).uniform(-1,1,(8,3)),
                 [1,0,0]+a['halfwidth']*np.array(list(itertools.product([-1.,1.],repeat=3))),[[1,0,0]]))
    op=FourierPoisson(a['cutoff'],a['ell'],axis,dtype='float64',eps=1e-11)
    c=[]
    for point in q:
        c.append(cp.asnumpy(op.cross_values(cp.asarray(point)[None]-cp.asarray(x))))
    c=np.array(c);del op;cp.get_default_memory_pool().free_all_blocks()
    factor=analytic_covariance_rows32(x,x,a['ell'],axis);factor.flat[::n+1]+=noise
    factor=cholesky_in_place(factor);cp.get_default_memory_pool().free_all_blocks()
    def solve(b):
        z=solve_triangular(factor,cp.asarray(b,dtype='f'),lower=True)
        return solve_triangular(factor,z,lower=True,trans='T').astype('d')
    def action(z):
        out=cp.empty_like(z)
        for first in range(0,n,512):
            stop=min(first+512,n)
            matrix=analytic_covariance_rows64(x[first:stop],x,a['ell'],axis)
            out[first:stop]=matrix@z+noise*z[first:stop]
            del matrix
        return out
    field=read_field(d['field']);prior=field.prior_variance
    m,v,info=refined_moments(solve,action,c,y,prior,noise)
    qm,qv=field.predict(q);quality=assess(qm,qv,m,v,prior)
    bound_ok=max(info['variance_error_bound'])/prior<1e-8 and max(info['mean_error_bound'])/np.sqrt(min(v))<1e-6
    quality['passes'] &= bool(bound_ok);quality['reference_bounds_pass']=bool(bound_ok)
    result=dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                quality=quality,reference=info,query=q.tolist(),mean=m.tolist(),variance=v.tolist(),
                seconds_excluded=perf_counter()-t,scope='17 uncompressed probes, full float64 analytic tiled residuals')
    out=path.with_name(path.stem+'.audit.json');out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['quality']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('record');audit(p.parse_args().record)
