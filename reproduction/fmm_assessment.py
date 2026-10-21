"""Far-field rank diagnostics for a future kernel-independent FMM.

This is a compression experiment, NOT an FMM implementation or speed benchmark.
"""
import os
os.environ['JAX_ENABLE_X64']='true'
import json
from pathlib import Path
import numpy as np
import cupy as cp
import jax
import jax.numpy as jnp
from geospsr.spectral import FourierPoisson
from geospsr.kernels import Matern32Kernel,ProductKernel
out=Path(__file__).resolve().parent/'results';rng=np.random.default_rng(1234)
result={'scope':'256-by-256 far-field blocks, float64, NUFFT eps=1e-11, 101^3 finite cross spectrum; no FMM speedup claim','rows':[]}
for ell in [.01,.04]:
    op=FourierPoisson(lengthscale=ell,dtype='float64',eps=1e-11)
    k=Matern32Kernel(ell,.1);kv=ProductKernel(k,k,k)
    evalkv=jax.jit(jax.vmap(jax.vmap(kv,(None,0)),(0,None)))
    for name,source_center,target_center in [('separated_x',[0,0,0],[.8,0,0]),('diagonal',[0,0,0],[.8,.8,.8]),('near',[0,0,0],[.25,0,0]),('periodic_seam',[-3.02,0,0],[3.02,0,0])]:
        x=rng.uniform(-.1,.1,(256,3))+source_center;q=rng.uniform(-.1,.1,(256,3))+target_center
        kvv=np.asarray(evalkv(jnp.asarray(q),jnp.asarray(x)))
        diffs=(q[:,None]-x[None]).reshape(-1,3)
        kfv=cp.asnumpy(op.cross_values(diffs)).reshape(256,256*3)
        row={'lengthscale':ell,'geometry':name}
        for label,matrix in [('kvv',kvv),('kfv',kfv)]:
            sv=np.linalg.svd(matrix,compute_uv=False)
            # Frobenius relative approximation error after rank r.
            tail=np.sqrt(np.r_[np.cumsum(sv[::-1]**2)[::-1],0])/np.linalg.norm(sv)
            row[label]={'max_abs':float(np.max(np.abs(matrix))),
                        'ranks':{str(tol):int(np.flatnonzero(tail<=tol)[0]) for tol in [1e-2,1e-4,1e-6]}}
        result['rows'].append(row);print(row,flush=True)
(out/'fmm_assessment.json').write_text(json.dumps(result,indent=2))
