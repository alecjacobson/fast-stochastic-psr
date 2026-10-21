"""Large-N NUFFT-CG experiment, with analytic-kernel residual spot checks."""
import json,time
from pathlib import Path
import numpy as np
import cupy as cp
import jax
import jax.numpy as jnp
from geospsr.spectral import FourierPoisson,FourierKernel
from geospsr.kernels import Matern32Kernel,ProductKernel
from posterior import sync
out=Path(__file__).resolve().parent/'results'
rng=np.random.default_rng(321)
result={'scope':'three normal RHS, finite-Fourier training model; analytic residual checked on 64 random rows','noise_variance':.02,'lengthscale':.01,'rtol':1e-5,'rows':[]}
k=Matern32Kernel(.01,.1);kv=ProductKernel(k,k,k)
@jax.jit
def exact_rows(q,x):return jax.vmap(jax.vmap(kv,(None,0)),(0,None))(q,x)
for cutoff,n in [(50,4096),(100,4096),(50,16384),(50,65536),(50,262144),(50,1048576),(100,1048576)]:
    x=rng.normal(size=(n,3)).astype('f');x/=np.linalg.norm(x,axis=1)[:,None];normals=x.copy();x*=1.2
    # Use the same small cloud for cutoff convergence.
    if n in (4096,1048576):
        if cutoff==50:saved=(x.copy(),normals.copy())
        else:x,normals=saved
    sync();t=time.perf_counter();op=FourierPoisson(cutoff=cutoff);K=FourierKernel(x,op,.02);sync();setup=time.perf_counter()-t
    rhs=cp.asarray(normals)
    K(rhs);sync()  # warm transform plans and kernels
    t=time.perf_counter();alpha,info=K.solve(rhs);sync();elapsed=time.perf_counter()-t
    ids=rng.choice(n,64,replace=False)
    exact=exact_rows(jnp.asarray(x[ids]),jnp.asarray(x));exact.block_until_ready()
    exact=cp.asarray(np.asarray(exact))
    residual=exact@alpha+.02*alpha[ids]-rhs[ids]
    row={'N':n,'cutoff':cutoff,'setup_seconds':setup,'solve_seconds':elapsed,
         'analytic_diagonal_fraction_retained':float(op.rho.sum()/.001),
         'analytic_kernel_relative_residual_64_rows':float(cp.linalg.norm(residual)/cp.linalg.norm(rhs[ids])),**info}
    if n==4096:
        # Dense reference includes full analytic K, factorization, and solve.
        t=time.perf_counter();full=exact_rows(jnp.asarray(x),jnp.asarray(x));full.block_until_ready()
        full=cp.asarray(np.asarray(full))+.02*cp.eye(n,dtype='f')
        ref=cp.linalg.solve(full,rhs);sync();row['dense_solve_seconds']=time.perf_counter()-t
        row['alpha_relative_l2_vs_analytic']=float(cp.linalg.norm(alpha-ref)/cp.linalg.norm(ref))
    result['rows'].append(row);(out/'training.json').write_text(json.dumps(result,indent=2));print(json.dumps(row),flush=True)
    del K,op,exact,alpha
    cp.get_default_memory_pool().free_all_blocks()
