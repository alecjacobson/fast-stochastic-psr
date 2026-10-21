"""Synchronized fast-sum benchmarks; source coefficients supplied, no solve included."""
import json
import time
from pathlib import Path
import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.interpolate import RegularGridInterpolator
from geospsr.spectral import FourierPoisson
from geospsr.gp import poisson_cross_covariances
from geospsr.kernels import Matern32Kernel,ProductKernel
from geospsr.utils import trunc_Zd
from posterior import Posterior,sync
out=Path(__file__).resolve().parent/'results'
rng=np.random.default_rng(123)
result={'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),'jax':jax.__version__,'dtype':'float32; highest JAX matmul precision','repeats':3,'cutoff':50,'modes':101**3,'eps':1e-6,'scope':'cross-covariance action on supplied alpha; training solve excluded','rows':[]}
def bench(fn):
    sync();t=time.perf_counter();v=fn();jax.block_until_ready(v);sync();first=time.perf_counter()-t
    ts=[]
    for _ in range(3):
        sync();t=time.perf_counter();v=fn();jax.block_until_ready(v);sync();ts.append(time.perf_counter()-t)
    return v,{'first':first,'warm':ts,'median':float(np.median(ts))}
op=FourierPoisson()
xis=(np.linspace(-1,1,50,dtype='f')**5)*np.float32(np.pi)
grid=np.stack(np.meshgrid(xis,xis,xis,indexing='ij'),axis=-1).reshape(-1,3)
values,times=bench(lambda:op.cross_values(grid));result['amortization_via_nufft']=times
rgi=RegularGridInterpolator((jnp.asarray(xis),)*3,jnp.asarray(cp.asnumpy(values).reshape(50,50,50,3)))
# Direct check at full paper cutoff, independent of interpolation.
k=Matern32Kernel(.01,.1);kv=ProductKernel(k,k,k);freq=trunc_Zd(50);ev=jax.vmap(kv.spectral_density)(freq)**.5
diff=rng.uniform(-np.pi,np.pi,(32,3)).astype('f')
exact=jax.jit(jax.vmap(lambda d:poisson_cross_covariances(d,freq,ev)))(jnp.asarray(diff))
fast=cp.asnumpy(op.cross_values(diff));result['cross_values_relative_l2']=float(np.linalg.norm(fast-np.asarray(exact))/np.linalg.norm(np.asarray(exact)))
# Repeat on scan using identical Cholesky weights.
data=np.load(out/'upstream_baseline.npz');x=data['points'];q=data['query']
model=Posterior(x,data['normals'])
fast=cp.asnumpy(model.mean(q));ref=data['mean']
result['scorpion_nufft_vs_interpolation_relative_l2']=float(np.linalg.norm(fast-ref)/np.linalg.norm(ref))
result['scorpion_nufft_vs_interpolation_max_abs']=float(np.max(np.abs(fast-ref)))
_,result['scorpion_cached_mean']=bench(lambda:model.mean(q))

def direct_factory(x,alpha):
    x=jnp.asarray(x);alpha=jnp.asarray(alpha)
    @jax.jit
    def block(q,x,alpha):
        diffs=(q[:,None,:]-x[None]+jnp.pi)%(2*jnp.pi)-jnp.pi
        vals=rgi(diffs.reshape(-1,3)).reshape(len(q),len(x),3)
        return jnp.einsum('qnd,nd->q',vals,alpha)
    def evaluate(q):
        return jnp.concatenate([block(q[i:i+64],x,alpha) for i in range(0,len(q),64)])
    return evaluate
for n in [1024,4096,16384,65536,262144,1048576]:
    # Surface-supported sources; deterministic oriented unit sphere cloud.
    x=rng.normal(size=(n,3)).astype('f');x/=np.linalg.norm(x,axis=1)[:,None];x*=1.2
    alpha=rng.normal(size=(n,3)).astype('f')
    q=rng.uniform(-1.5,1.5,(4096,3)).astype('f')
    coeff,agg=bench(lambda:op.aggregate(x,alpha))
    f,query=bench(lambda:op.evaluate(q,coeff))
    row={'N':n,'M':len(q),'aggregate':agg,'cached_query':query,'aggregate_plus_query_median':agg['median']+query['median']}
    if n<=1048576:
        direct=direct_factory(x,alpha)
        d,dt=bench(lambda:direct(jnp.asarray(q)))
        row['interpolated_pairwise']=dt
        row['speedup_including_aggregation']=dt['median']/(agg['median']+query['median'])
        row['relative_l2_vs_interpolation']=float(np.linalg.norm(cp.asnumpy(f)[0]-np.asarray(d))/np.linalg.norm(np.asarray(d)))
    result['rows'].append(row)
    (out/'benchmark.json').write_text(json.dumps(result,indent=2));print(json.dumps(row),flush=True)
# Output sensitivity with fixed source aggregate.
result['output_rows']=[]
for m in [256,4096,65536,1048576]:
    q=rng.uniform(-1.5,1.5,(m,3)).astype('f')
    _,timing=bench(lambda:op.evaluate(q,coeff))
    result['output_rows'].append({'N':n,'M':m,'query':timing})
(out/'benchmark.json').write_text(json.dumps(result,indent=2))
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,4))
r=result['rows'];axes[0].loglog([x['N'] for x in r],[x['aggregate_plus_query_median'] for x in r],'o-',label='NUFFT source + query')
r=[x for x in r if 'interpolated_pairwise' in x];axes[0].loglog([x['N'] for x in r],[x['interpolated_pairwise']['median'] for x in r],'o-',label='Pairwise interpolation');axes[0].set(xlabel='N (M=4096)',ylabel='Seconds');axes[0].legend()
r=result['output_rows'];axes[1].loglog([x['M'] for x in r],[x['query']['median'] for x in r],'o-');axes[1].set(xlabel='M (N=1,048,576, source cached)',ylabel='Seconds')
fig.tight_layout();fig.savefig(out/'scaling.png',dpi=160)
