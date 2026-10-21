"""Compare mean evaluation for the actual teaser against upstream's pairwise kernel.

Both receive identical solved weights. Training, variance, and rendering are
excluded from the speedup. The full released dense fit would require a 4-TB K.
"""
import json,time
from pathlib import Path
from functools import partial
import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np
from geospsr.gp import poisson_cross_covariances
from geospsr.kernels import Matern32Kernel,ProductKernel
from geospsr.utils import periodic_stationary_interpolator,trunc_Zd
from geospsr.spectral import FourierPoisson

root=Path(__file__).resolve().parents[1];out=root/'reproduction/results/teaser'
cloud=np.load(out/'cloud_1000000.npz');x=cloud['points'];y=cloud['normals']
w=np.load(out/'weights.npz');alpha=w['alpha'];offset=float(w['offset'])
f=np.load(out/'fields.npz');g=f['axis'];xx,zz=np.meshgrid(g,g,indexing='xy')
q=np.concatenate([np.stack([xx.ravel(),np.full(xx.size,h),zz.ravel()],axis=1) for h in f['heights']]).astype('f')
result={'N':len(x),'M':len(q),'scope':'mean cross-covariance action for both rendered slices, same solved weights; excludes training, variance, rendering',
        'upstream_reference':'released poisson_cross_covariances + periodic_stationary_interpolator + batched sum, with solved weights supplied',
        'pairwise_repetitions':1,'nufft_repetitions':3,'dtype':'float32, highest JAX matmul precision','cutoff':50,'interpolation_grid':50,
        'dense_training_matrix_bytes':4*len(x)**2}
def save(): (out/'comparison.json').write_text(json.dumps(result,indent=2))
def sync(value=None):
    if value is not None:jax.block_until_ready(value)
    cp.cuda.runtime.deviceSynchronize()

def measure(fn):
    sync();begin=time.perf_counter();v=fn();sync(v);return v,time.perf_counter()-begin

k=Matern32Kernel(.01,.1);kv=ProductKernel(k,k,k)
freq=trunc_Zd(50);ev=jax.vmap(kv.spectral_density)(freq)**.5
cross=partial(poisson_cross_covariances,eigenvectors=freq,eigenvalues=ev)
kfv,secs=measure(lambda:periodic_stationary_interpolator(cross,3,50,exponent=5))
result['upstream_interpolation_setup_seconds']=secs;save();print('upstream kernel precomputation',secs,flush=True)
xj,aj,qj=jnp.asarray(x),jnp.asarray(alpha),jnp.asarray(q)
@jax.jit
def rows(queries,points,weights):
    def row(query):return (jax.vmap(kfv,(None,0))(query,points)*weights).sum()
    return jax.vmap(row)(queries)
# Compile and warm the exact block shape before the full-slice measurement.
rows(qj[:64],xj,aj).block_until_ready()
def pairwise():
    values=[]
    for start in range(0,len(qj),64):
        values.append(rows(qj[start:start+64],xj,aj))
        if start%8192==0:
            values[-1].block_until_ready();print('upstream queries',start,'/',len(qj),flush=True)
    return jnp.concatenate(values)-offset
baseline,secs=measure(pairwise);baseline=np.asarray(baseline)
result['upstream_pairwise_seconds']=secs;save();print('upstream full slices',secs,flush=True)
op=FourierPoisson();xcp=cp.asarray(x);acp=cp.asarray(alpha)
def fast():
    coeff=op.aggregate(xcp,acp)
    return op.evaluate(q,coeff)[0]-offset
fast();sync()
ts=[]
for _ in range(3):
    value,secs=measure(fast);ts.append(secs)
fast_value=cp.asnumpy(value)
result['nufft_aggregate_and_query_seconds']=ts
result['nufft_median_seconds']=float(np.median(ts))
result['speedup_mean_action']=result['upstream_pairwise_seconds']/result['nufft_median_seconds']
result['relative_l2_vs_upstream_interpolation']=float(np.linalg.norm(fast_value-baseline)/np.linalg.norm(fast_value))
result['max_abs_difference']=float(np.max(np.abs(fast_value-baseline)))
# Assess the training approximation separately on real scorpion rows.
ids=np.random.default_rng(100).choice(len(x),64,replace=False)
exact=jax.jit(jax.vmap(jax.vmap(kv,(None,0)),(0,None)))(jnp.asarray(x[ids]),xj)
exact.block_until_ready();Krows=cp.from_dlpack(exact)
res=Krows@acp+.02*acp[ids]-cp.asarray(y[ids])
result['full_analytic_kernel_residual_64_rows']=float(cp.linalg.norm(res)/cp.linalg.norm(cp.asarray(y[ids])))
np.savez_compressed(out/'comparison_fields.npz',pairwise=baseline.reshape(f['mean'].shape),nufft=fast_value.reshape(f['mean'].shape))
save();print(json.dumps(result,indent=2),flush=True)
