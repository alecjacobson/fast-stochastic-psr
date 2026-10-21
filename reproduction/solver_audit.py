"""Check SDD convergence under the example's small-amplitude kernel."""
from pathlib import Path
import json,time
import jax
import jax.numpy as jnp
import numpy as np
from geospsr.kernels import Matern32Kernel,ProductKernel
from geospsr.sdd import sdd_solve
out=Path(__file__).resolve().parent/'results';data=np.load(out/'upstream_baseline.npz')
x=jnp.asarray(data['points']);y=jnp.asarray(data['normals']);k=Matern32Kernel(.01,.1);kv=ProductKernel(k,k,k)
import argparse
p=argparse.ArgumentParser();p.add_argument('--noise-variance',type=float,default=.02);a=p.parse_args()
K=jax.vmap(jax.vmap(kv,(None,0)),(0,None))(x,x)+a.noise_variance*jnp.eye(len(x));ref=jax.scipy.linalg.solve(K,y,assume_a='pos');ref.block_until_ready()
rows=[]
for lr in [.001,.1,1.]:
    t=time.perf_counter();alpha=sdd_solve(x,y,kv,a.noise_variance,jax.random.key(0),bs=128,lr=lr,iterations=1000);alpha.block_until_ready();elapsed=time.perf_counter()-t
    rows.append({'lr':lr,'iterations':1000,'seconds_first':elapsed,'relative_residual':float(jnp.linalg.norm(K@alpha-y)/jnp.linalg.norm(y)),'alpha_relative_l2':float(jnp.linalg.norm(alpha-ref)/jnp.linalg.norm(ref))})
(out/('solver_audit.json' if a.noise_variance==.02 else 'solver_audit_matched_noise.json')).write_text(json.dumps({'noise_variance':a.noise_variance,'N':len(x),'rows':rows},indent=2));print(rows)
