"""Headless reproduction of experiments/example.py with synchronized timings."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--upstream', action='store_true')
p.add_argument('--cutoff', type=int, default=50)
p.add_argument('--density', type=int, default=50)
p.add_argument('--grid', type=int, default=100)
p.add_argument('--samples', type=int, default=4)
p.add_argument('--skip-sdd', action='store_true')
a = p.parse_args()
ROOT = Path(__file__).resolve().parents[1]
if a.upstream:
    upstream = Path(os.environ.get('GEOSPSR_RUNTIME','/opt/geospsr-runtime'))/'upstream'
    if not (upstream/'geospsr/gp.py').is_file():
        raise SystemExit('Upstream worktree missing: run reproduction/setup.sh first.')
    sys.path.insert(0, str(upstream))
sys.path.insert(0, str(ROOT / 'experiments'))
import igl
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from utils import normalize_points, simulate_scan
from geospsr.gp import (compute_mean, compute_variance, f_eigenvalues_from_v_eigenvalues,
                       f_gamma_from_v_xi, poisson_cross_covariances, sample_pathwise_conditioning)
from geospsr.kernels import Matern32Kernel, ProductKernel
from geospsr.utils import periodic_stationary_interpolator, trunc_Zd

out = ROOT / 'reproduction/results'
tag = 'upstream' if a.upstream else 'corrected'
result = {'args': vars(a), 'jax': jax.__version__, 'devices': list(map(str, jax.devices())),
          'timing_policy': 'wall clock, block_until_ready; first invocation includes compilation', 'seconds': {}}
def timed(name, fn):
    t = time.perf_counter()
    value = fn()
    jax.block_until_ready(value)
    result['seconds'][name] = time.perf_counter() - t
    print(name, result['seconds'][name], flush=True)
    (out / f'{tag}_baseline.json').write_text(json.dumps(result, indent=2))
    return value

read_obj = getattr(igl, 'read_obj', None) or igl.readOBJ
v, _, _, f, _, _ = read_obj(str(ROOT / 'experiments/meshes/scorpion.obj'))
v = normalize_points(v) * np.pi
points, normals = simulate_scan(v, f, np.array([[0,-2,0],[1,1,1]])*np.pi,
                                np.array([[0,1,0],[-1,-1,-1]])*np.pi/2, np.pi/3, 200)
rng = np.random.default_rng(0)
points += rng.standard_normal(points.shape) * .02
normals += rng.standard_normal(normals.shape) * .02
x, y = jnp.asarray(points), jnp.asarray(normals)
g = np.linspace(-np.pi/2, np.pi/2, a.grid)
g1,g2 = np.meshgrid(g,g,indexing='ij')
q = jnp.asarray(np.stack([g1.ravel(),np.full(g1.size,-.75),g2.ravel()],axis=-1))
result['N'] = len(x)
result['M'] = len(q)
print('device',jax.devices(),'N',len(x),'M',len(q),flush=True)
k = Matern32Kernel(.01,.1)
kv = ProductKernel(k,k,k)
freq = trunc_Zd(a.cutoff)
ev = jax.vmap(kv.spectral_density)(freq)**.5
cross = partial(poisson_cross_covariances,eigenvectors=freq,eigenvalues=ev)
kfv = timed('amortization_first',lambda: periodic_stationary_interpolator(cross,3,a.density,exponent=5))
fev = f_eigenvalues_from_v_eigenvalues(freq,ev)
varf = (fev**2).sum()
mu_data = timed('mean_data_first',lambda: compute_mean(x,x,y,kv,kfv,.02,bs=128))
mu = timed('mean_grid_first',lambda: compute_mean(q,x,y,kv,kfv,.02,bs=128))
mu_warm = timed('mean_grid_repeat',lambda: compute_mean(q,x,y,kv,kfv,.02,bs=128))
var = timed('variance_grid_first',lambda: compute_variance(q,x,kv,kfv,varf,.02,bs=64))
xi = jax.random.normal(jax.random.key(0),(a.samples,3,len(freq),2))
gamma = f_gamma_from_v_xi(xi,freq,ev)
samples = timed('samples_grid_first',lambda: sample_pathwise_conditioning(q,x,y,kv,kfv,.02,xi,gamma,freq,ev,freq,fev,64,64,64))
arrays = dict(points=points,normals=normals,query=np.asarray(q),mean=np.asarray(mu-mu_data.mean()),
              variance=np.asarray(var),samples=np.asarray(samples),offset=np.asarray(mu_data.mean()))
if not a.skip_sdd:
    kwargs = dict(key=jax.random.key(0),lr=1e-3,bs=128,iterations=1000)
    smu = timed('sdd_mean_grid_first',lambda: compute_mean(q,x,y,kv,kfv,.02,bs=128,sdd_params=kwargs))
    arrays['sdd_mean_uncentered'] = np.asarray(smu)
    result['sdd_vs_cholesky_relative_l2'] = float(jnp.linalg.norm(smu-mu)/jnp.linalg.norm(mu))
result['finite'] = all(np.isfinite(value).all() for value in arrays.values())
result['variance_min'] = float(var.min())
np.savez_compressed(out / f'{tag}_baseline.npz',**arrays)
fig,axes = plt.subplots(1,3,figsize=(13,4))
for ax,z,title in zip(axes,[arrays['mean'],arrays['variance'],arrays['samples'][0]-arrays['offset']],['Posterior mean','Marginal variance','Posterior sample']):
    im=ax.pcolormesh(g1,g2,z.reshape(g1.shape),shading='auto',cmap='RdBu_r' if title!='Marginal variance' else 'viridis')
    if title!='Marginal variance': ax.contour(g1,g2,z.reshape(g1.shape),levels=[0],colors='k',linewidths=.6)
    ax.set_aspect('equal'); ax.set_title(title); fig.colorbar(im,ax=ax)
fig.tight_layout(); fig.savefig(out / f'{tag}_baseline.png',dpi=160)
(out / f'{tag}_baseline.json').write_text(json.dumps(result,indent=2))
