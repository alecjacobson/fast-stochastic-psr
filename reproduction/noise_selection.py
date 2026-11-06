"""Independent geometry validation for a noisy-normal H2/locality comparison."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from reproduction.benchmark_hierarchical_variance import data
from reproduction.run_matched_lengthscales import digest
from geospsr.nonperiodic import FreeSpacePoissonMatern, RegionalVarianceCache
from geospsr.regional_field import _coefficients

AMPLITUDE = 1/3  # Fixed component variance scale for unit-vector observations.
HALFWIDTH = .005
SIZES = [2048,8192,32768,131072]
LENGTHS = [.015,.03,.06,.12,.24,.48]
NOISES = [.05,.2]
VALIDATION_SEEDS = [3101,3102]
TUNING_SEED = 4101
TEST_SEEDS = [5101,5102,5103]
NOISE_REPLICATES = 8


def validation_observations(clean,sigma,seed,replicates=NOISE_REPLICATES):
    """Independent noisy realizations sharing positions and the same likelihood."""
    samples=[clean+sigma*np.random.default_rng(seed+1000+100000*rep).normal(size=clean.shape)
             for rep in range(replicates)]
    return samples+[clean]


def surface_points(seed,count=256,halfwidth=HALFWIDTH):
    yz=np.random.default_rng(seed).uniform(-.95*halfwidth,.95*halfwidth,(count,2))
    x=np.sqrt(1-(yz[:,0]/.8)**2-(yz[:,1]/.6)**2)
    return np.column_stack((x,yz))


def mean_values(coefficients,points,halfwidth=HALFWIDTH):
    t=(np.asarray(points)-[1,0,0])/halfwidth
    return np.polynomial.chebyshev.chebval3d(*t.T,coefficients)


def surface_metrics(coefficients,points,halfwidth=HALFWIDTH):
    points=np.asarray(points);level=float(mean_values(coefficients,[[1,0,0]],halfwidth)[0])
    left=points.copy();right=points.copy();left[:,0]=1-halfwidth;right[:,0]=1+halfwidth
    fl=mean_values(coefficients,left,halfwidth)-level;fr=mean_values(coefficients,right,halfwidth)-level
    valid=np.isfinite(fl)&np.isfinite(fr)&(fl*fr<=0)
    # Vectorized bracketed roots; missing roots receive a fixed full-box penalty.
    for _ in range(30):
        middle=(left+right)/2;fm=mean_values(coefficients,middle,halfwidth)-level
        take_left=fl*fm<=0
        right[take_left]=middle[take_left];left[~take_left]=middle[~take_left];fl[~take_left]=fm[~take_left]
    error=(left[:,0]+right[:,0])/2-points[:,0]
    penalized=np.where(valid,error,2*halfwidth)
    gradient=np.column_stack([mean_values(np.polynomial.chebyshev.chebder(coefficients,axis=d),points,halfwidth)/halfwidth for d in range(3)])
    truth=points/[1,.8**2,.6**2];truth/=np.linalg.norm(truth,axis=1)[:,None]
    norm=np.linalg.norm(gradient,axis=1)
    return dict(penalized_surface_rms=float(np.sqrt(np.mean(penalized**2))),
                surface_rms=float(np.sqrt(np.mean(error[valid]**2))) if valid.any() else None,
                missing_roots=int((~valid).sum()),count=len(points),
                normal_alignment=float(np.mean(np.sum(gradient*truth,axis=1)/norm)),
                root_errors=[float(e) if ok else None for e,ok in zip(error,valid)],gradient_norm=norm.tolist())


def build_mean_fields(solve,model,x,observations,order=7):
    count=len(observations);rhs=np.column_stack(observations)
    alpha=solve(rhs)
    cache=RegionalVarianceCache.__new__(RegionalVarianceCache);cache._initialize_box([1,0,0],HALFWIDTH,order)
    means=np.zeros((len(cache.nodes),count))
    for first in range(0,len(x),4096):
        last=min(first+4096,len(x));c=model.cross_covariance(cache.nodes,x[first:last])
        weights=alpha[first:last].reshape(last-first,count,3).transpose(0,2,1)
        means+=np.einsum('qnd,ndb->qb',c,weights,optimize=True)
    return [_coefficients(means[:,i].reshape((order,)*3)) for i in range(count)],alpha


def run_one(n,sigma,ell,seed,output):
    import cupy as cp
    from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem
    from geospsr.hierarchical import HierarchicalMaternSystem
    from geospsr.hierarchical_gpu import GPUHierarchicalSystem
    start=perf_counter();x,noisy,_=data(n,seed,'ellipsoid',normal_noise=sigma)
    _,clean,_=data(n,seed,'ellipsoid')
    model=FreeSpacePoissonMatern(ell,AMPLITUDE);noise=sigma**2
    if n<=8192:
        system=GPUDenseMaternSystem(model,x,noise,dtype='float64');backend='float64 dense'
    else:
        native=HierarchicalMaternSystem(model,x,noise,tolerance=1e-7,leaf_size=40,
            preconditioner_rank=50,neighbors=40,threads=4,seed=seed,preconditioner_backend='prefix-kd')
        system=GPUHierarchicalSystem(native,interaction_budget_bytes=256*2**20,rhs_block_size=72,release_native=True)
        backend='H2 tolerance 1e-7, solve tolerance 1e-10'
    solve=(lambda b:system.solve(b,tolerance=1e-10)) if backend.startswith('H2') else system.solve
    observations=validation_observations(clean,sigma,seed)
    np.testing.assert_array_equal(observations[0],noisy)
    fields,alpha=build_mean_fields(solve,model,x,observations)
    q=surface_points(6101);metrics=[surface_metrics(f,q) for f in fields]
    # Direct non-interpolated evaluation checks the mean cache on held-out nodes.
    probe=q[::32];direct=np.zeros((len(probe),len(observations)))
    for first in range(0,n,8192):
        last=min(first+8192,n);c=model.cross_covariance(probe,x[first:last])
        direct+=np.einsum('qnd,ndb->qb',c,alpha[first:last].reshape(last-first,len(observations),3).transpose(0,2,1),optimize=True)
    interpolation=[float(np.max(abs(mean_values(f,probe)-direct[:,i]))) for i,f in enumerate(fields)]
    result=dict(n=n,normal_noise=sigma,noise_variance=noise,amplitude=AMPLITUDE,lengthscale=ell,seed=seed,
                validation_point_seed=6101,backend=backend,mean_only_selection=True,
                noisy_metrics=metrics[:-1],clean_same_likelihood_metrics=metrics[-1],noise_replicates=NOISE_REPLICATES,
                mean_coefficients=[f.tolist() for f in fields],interpolation_max_mean_error=interpolation,
                mean_solve_info=getattr(system,'last_solve',None),seconds_excluded=perf_counter()-start,
                source_sha256={p:digest(p) for p in [__file__,'geospsr/nonperiodic.py','geospsr/hierarchical_gpu.py',
                    'reproduction/benchmark_hierarchical_variance.py']})
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(n,sigma,ell,seed,float(np.sqrt(np.mean([m['penalized_surface_rms']**2 for m in metrics[:-1]]))),max(interpolation),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name,typ in [('n',int),('sigma',float),('ell',float),('seed',int)]:p.add_argument('--'+name,type=typ,required=True)
    p.add_argument('--output',required=True);a=p.parse_args();run_one(a.n,a.sigma,a.ell,a.seed,a.output)
