"""Nested finite geometry and paired-disagreement probability bounds."""
import numpy as np
from scipy.stats import binomtest


def nested_rays(rays,distance,levels=4):
    rays=np.asarray(rays);distance=np.asarray(distance)
    if rays.ndim!=3 or rays.shape[-1]!=3 or rays.shape[1]!=len(distance) or len(distance)<2:
        raise ValueError('expected ray grid and matching distance samples')
    if levels!=int(levels) or levels<0 or not np.all(np.diff(distance)>0):
        raise ValueError('require nonnegative levels and increasing distance')
    positions=[rays.copy()];distances=[distance.copy()]
    for _ in range(int(levels)):
        x=positions[-1];d=distances[-1]
        fine=np.empty((len(x),2*x.shape[1]-1,3),dtype=x.dtype)
        fine[:,::2]=x;fine[:,1::2]=(x[:,:-1]+x[:,1:])*.5
        fd=np.empty(2*len(d)-1,dtype=d.dtype);fd[::2]=d;fd[1::2]=(d[:-1]+d[1:])*.5
        positions.append(fine);distances.append(fd)
    return positions,distances


def nested_spheres(objects,levels=4,radius=.12,seed=1):
    objects=np.asarray(objects)
    if objects.ndim!=3 or objects.shape[-1]!=3 or objects.shape[1]<1:
        raise ValueError('expected sphere-surface arrays')
    if levels!=int(levels) or levels<0 or radius<=0: raise ValueError('invalid refinement parameters')
    n=objects.shape[1];positions=[objects.copy()]
    unit=np.random.default_rng(seed).normal(size=(n*2**int(levels),3))
    unit/=np.linalg.norm(unit,axis=1)[:,None]
    centers=objects[:,0].astype('d')-radius*unit[0]
    if not np.allclose(centers[:,None]+radius*unit[None,:n],objects,rtol=0,atol=2e-6):
        raise ValueError('source sphere points do not match declared radius/seed')
    for level in range(1,int(levels)+1):
        fine=(centers[:,None]+radius*unit[None,:n*2**level]).astype(objects.dtype)
        fine[:,:len(positions[-1][0])]=positions[-1]
        positions.append(fine)
    return positions


def discrepancy_bound(disagreeing_draws,count,comparisons=8,alpha=.05):
    """Upper bound for ANY location disagreement on one independent draw.

    Each draw is one Bernoulli trial even when spatial locations correlate.
    Bonferroni covers the declared comparison family at one fixed final look.
    Any pointwise probability difference is at most this any-event probability.
    This compares finite grids and does not bound errors shared by both grids.
    """
    if (count!=int(count) or count<1 or disagreeing_draws!=int(disagreeing_draws)
            or not 0<=disagreeing_draws<=count or comparisons!=int(comparisons) or comparisons<1 or not 0<alpha<1):
        raise ValueError('invalid binomial count or confidence family')
    upper=binomtest(int(disagreeing_draws),int(count),alternative='less').proportion_ci(
        confidence_level=1-alpha/comparisons,method='exact').high
    return {'disagreeing_draws':int(disagreeing_draws),'count':int(count),
            'empirical_any_probability':disagreeing_draws/count,'upper':float(upper),
            'comparisons':int(comparisons),'family_alpha':alpha,'pass_2pp':bool(upper<=.02)}
