"""Stratified sampling of every mode omitted by the positive block model.

The corner stratum lies inside the extent cube and has at least two axes
outside the core. The outer stratum is the infinite complement of that cube.
Independent estimates from these disjoint strata add; empirical standard-error
variances add as well. These are diagnostics, not certified error bounds.
"""
import numpy as np
from geospsr.spectral_blocks import axis_weights,one_high_axis_blocks
from geospsr.spectral_tail import sample_tail_frequencies


def sample_corner_frequencies(count,core,extent,lengthscale=.01,variance=.1,seed=0):
    """Draw iid corner modes with probabilities proportional to their spectrum.

    Conditional patterns are HHL, HLH, LHH, HHH, where L is |n| <= core and
    H is core < |n| <= extent. If core == extent this stratum is empty and
    returns zero frequencies and zero mass rather than fabricating samples.
    """
    one_high_axis_blocks(core,extent)
    if count != int(count) or count < 1:
        raise ValueError('count must be a positive integer')
    low = np.arange(-core,core+1,dtype=np.int64)
    rho_low = axis_weights(low,lengthscale,variance)
    if core == extent:
        return np.empty((0,3),dtype=np.int64),0.
    high = np.r_[np.arange(-extent,-core),np.arange(core+1,extent+1)].astype(np.int64)
    rho_high = axis_weights(high,lengthscale,variance)
    l,h = rho_low.sum(),rho_high.sum()
    mass = h*h*(3*l+h)
    rng = np.random.default_rng(seed)
    # A pattern value 0/1/2 identifies its single low axis; 3 means all high.
    pattern = rng.choice(4,int(count),p=np.array([l,l,l,h])/(3*l+h))
    modes = np.empty((int(count),3),dtype=np.int64)
    for d in range(3):
        inside = pattern == d
        modes[inside,d] = rng.choice(low,int(inside.sum()),p=rho_low/l)
        modes[~inside,d] = rng.choice(high,int((~inside).sum()),p=rho_high/h)
    return modes,float(mass)


def sample_block_tail_frequencies(count,core,extent,lengthscale=.01,variance=.1,seed=0,
                                  normalization='analytic'):
    """Sample `count` frequencies per nonempty stratum using independent streams.

    Returns a dict with `corners` and `outside`, each containing `frequencies`
    and physical covariance `mass`. This is stratified, not an iid sample of
    the combined law: weight each stratum by its own mass and ADD its estimate.
    """
    if normalization != 'analytic':
        raise ValueError('block-tail sampling requires analytic normalization')
    one_high_axis_blocks(core,extent)
    if count != int(count) or count < 1:
        raise ValueError('count must be a positive integer')
    streams = np.random.SeedSequence(seed).spawn(2)
    corner = sample_corner_frequencies(count,core,extent,lengthscale,variance,streams[0])
    outer = sample_tail_frequencies(int(count),extent,lengthscale,variance,streams[1])
    return {name:{'frequencies':modes,'mass':mass}
            for name,(modes,mass) in zip(['corners','outside'],[corner,outer])}
