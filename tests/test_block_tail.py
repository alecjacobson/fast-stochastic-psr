"""Independent tests of corner/outer-tail sampling and posterior PSD brackets."""
import cupy as cp
import numpy as np
import pytest
from geospsr.block_tail import sample_block_tail_frequencies,sample_corner_frequencies
from geospsr.spectral_blocks import axis_weights,block_audit
from geospsr.spectral_tail import analytic_covariance_rows64,tail_quadratic_samples
from reproduction.block_reference import block_covariance_rows
from geospsr.training_spectrum import FourierCovariance,finite_covariance_rows


def test_corner_frequency_law_matches_independent_enumeration():
    core,extent,ell,count = 1,3,.2,131072
    modes,mass = sample_corner_frequencies(count,core,extent,ell,seed=12)
    axis = np.arange(-extent,extent+1)
    grid = np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
    corner = (np.abs(grid)>core).sum(1)>=2
    rho = axis_weights(grid,ell).prod(1)
    np.testing.assert_allclose(mass,rho[corner].sum(),rtol=2e-15)
    assert np.all((np.abs(modes)>core).sum(1)>=2)
    assert np.max(np.abs(modes)) <= extent
    index = np.ravel_multi_index((modes+extent).T,(2*extent+1,)*3)
    observed = np.bincount(index,minlength=len(grid))
    assert not observed[~corner].any()
    probability = rho[corner]/mass
    error = observed[corner]-count*probability
    assert np.max(np.abs(error)/np.sqrt(count*probability*(1-probability))) < 6


@pytest.mark.parametrize('ell,core,extent',[(.04,2,5),(.005,20,40),(1.,1,3)])
def test_stratified_tail_against_analytic_covariance(ell,core,extent):
    rng = np.random.default_rng(412)
    x = rng.uniform(-1,1,(13,3)).astype('f')
    x[1:5] = x[0]+rng.normal(0,.003,(4,3))
    weights = rng.normal(size=(len(x),3,3)).astype('f')
    strata = sample_block_tail_frequencies(65536,core,extent,ell,seed=34)
    expected_mass = block_audit(core,extent,ell)
    np.testing.assert_allclose(strata['corners']['mass'],expected_mass['omitted_inside_cube_diagonal'],rtol=2e-14)
    np.testing.assert_allclose(strata['outside']['mass'],expected_mass['omitted_outside_cube_diagonal'],rtol=2e-10)
    assert np.all(np.max(np.abs(strata['outside']['frequencies']),axis=1)>extent)
    full = cp.asnumpy(analytic_covariance_rows64(x,x,ell))
    finite = cp.asnumpy(block_covariance_rows(x,x,core,extent,ell))
    cube = cp.asnumpy(finite_covariance_rows(x,x,FourierCovariance(extent,ell),dtype='d'))
    estimates,variances = [],[]
    for name,delta in [('corners',cube-finite),('outside',full-cube)]:
        assert np.linalg.eigvalsh(delta).min() > -1e-15
        np.testing.assert_allclose(np.diag(delta),strata[name]['mass'],rtol=3e-10,atol=2e-17)
        exact = np.einsum('iqd,ij,jkd->qk',weights.astype('d'),delta,weights.astype('d'))
        draws = tail_quadratic_samples(x,weights,**strata[name],source_batch=5,frequency_batch=2048)
        mean,se = draws.mean(0),draws.std(0,ddof=1)/np.sqrt(len(draws))
        assert np.all(np.abs(mean-exact)<6*se+1e-12)
        estimates.append(mean);variances.append(se**2)
    exact_total = np.einsum('iqd,ij,jkd->qk',weights.astype('d'),full-finite,weights.astype('d'))
    assert np.all(np.abs(sum(estimates)-exact_total)<6*np.sqrt(sum(variances))+1e-12)
    # The same bracket works for any positive retained spectrum, not just cubes.
    cross = rng.normal(size=(len(x),4))
    noise = .0002
    a = np.linalg.solve(finite+noise*np.eye(len(x)),cross)
    gap = cross.T@(a-np.linalg.solve(full+noise*np.eye(len(x)),cross))
    bound = a.T@(full-finite)@a
    assert np.linalg.eigvalsh((gap+gap.T)/2).min() > -1e-10*np.linalg.norm(bound)
    assert np.linalg.eigvalsh((bound+bound.T-gap-gap.T)/2).min() > -1e-10*np.linalg.norm(bound)


def test_empty_corner_and_invalid_tail_requests():
    strata = sample_block_tail_frequencies(1000,3,3,.1,seed=4)
    assert strata['corners']['mass']==0
    assert strata['corners']['frequencies'].shape==(0,3)
    assert len(strata['outside']['frequencies'])==1000
    for core,extent in [(-1,3),(4,3),(1.5,3)]:
        with pytest.raises(ValueError):
            sample_block_tail_frequencies(10,core,extent)
    for count in [0,-1,1.5]:
        with pytest.raises(ValueError):
            sample_block_tail_frequencies(count,1,3)
    with pytest.raises(ValueError,match='analytic normalization'):
        sample_block_tail_frequencies(10,1,3,normalization='released')
