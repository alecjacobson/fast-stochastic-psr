"""Independent sums, dynamic-range leakage and CG checks for packed RHS."""
import cupy as cp
import numpy as np
import pytest
from geospsr.packed_spectral import PackedFourierKernel
from geospsr.training_spectrum import FourierCovariance
from geospsr.spectral import FourierKernel
from geospsr.half_spectral import HalfCubeSpectrum,HalfCubeKernel
from geospsr.spectral_blocks import axis_weights


@pytest.mark.parametrize('count,dtype,cutoff',[(1,'float64',0),(2,'float64',4),
    (3,'float32',5),(6,'float32',4),(7,'float64',5)])
def test_packed_actions_and_solves(count,dtype,cutoff):
    rng=np.random.default_rng(927)
    x=rng.uniform(-np.pi,np.pi,(23,3)).astype(dtype)
    x[:3]=[[np.pi-1e-5,0,0],[-np.pi+1e-5,0,0],[np.pi-2e-5,0,0]]
    weights=rng.normal(size=(len(x),count)).astype(dtype)
    weights*=np.geomspace(1e-12,1e12,count).astype(dtype)
    if count>3:
        weights[:,3]=0
    eps,upsamp=(1e-11,2.) if dtype=='float64' else (1e-6,1.25)
    spectrum=FourierCovariance(cutoff,lengthscale=1.,dtype=dtype,eps=eps,upsampfac=upsamp)
    kernel=PackedFourierKernel(x,spectrum,.0002,count)
    full=FourierKernel(x,spectrum,.0002,count)
    half=HalfCubeKernel(x,HalfCubeSpectrum(cutoff,lengthscale=1.,dtype=dtype,eps=eps,upsampfac=upsamp),.0002,count)
    ns=np.stack(np.meshgrid(*[np.arange(-cutoff,cutoff+1)]*3,indexing='ij'),-1).reshape(-1,3)
    rho=axis_weights(ns,1.).prod(1)
    matrix=np.cos((x[:,None].astype('d')-x[None].astype('d'))@ns.T)@rho+.0002*np.eye(len(x))
    expected=matrix@weights
    nonzero=np.linalg.norm(weights.astype('d'),axis=0)>0
    tolerance=4e-6 if dtype=='float32' else 3e-10
    actual=cp.asnumpy(kernel(weights))
    for reference in [expected,cp.asnumpy(full(weights)),cp.asnumpy(half(weights))]:
        error=np.linalg.norm((actual-reference).astype('d'),axis=0)[nonzero]/np.linalg.norm(reference.astype('d'),axis=0)[nonzero]
        assert error.max()<tolerance
    assert np.all(actual[:,~nonzero]==0)
    rtol=1e-5 if dtype=='float32' else 1e-9
    for initial in [None,.3*weights]:
        saved=None if initial is None else initial.copy()
        alpha,info=kernel.solve(weights,rtol=rtol,x0=initial)
        assert info['converged']
        residual=matrix@cp.asnumpy(alpha)-weights
        assert (np.linalg.norm(residual,axis=0)[nonzero]/np.linalg.norm(weights.astype('d'),axis=0)[nonzero]).max()<2*rtol
        assert np.all(cp.asnumpy(alpha)[:,~nonzero]==0)
        if initial is not None:
            np.testing.assert_array_equal(initial,saved)
    with pytest.raises(ValueError,match='real RHS'):
        kernel.solve(weights.astype('complex128'))


def test_packed_rejects_asymmetric_spectrum():
    spectrum=FourierCovariance(2)
    spectrum.rho[0,1,1]*=2
    with pytest.raises(ValueError,match='real even'):
        PackedFourierKernel(cp.zeros((3,3)),spectrum,.02)


@pytest.mark.parametrize('solve_cutoff',[None,3])
def test_packed_conditioning_preserves_coherent_draws_and_cache(solve_cutoff):
    from reproduction.half_posterior import HalfCubePosterior
    rng=np.random.default_rng(28)
    x=rng.uniform(-1,1,(13,3)).astype('f');q=rng.uniform(-1,1,(5,3)).astype('f')
    y=rng.normal(size=x.shape).astype('f')
    model=HalfCubePosterior(x,y,training_cutoff=5,cutoff=2,lengthscale=.1,noise_variance=.0002)
    model.draw(3,seed=7,batch=2,retain_components=True,retain_rhs=True,solve_cutoff=solve_cutoff)
    rhs=model.sample_rhs.copy();prior=model.sample_prior_coeff.copy();fields=model.samples(q).copy()
    model.draw(3,seed=7,batch=2,retain_components=True,retain_rhs=True,solve_cutoff=solve_cutoff,pack_rhs=True)
    np.testing.assert_array_equal(cp.asnumpy(model.sample_rhs),cp.asnumpy(rhs))
    np.testing.assert_array_equal(cp.asnumpy(model.sample_prior_coeff),cp.asnumpy(prior))
    assert float(cp.linalg.norm(model.samples(q)-fields)/cp.linalg.norm(fields))<3e-5
    cutoff=5 if solve_cutoff is None else solve_cutoff
    ns=np.stack(np.meshgrid(*[np.arange(-cutoff,cutoff+1)]*3,indexing='ij'),-1).reshape(-1,3)
    rho=axis_weights(ns,.1).prod(1)
    k=np.cos((x[:,None].astype('d')-x[None].astype('d'))@ns.T)@rho
    k+=.0002*np.eye(len(x))
    flat=cp.asnumpy(rhs).transpose(1,0,2).reshape(len(x),-1)
    if solve_cutoff is None:
        exact=np.linalg.solve(k,flat)
    else:
        exact=np.linalg.solve(k,flat-np.tile(y,(1,3)))+np.tile(cp.asnumpy(model.alpha),(1,3))
    actual=cp.asnumpy(model.sample_weights).transpose(1,0,2).reshape(len(x),-1)
    assert np.linalg.norm(actual-exact)/np.linalg.norm(exact)<3e-5
    model.prepare_query_cache(batch=2)
    assert float(cp.linalg.norm(model.samples(q)-fields)/cp.linalg.norm(fields))<3e-5
    assert all(r['pack_rhs'] for r in model.sample_solve_records)
    assert model._packed_kernel is None and model._packed_spectrum is None
