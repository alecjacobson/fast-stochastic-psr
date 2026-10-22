"""Independent cube sums and solve checks for real-field conjugacy folding."""
import cupy as cp
import numpy as np
import pytest
from geospsr.half_spectral import HalfCubeSpectrum,HalfCubeKernel
from geospsr.spectral import FourierKernel
from geospsr.spectral_blocks import axis_weights
from geospsr.training_spectrum import FourierCovariance


@pytest.mark.parametrize('cutoff,axis,ell,dtype,normalization',[
    (0,0,.1,'float64','analytic'),(1,1,.1,'float64','analytic'),
    (4,0,.03,'float64','analytic'),(5,2,1.,'float64','analytic'),
    (4,1,1.,'float64','released'),(5,0,.03,'float32','analytic')])
def test_half_cube_matches_direct_sums_and_full_cube(cutoff,axis,ell,dtype,normalization):
    rng=np.random.default_rng(41)
    x=rng.uniform(-np.pi,np.pi,(19,3)).astype(dtype)
    x[:3]=[[np.pi-1e-5,0,0],[-np.pi+1e-5,0,0],[np.pi-2e-5,0,0]]
    weights=rng.normal(size=(len(x),4)).astype(dtype)
    ns=np.stack(np.meshgrid(*[np.arange(-cutoff,cutoff+1)]*3,indexing='ij'),-1).reshape(-1,3)
    rho=axis_weights(ns,ell).prod(1)
    if normalization=='released':
        rho/=(1+np.exp(-np.sqrt(3)/ell))**3
    matrix=np.cos((x[:,None].astype('d')-x[None].astype('d'))@ns.T)@rho+.0002*np.eye(len(x))
    eps,upsamp=(1e-11,2.) if dtype=='float64' else (1e-6,1.25)
    spectrum=HalfCubeSpectrum(cutoff,ell,dtype=dtype,eps=eps,upsampfac=upsamp,normalization=normalization)
    kernel=HalfCubeKernel(x,spectrum,.0002,rhs_count=4,axis=axis)
    cube=FourierKernel(x,FourierCovariance(cutoff,ell,dtype=dtype,eps=eps,upsampfac=upsamp,normalization=normalization),.0002,rhs_count=4)
    tolerance=3e-6 if dtype=='float32' else 2e-10
    actual=cp.asnumpy(kernel(weights));expected=matrix@weights
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<tolerance
    assert np.linalg.norm(cp.asnumpy(cube(weights))-actual)/np.linalg.norm(expected)<tolerance
    rtol=1e-5 if dtype=='float32' else 1e-9
    alpha,info=kernel.solve(weights,rtol=rtol)
    assert info['converged']
    assert np.max(np.linalg.norm(matrix@cp.asnumpy(alpha)-weights,axis=0)/np.linalg.norm(weights,axis=0))<2*rtol
    assert spectrum.retained_modes==(2*cutoff+1)**3
    assert spectrum.stored_modes==(cutoff+1)*(2*cutoff+1)**2
    with pytest.raises(ValueError,match='real RHS'):
        kernel(weights.astype('complex128'))
    with pytest.raises(ValueError,match='real RHS'):
        kernel.solve(weights.astype('complex128'))


def test_invalid_half_cube_arguments():
    for cutoff in [-1,1.5]:
        with pytest.raises(ValueError):
            HalfCubeSpectrum(cutoff)
    with pytest.raises(ValueError,match='dtype'):
        HalfCubeSpectrum(2,dtype='float16')
    with pytest.raises(ValueError,match='axis'):
        HalfCubeKernel(np.zeros((2,3)),HalfCubeSpectrum(2),.02,axis=3)


def explicit_joint(x,q,training,scalar,ell):
    ns=np.stack(np.meshgrid(*[np.arange(-training,training+1)]*3,indexing='ij'),-1).reshape(-1,3)
    rho=axis_weights(ns,ell).prod(1)
    k=np.cos((x[:,None]-x[None])@ns.T)@rho
    selected=(np.abs(ns)<=scalar).all(1)&((ns*ns).sum(1)>0)
    ns,rho=ns[selected],rho[selected]
    norm=(ns*ns).sum(1)
    cross=np.einsum('qnf,fd->qnd',np.sin((q[:,None]-x[None])@ns.T),rho[:,None]*ns/norm[:,None])
    prior=np.cos((q[:,None]-q[None])@ns.T)@(rho/norm)
    return k,cross,prior


@pytest.mark.parametrize('axis',[0,2])
def test_folded_joint_prior_matches_full_cube_law(axis):
    from geospsr.spectral import FourierPoisson
    rng=np.random.default_rng(44)
    x=rng.uniform(-1,1,(4,3)).astype('f');q=rng.uniform(-1,1,(3,3)).astype('f')
    spectrum=HalfCubeSpectrum(5,lengthscale=.1)
    scalar=FourierPoisson(2,lengthscale=.1)
    kernel=HalfCubeKernel(x,spectrum,0,rhs_count=3*256,axis=axis)
    random=cp.random.RandomState(18)
    draws=[]
    for _ in range(16):
        v,f=kernel.joint_prior(random,scalar)
        v=cp.asnumpy(v).reshape(len(x),256,3).transpose(1,0,2).reshape(256,-1)
        draws.append(np.concatenate([v,cp.asnumpy(scalar.evaluate(q,f))],axis=1))
    draws=np.concatenate(draws)
    k,c,p=explicit_joint(x.astype('d'),q.astype('d'),5,2,.1)
    c=c.reshape(len(q),-1)
    cov=np.block([[np.kron(k,np.eye(3)),c.T],[c,p]])
    se=np.sqrt((np.diag(cov)[:,None]*np.diag(cov)[None]+cov**2)/(len(draws)-1))
    assert np.max(np.abs(np.cov(draws.T)-cov)/se)<5
    assert np.max(np.abs(draws.mean(0))/np.sqrt(np.diag(cov)/len(draws)))<5


def test_half_posterior_moments_partial_batch_and_nonhermitian_cache():
    from reproduction.half_posterior import HalfCubePosterior
    rng=np.random.default_rng(59)
    x=rng.uniform(-1,1,(11,3)).astype('f');q=rng.uniform(-1,1,(4,3)).astype('f')
    y=rng.normal(size=x.shape).astype('f')
    noise=.0002
    model=HalfCubePosterior(x,y,training_cutoff=5,cutoff=2,lengthscale=.1,noise_variance=noise,axis=1)
    k,c,prior=explicit_joint(x.astype('d'),q.astype('d'),5,2,.1)
    _,source,_=explicit_joint(x.astype('d'),x.astype('d'),5,2,.1)
    k+=noise*np.eye(len(x));alpha=np.linalg.solve(k,y)
    offset=np.einsum('qnd,nd->q',source,alpha).mean()
    mean=np.einsum('qnd,nd->q',c,alpha)-offset
    covariance=prior.copy()
    for d in range(3):
        covariance-=c[:,:,d]@np.linalg.solve(k,c[:,:,d].T)
    np.testing.assert_allclose(cp.asnumpy(model.mean(q)),mean,rtol=3e-4,atol=1e-8)
    model.draw(2051,seed=91,batch=128)
    samples=cp.asnumpy(model.samples(q,batch=128))
    assert np.max(np.abs(samples.mean(0)-mean)/np.sqrt(np.diag(covariance)/len(samples)))<5
    se=np.sqrt((np.diag(covariance)[:,None]*np.diag(covariance)[None]+covariance**2)/(len(samples)-1))
    assert np.max(np.abs(np.cov(samples.T)-covariance)/se)<5
    model.draw(3,seed=4,batch=2,retain_components=True,retain_rhs=True,initial_mean=True)
    rhs=cp.asnumpy(model.sample_rhs).transpose(1,0,2).reshape(len(x),-1)
    actual=cp.asnumpy(model.sample_weights).transpose(1,0,2).reshape(len(x),-1)
    assert np.linalg.norm(actual-np.linalg.solve(k,rhs))/np.linalg.norm(actual)<3e-5
    expected=model.samples(q).copy()
    model.prepare_query_cache(batch=2)
    assert float(cp.linalg.norm(model.samples(q)-expected)/cp.linalg.norm(expected))<2e-5
    saved_rhs=model.sample_rhs.copy();saved_prior=model.sample_prior_coeff.copy()
    model.draw(3,seed=4,batch=2,retain_components=True,retain_rhs=True,solve_cutoff=3)
    np.testing.assert_array_equal(cp.asnumpy(saved_rhs),cp.asnumpy(model.sample_rhs))
    np.testing.assert_array_equal(cp.asnumpy(saved_prior),cp.asnumpy(model.sample_prior_coeff))
    narrow,_,_=explicit_joint(x.astype('d'),q.astype('d'),3,2,.1)
    narrow+=noise*np.eye(len(x))
    exact=np.linalg.solve(narrow,rhs-np.tile(y,(1,3)))+np.tile(cp.asnumpy(model.alpha),(1,3))
    actual=cp.asnumpy(model.sample_weights).transpose(1,0,2).reshape(len(x),-1)
    assert np.linalg.norm(actual-exact)/np.linalg.norm(exact)<3e-5
    assert all(r['centered_fluctuations'] and r['solve_cutoff']==3 for r in model.sample_solve_records)
    with pytest.raises(ValueError,match='zero start'):
        model.draw(1,solve_cutoff=3,initial_mean=True)
