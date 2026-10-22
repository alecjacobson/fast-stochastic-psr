"""Independent integer-frequency references for the shifted block backend."""
import cupy as cp
import numpy as np
import pytest

from geospsr.block_spectral import BlockSpectrum, BlockKernel
from geospsr.spectral import FourierPoisson
from geospsr.spectral_blocks import axis_weights


def explicit_model(x, q, core, extent, cutoff, ell, variance=.1):
    # Enumerate the enclosing cube, then select membership independently of
    # the production box partition or its frequency shifts.
    ns = np.stack(np.meshgrid(*[np.arange(-extent,extent+1)]*3,indexing='ij'),-1).reshape(-1,3)
    ns = ns[(np.abs(ns)>core).sum(1)<=1]
    rho = axis_weights(ns,ell,variance).prod(1)
    k = np.cos((x[:,None]-x[None])@ns.T)@rho
    low = (np.abs(ns)<=cutoff).all(1) & (np.square(ns).sum(1)>0)
    n,r = ns[low],rho[low]
    norm = np.square(n).sum(1)
    cross = np.einsum('qnf,fd->qnd',np.sin((q[:,None]-x[None])@n.T),r[:,None]*n/norm[:,None])
    prior = np.cos((q[:,None]-q[None])@n.T)@(r/norm)
    return k,cross,prior


@pytest.mark.parametrize('core,extent,ell,dtype,eps,upsamp',[
    (2,4,.04,'float64',1e-11,2.), # even slab width
    (2,5,.5,'float64',1e-11,2.),  # odd slab width, normalization matters
    (3,3,1.,'float64',1e-11,2.),  # core only
    (0,3,.1,'float64',1e-11,2.),  # zero-frequency core
    (2,5,.04,'float32',1e-6,1.25),
])
def test_shifted_actions_and_solves(core,extent,ell,dtype,eps,upsamp):
    rng = np.random.default_rng(133)
    x = rng.uniform(-np.pi,np.pi,(17,3)).astype(dtype)
    x[:3] = [[np.pi-1e-5,0,0],[-np.pi+1e-5,0,0],[np.pi-2e-5,0,0]]
    weights = rng.normal(size=(len(x),4)).astype(dtype)
    op = BlockSpectrum(core,extent,ell,eps=eps,dtype=dtype,upsampfac=upsamp)
    kernel = BlockKernel(x,op,.0002,rhs_count=4)
    expected,_,_ = explicit_model(x.astype('d'),x.astype('d'),core,extent,core,ell)
    from reproduction.block_reference import block_covariance_rows
    fixed = x.astype('f').astype('d')
    direct,_,_ = explicit_model(fixed,fixed,core,extent,core,ell)
    np.testing.assert_allclose(cp.asnumpy(block_covariance_rows(x,x,core,extent,ell)),direct,rtol=1e-11,atol=1e-17)
    assert np.linalg.eigvalsh(expected).min() > -1e-15
    expected += .0002*np.eye(len(x))
    tolerance = 3e-6 if dtype=='float32' else 2e-10
    actual = cp.asnumpy(kernel(weights))
    assert np.linalg.norm(actual-expected@weights)/np.linalg.norm(expected@weights) < tolerance
    rtol = 1e-5 if dtype=='float32' else 1e-9
    solution,info = kernel.solve(weights,rtol=rtol)
    assert info['converged']
    assert np.max(np.linalg.norm(expected@cp.asnumpy(solution)-weights,axis=0)/np.linalg.norm(weights,axis=0)) < 2*rtol
    with pytest.raises(ValueError,match='real RHS'):
        kernel(weights.astype('complex64'))
    with pytest.raises(ValueError,match='shape'):
        kernel(weights[:,:2])


def test_joint_prior_covariance_and_scalar_cross_coupling():
    rng = np.random.default_rng(231)
    x = rng.uniform(-1,1,(4,3)).astype('f')
    q = rng.uniform(-1,1,(3,3)).astype('f')
    ell,core,extent,cutoff = .1,2,5,2
    op = BlockSpectrum(core,extent,ell)
    scalar = FourierPoisson(cutoff,ell)
    kernel = BlockKernel(x,op,0,rhs_count=3*256)
    random = cp.random.RandomState(319)
    draws = []
    for _ in range(16):
        vector,coeff = kernel.joint_prior(random,scalar)
        vector = cp.asnumpy(vector).reshape(len(x),256,3).transpose(1,0,2).reshape(256,-1)
        draws.append(np.concatenate([vector,cp.asnumpy(scalar.evaluate(q,coeff))],axis=1))
    draws = np.concatenate(draws)
    k,cross,prior = explicit_model(x.astype('d'),q.astype('d'),core,extent,cutoff,ell)
    c = cross.reshape(len(q),-1)
    expected = np.block([[np.kron(k,np.eye(3)),c.T],[c,prior]])
    standard_error = np.sqrt((np.diag(expected)[:,None]*np.diag(expected)[None]+expected**2)/(len(draws)-1))
    assert np.max(np.abs(np.cov(draws.T)-expected)/standard_error) < 5
    assert np.max(np.abs(draws.mean(0))/np.sqrt(np.diag(expected)/len(draws))) < 5


def test_block_posterior_against_dense_finite_model_and_partial_batch():
    from reproduction.block_posterior import BlockPosterior
    rng = np.random.default_rng(312)
    x = rng.uniform(-1,1,(11,3)).astype('f')
    q = rng.uniform(-1,1,(4,3)).astype('f')
    y = rng.normal(size=x.shape).astype('f')
    ell,core,extent,cutoff,noise = .1,2,5,2,.0002
    model = BlockPosterior(x,y,core=core,extent=extent,cutoff=cutoff,lengthscale=ell,noise_variance=noise)
    k,cross,prior = explicit_model(x.astype('d'),q.astype('d'),core,extent,cutoff,ell)
    _,source_cross,_ = explicit_model(x.astype('d'),x.astype('d'),core,extent,cutoff,ell)
    k += noise*np.eye(len(x))
    alpha = np.linalg.solve(k,y)
    offset = np.einsum('qnd,nd->q',source_cross,alpha).mean()
    mean = np.einsum('qnd,nd->q',cross,alpha)-offset
    covariance = prior.copy()
    for d in range(3):
        covariance -= cross[:,:,d]@np.linalg.solve(k,cross[:,:,d].T)
    np.testing.assert_allclose(cp.asnumpy(model.mean(q)),mean,rtol=3e-4,atol=1e-8)
    model.draw(2051,seed=31,batch=128)
    samples = cp.asnumpy(model.samples(q,batch=128))
    assert np.max(np.abs(samples.mean(0)-mean)/np.sqrt(np.diag(covariance)/len(samples))) < 5
    se = np.sqrt((np.diag(covariance)[:,None]*np.diag(covariance)[None]+covariance**2)/(len(samples)-1))
    assert np.max(np.abs(np.cov(samples.T)-covariance)/se) < 5
    model.draw(3,seed=65,batch=2,retain_components=True,retain_rhs=True,initial_mean=True)
    expected = np.linalg.solve(k,cp.asnumpy(model.sample_rhs).transpose(1,0,2).reshape(len(x),-1))
    weights = cp.asnumpy(model.sample_weights).transpose(1,0,2).reshape(len(x),-1)
    assert np.linalg.norm(weights-expected)/np.linalg.norm(expected) < 3e-5
    assert model.sample_coeff.shape[0] == 3


def test_invalid_block_configuration():
    for core,extent in [(-1,4),(5,4),(1.5,4)]:
        with pytest.raises(ValueError):
            BlockSpectrum(core,extent)
    with pytest.raises(ValueError,match='dtype'):
        BlockSpectrum(2,4,dtype='float16')
    kernel = BlockKernel(np.zeros((2,3)),BlockSpectrum(2,4),.02)
    with pytest.raises(ValueError,match='scalar cutoff'):
        kernel.joint_prior(cp.random.RandomState(1),FourierPoisson(3))
    with pytest.raises(ValueError,match='kernel parameters'):
        kernel.joint_prior(cp.random.RandomState(1),FourierPoisson(2,lengthscale=.1))
