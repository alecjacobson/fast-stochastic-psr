import numpy as np
import cupy as cp
import pytest
from geospsr.spectral import FourierPoisson
from reproduction.factor_deflation import FourierCorrectionFactor,FactorDeflatedMoments
from reproduction.correction_sampling import DenseCorrectionSampler


@pytest.mark.parametrize('dtype,custom',[('float64',False),('float32',False),('float64',True)])
def test_factor_actions_against_explicit_fourier_and_whitening(dtype,custom):
    rng=np.random.default_rng(197);n=11;m=9
    x=rng.uniform(-2,2,(n,3)).astype(dtype);q=rng.uniform(-2,2,(m,3)).astype(dtype)
    x[0]=[np.pi-1e-5,0,0];q[0]=[-np.pi+1e-5,0,0]
    a=rng.normal(size=(n,n));chol=np.linalg.cholesky(a@a.T+.3*np.eye(n)).astype(dtype)
    op=FourierPoisson(3,lengthscale=.2,dtype=dtype,eps=1e-12 if dtype=='float64' else 1e-6)
    coeff=(-1j*op.multiplier).get()
    if custom:coeff=rng.normal(size=coeff.shape)+1j*rng.normal(size=coeff.shape)
    phase=np.exp(1j*((q[:,None].astype('d')-x[None].astype('d'))@op.ns.get().reshape(3,-1)))
    cross=np.real(np.einsum('qnf,df->qnd',phase,coeff.reshape(3,-1)))
    # H columns ordered (observation,component); each component has its own L.
    h=np.stack([np.linalg.solve(chol.astype('d'),cross[:,:,d].T).T for d in range(3)],axis=2).reshape(m,3*n)
    factor=FourierCorrectionFactor(x,q,op,cp.asarray(chol),batch=2,coefficients=coeff)
    z=rng.normal(size=(3*n,5));w=rng.normal(size=(m,5))
    hz=factor.forward(cp.asarray(z)).get();htw=factor.adjoint(cp.asarray(w)).get()
    tol=5e-5 if dtype=='float32' else 1e-9
    assert np.linalg.norm(hz-h@z)/np.linalg.norm(h@z)<tol
    assert np.linalg.norm(htw-h.T@w)/np.linalg.norm(h.T@w)<tol
    assert abs(np.sum(hz*w)-np.sum(z*htw))<tol*np.linalg.norm(hz)*np.linalg.norm(w)
    assert set(factor.plans)=={1,2}
    v=np.linalg.qr(h.T@rng.normal(size=(m,4)))[0];u=factor.forward(cp.asarray(v)).get()
    gpu=FactorDeflatedMoments(cp.asarray(v),cp.asarray(u),xp=cp);cpu=FactorDeflatedMoments(v,u)
    gpu.update(cp.asarray(hz.T),cp.asarray(z));cpu.update(hz.T,z)
    np.testing.assert_allclose(gpu.diagonal().get(),cpu.diagonal(),rtol=1e-12,atol=1e-10)
    assert factor.forward(cp.empty((3*n,0))).shape==(m,0)
    assert factor.adjoint(cp.empty((m,0))).shape==(3*n,0)
    with pytest.raises(ValueError):factor.forward(cp.ones((3*n,1),dtype='complex64'))


def test_retained_normals_preserve_draws_and_match_factor_layout():
    rng=np.random.default_rng(22);n=7;q=rng.normal(size=(6,3));x=rng.normal(size=(n,3))
    a=rng.normal(size=(n,n));chol=cp.asarray(np.linalg.cholesky(a@a.T+.5*np.eye(n)))
    op=FourierPoisson(3,lengthscale=.2,dtype='float64',eps=1e-12)
    sampler=DenseCorrectionSampler(x,chol,op);factor=FourierCorrectionFactor(x,q,op,chol,batch=2)
    plain=list(sampler.draw_batches(5,seed=59,batch=3,retain_weights=True))
    paired=list(sampler.draw_batches(5,seed=59,batch=3,retain_weights=True,retain_standard_normal=True))
    for old,new in zip(plain,paired):
        np.testing.assert_array_equal(old['weights'].get(),new['weights'].get())
        # Parallel NUFFT reductions need not be bitwise deterministic.
        a,b=old['coefficients'].get(),new['coefficients'].get()
        assert np.linalg.norm(a-b)<=1e-13*np.linalg.norm(a)
        actual=op.evaluate(q,new['coefficients']).get()
        expected=factor.forward(new['standard_normal']).get().T
        np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-11)
        assert 'standard_normal' not in old
