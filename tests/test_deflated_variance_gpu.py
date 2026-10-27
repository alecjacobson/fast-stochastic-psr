import cupy as cp
import cupyx.scipy.linalg as cla
import numpy as np
import pytest
from geospsr.spectral import FourierPoisson
from reproduction.deflated_variance import FourierCorrectionCovariance,DeflatedCovarianceMoments


@pytest.mark.parametrize('dtype,custom',[('float64',False),('float32',False),('float64',True)])
def test_nufft_covariance_action_against_explicit_cross_sum_and_dense_inverse(dtype,custom):
    rng=np.random.default_rng(901);x=rng.uniform(-2,2,(11,3));q=rng.uniform(-2,2,(9,3))
    x[0]=[np.pi-1e-5,0,0];q[0]=[-np.pi+1e-5,0,0]
    x=x.astype(dtype);q=q.astype(dtype)
    f=rng.normal(size=(len(x),len(x)));a=f@f.T+.3*np.eye(len(x))
    op=FourierPoisson(3,lengthscale=.2,dtype=dtype,eps=1e-12 if dtype=='float64' else 1e-6)
    coeff=(-1j*op.multiplier).get()
    if custom:coeff=rng.normal(size=coeff.shape)+1j*rng.normal(size=coeff.shape)
    phase=np.exp(1j*((q[:,None].astype('d')-x[None].astype('d'))@op.ns.get().reshape(3,-1)))
    c=np.real(np.einsum('qnf,df->qnd',phase,coeff.reshape(3,-1)))
    b=sum(c[:,:,d]@np.linalg.solve(a,c[:,:,d].T) for d in range(3))
    chol=cp.asarray(np.linalg.cholesky(a),dtype=dtype)
    solve=lambda rhs:cla.solve_triangular(chol.T,cla.solve_triangular(chol,rhs,lower=True),lower=False)
    action=FourierCorrectionCovariance(x,q,op,solve,batch=2,coefficients=coeff)
    basis=np.linalg.qr(rng.normal(size=(len(q),5)))[0]
    actual=action(cp.asarray(basis)).get()
    tolerance=5e-5 if dtype=='float32' else 1e-9
    assert np.linalg.norm(actual-b@basis)/np.linalg.norm(b@basis)<tolerance
    assert set(action.plans)=={1,2}  # Partial final block.
    matrix=action(cp.eye(len(q),dtype=dtype)).get()
    assert np.linalg.norm(matrix-matrix.T)/np.linalg.norm(matrix)<tolerance
    assert np.linalg.eigvalsh((matrix+matrix.T)/2).min()>-tolerance*np.linalg.norm(matrix)
    state=DeflatedCovarianceMoments(cp.asarray(basis),cp.asarray(actual),xp=cp)
    samples=rng.normal(size=(17,len(q)))
    state.update(cp.asarray(samples));cpu=DeflatedCovarianceMoments(basis,actual);cpu.update(samples)
    np.testing.assert_allclose(state.diagonal().get(),cpu.diagonal(),rtol=1e-12,atol=1e-10)
    with pytest.raises(ValueError):action(cp.zeros((len(q),1),dtype='complex64'))
    assert action(cp.empty((len(q),0))).shape==(len(q),0)
