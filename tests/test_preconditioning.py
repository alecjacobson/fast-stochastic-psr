import cupy as cp
import numpy as np
import pytest
from geospsr.preconditioning import SpatialBlockJacobi,analytic_blocks,pcg
from geospsr.spectral import spectral_axis
from geospsr.training_spectrum import FourierCovariance
from geospsr.packed_spectral import PackedFourierKernel


@pytest.mark.parametrize('block_size',[1,8,37])
def test_block_inverse_against_independent_spectral_covariance(block_size):
    rng=np.random.default_rng(2026)
    x=rng.uniform(-np.pi,np.pi,(37,3))
    x[:3]=[[np.pi-1e-6,0,0],[-np.pi+1e-6,0,0],[3*np.pi,0,0]]
    noise=np.geomspace(.0001,.02,len(x))
    pre=SpatialBlockJacobi(x,noise,lengthscale=.8,block_size=block_size,setup_batch=2,dtype='float64')
    modes,axis=spectral_axis(1500,.8,.1)
    matrix=np.ones((len(x),len(x)))
    for d in range(3): matrix*=np.cos((x[:,None,d]-x[None,:,d])[...,None]*modes)@axis
    matrix+=np.diag(noise)
    order=cp.asnumpy(pre.order);r=rng.normal(size=(len(x),4));expected=np.empty_like(r)
    for start in range(0,len(x),block_size):
        ids=order[start:start+block_size]
        expected[ids]=np.linalg.solve(matrix[np.ix_(ids,ids)],r[ids])
    actual=cp.asnumpy(pre(r))
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<2e-8
    assert np.all((r*actual).sum(0)>0)


def test_pcg_preserves_full_finite_model_and_initial_state():
    rng=np.random.default_rng(180)
    x=rng.normal(0,.04,(43,3)).astype('f');rhs=rng.normal(size=(43,6)).astype('f');rhs[:,2]=0
    spectrum=FourierCovariance(8,lengthscale=.4,dtype='float64',eps=1e-11,upsampfac=2.)
    action=PackedFourierKernel(x,spectrum,.0001,6)
    pre=SpatialBlockJacobi(x,.0001,.4,block_size=16,dtype='float64')
    ns,axis=spectral_axis(8,.4,.1)
    matrix=np.ones((len(x),len(x)))
    for d in range(3): matrix*=np.cos((x[:,None,d].astype('d')-x[None,:,d])[...,None]*ns)@axis
    matrix+=.0001*np.eye(len(x))
    expected=np.linalg.solve(matrix,rhs)
    for initial in [None,.2*rhs]:
        saved=None if initial is None else initial.copy()
        weights,record=action.solve(rhs,rtol=1e-9,preconditioner=pre,x0=initial)
        assert record['converged'] and record['preconditioned']
        assert np.linalg.norm(cp.asnumpy(weights)-expected)/np.linalg.norm(expected)<2e-8
        if saved is not None: np.testing.assert_array_equal(saved,initial)
    zero,record=action.solve(cp.zeros_like(cp.asarray(rhs)),preconditioner=pre)
    assert record['iterations']==0 and record['converged']
    np.testing.assert_array_equal(cp.asnumpy(zero),0)
    with pytest.raises(RuntimeError,match='preconditioner curvature'):
        pcg(action,rhs,lambda v:-v)
