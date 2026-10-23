import cupy as cp
import numpy as np
import pytest
from geospsr.nystrom import LandmarkNystrom
from geospsr.spectral import spectral_axis
from geospsr.training_spectrum import FourierCovariance
from geospsr.packed_spectral import PackedFourierKernel


@pytest.mark.parametrize('options',[{}, {'recompute_features':True,'apply_row_batch':7},
    {'recompute_features':True,'basis_dtype':'float64','apply_row_batch':7},
    {'recompute_features':True,'basis_dtype':'float16','apply_row_batch':7}])
@pytest.mark.parametrize('dtype',['float32','float64'])
def test_nystrom_against_independent_long_fourier_covariance(dtype,options):
    rng=np.random.default_rng(11);x=rng.uniform(-1,1,(31,3)).astype(dtype)
    x[:3]=[[np.pi-1e-5,0,0],[-np.pi+1e-5,0,0],[np.pi-1e-5,0,0]]
    pre=LandmarkNystrom(x,.002,rank=15,lengthscale=.7,dtype=dtype,row_batch=9,**options)
    modes,axis=spectral_axis(1500,.7,.1);matrix=np.ones((len(x),len(x)))
    for d in range(3):matrix*=np.cos((x[:,None,d].astype('d')-x[None,:,d].astype('d'))[...,None]*modes)@axis
    ids=pre.landmark_ids;c=matrix[:,ids];w=matrix[np.ix_(ids,ids)]
    approx=c@np.linalg.pinv(w,rcond=1e-10)@c.T
    ev,u=np.linalg.eigh((approx+approx.T)/2);keep=ev>ev[-1]*1e-10;ev,u=ev[keep],u[:,keep]
    expected=np.eye(len(x))-(u*(1-(ev[0]+.002)/(ev+.002)))@u.T
    actual=cp.asnumpy(pre(cp.eye(len(x))))
    limit=5e-4 if options.get('basis_dtype')=='float16' else (3e-6 if dtype=='float32' else 2e-7)
    assert np.linalg.norm(actual-expected)<limit
    assert np.linalg.norm(actual-actual.T)<(3e-7 if dtype=='float32' else 1e-14)
    assert np.linalg.eigvalsh((actual+actual.T)/2).min()>0
    assert pre.record['spd_margin_guard']>0
    decoded=cp.asnumpy(pre.basis).astype(dtype).astype('d')/pre.basis_scale
    independently_measured=np.linalg.norm(decoded.T@decoded-np.eye(pre.rank))
    assert np.isclose(pre.record['orthogonality_frobenius_error'],independently_measured,rtol=1e-5,atol=1e-12)


def test_nystrom_duplicate_landmarks_and_validation():
    x=cp.zeros((17,3));x[8:,0]=.1
    pre=LandmarkNystrom(x,.002,rank=17,lengthscale=.5,row_batch=7)
    assert pre.record['landmark_rank']==2 and pre.rank==2
    np.testing.assert_array_equal(cp.asnumpy(pre(cp.zeros((17,3)))),0.)
    with pytest.raises(ValueError,match='scalar'):LandmarkNystrom(x,cp.ones(17),rank=4)
    with pytest.raises(ValueError,match='exceeds'):LandmarkNystrom(x,.02,rank=18)
    with pytest.raises(ValueError,match='basis dtype'):LandmarkNystrom(x,.02,rank=4,basis_dtype='int8')
    with pytest.raises(ValueError,match='row batch'):LandmarkNystrom(x,.02,rank=4,apply_row_batch=0)


@pytest.mark.parametrize('basis_dtype',['float64','float16'])
def test_nystrom_pcg_keeps_original_finite_covariance(basis_dtype):
    rng=np.random.default_rng(71);x=rng.normal(0,.1,(47,3)).astype('f')
    b=rng.normal(size=(47,6)).astype('f');b[:,3]=0
    op=FourierCovariance(9,lengthscale=.15,dtype='float64',eps=1e-11,upsampfac=2.)
    action=PackedFourierKernel(x,op,.0001,6)
    pre=LandmarkNystrom(x,.0001,rank=19,lengthscale=.15,dtype='float64',row_batch=13,
                        basis_dtype=basis_dtype,recompute_features=True,apply_row_batch=11)
    result,info=action.solve(b,preconditioner=pre,rtol=1e-9)
    modes,axis=spectral_axis(9,.15,.1);matrix=np.ones((len(x),len(x)))
    for d in range(3):matrix*=np.cos((x[:,None,d].astype('d')-x[None,:,d].astype('d'))[...,None]*modes)@axis
    matrix+=.0001*np.eye(len(x))
    assert info['converged']
    assert np.linalg.norm(cp.asnumpy(result)-np.linalg.solve(matrix,b))/np.linalg.norm(result.get())<2e-8
