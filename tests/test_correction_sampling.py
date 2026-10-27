import cupy as cp
import numpy as np
import pytest
from geospsr.spectral import FourierPoisson, spectral_axis
from reproduction.correction_sampling import DenseCorrectionSampler, CorrectionMoments


def test_correction_draws_match_dense_covariance_and_direct_cross_sum():
    rng = np.random.default_rng(987)
    x=rng.uniform(-1,1,(9,3)); q=rng.uniform(-1,1,(5,3))
    matrix=rng.normal(size=(9,9)); matrix=matrix@matrix.T+.2*np.eye(9)
    op=FourierPoisson(2,lengthscale=.2,dtype='float64',eps=1e-12)
    sampler=DenseCorrectionSampler(x,cp.asarray(np.linalg.cholesky(matrix)),op)
    modes,axis=spectral_axis(2,.2)
    ns=np.stack(np.meshgrid(modes,modes,modes,indexing='ij'),-1).reshape(-1,3)
    rho=(axis[:,None,None]*axis[None,:,None]*axis[None,None,:]).ravel()
    norm=(ns*ns).sum(1); keep=norm>0
    cross=np.einsum('qnf,fd->qnd',np.sin((q[:,None]-x[None])@ns[keep].T),
                    rho[keep,None]*ns[keep]/norm[keep,None])
    expected=sum(cross[:,:,d]@np.linalg.solve(matrix,cross[:,:,d].T) for d in range(3))
    blocks=[]
    for block in sampler.draw_batches(8193,seed=56,batch=128,retain_weights=True):
        actual=op.evaluate(q,block['coefficients']).get()
        direct=np.einsum('qnd,snd->sq',cross,block['weights'].get())
        np.testing.assert_allclose(actual,direct,rtol=2e-10,atol=1e-14)
        blocks.append(actual)
    draws=np.concatenate(blocks)
    se=np.sqrt((np.diag(expected)[:,None]*np.diag(expected)[None]+expected**2)/len(draws))
    assert np.max(np.abs(draws.T@draws/len(draws)-expected)/se)<6
    assert len(blocks[-1])==1


def test_online_correction_moments_and_explicit_negative_estimates():
    rng=np.random.default_rng(67); values=rng.normal(size=(17,7))
    state=CorrectionMoments()
    for block in np.array_split(values,4):state.update(block)
    result=state.summary(.01,looks=4)
    np.testing.assert_allclose(result['variance'],.01-(values**2).mean(0),rtol=1e-14)
    assert (result['variance']<0).all()
    assert state.sum_squares.shape==(7,) and state.count==17
    with pytest.raises(ValueError):state.update(np.ones((2,8)))
    with pytest.raises(ValueError):state.update(np.full((2,7),np.nan))
    with pytest.raises(ValueError):state.summary(0)
    with pytest.raises(ValueError):state.summary(1,looks=0)
    assert state.count==17
