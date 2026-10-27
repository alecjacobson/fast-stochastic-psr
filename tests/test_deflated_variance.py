import numpy as np
import pytest
from reproduction.deflated_variance import DeflatedCovarianceMoments


def test_fixed_projection_identity_unbiased_residual_and_full_rank_limit():
    rng=np.random.default_rng(91);d=7
    root=rng.normal(size=(d,d));b=root@root.T+.2*np.eye(d)
    q=np.linalg.qr(rng.normal(size=(d,d)))[0]
    fields=rng.normal(size=(32768,d))@np.linalg.cholesky(b).T
    for rank in [0,3,d]:
        state=DeflatedCovarianceMoments(q[:,:rank],b@q[:,:rank])
        proj=q[:,:rank]@q[:,:rank].T;residual=(np.eye(d)-proj)@b@(np.eye(d)-proj)
        np.testing.assert_allclose(state.deterministic+np.diag(residual),np.diag(b),atol=1e-12)
        for batch in np.array_split(fields,13):state.update(batch)
        se=np.sqrt(2/len(fields))*np.diag(residual)
        assert np.all(np.abs(state.diagonal()-np.diag(b))<6*se+1e-12)
        if rank==d:np.testing.assert_allclose(state.diagonal(),np.diag(b),atol=2e-12)
        if rank==0:np.testing.assert_allclose(state.diagonal(),(fields*fields).mean(0),atol=1e-12)


def test_negative_control_estimates_are_retained_and_bad_inputs_rejected():
    q=np.ones((2,1))/np.sqrt(2);b=np.diag([1.,100.])
    state=DeflatedCovarianceMoments(q,b@q)
    assert state.deterministic[0]<0
    with pytest.raises(ValueError,match='no evaluation'):state.diagonal()
    state.update(np.zeros((3,2)));assert state.diagonal()[0]<0
    for values in [np.ones((1,3)),np.empty((0,2)),np.full((1,2),np.nan),np.ones((1,2),dtype=complex)]:
        with pytest.raises(ValueError):state.update(values)
    assert state.count==3
    for basis,action in [(q*2,b@q),(q,np.zeros((2,2))),(q.astype(complex),b@q),(q,np.full((2,1),np.inf))]:
        with pytest.raises(ValueError):DeflatedCovarianceMoments(basis,action)
