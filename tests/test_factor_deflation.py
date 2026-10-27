import numpy as np
import pytest
from reproduction.factor_deflation import FactorDeflatedMoments


def test_factor_projection_identity_and_paired_residual_moments():
    rng=np.random.default_rng(457);h=rng.normal(size=(8,12));v=np.linalg.qr(rng.normal(size=(12,12)))[0]
    xi=rng.normal(size=(12,32768));fields=(h@xi).T
    for rank in [0,5,12]:
        basis=v[:,:rank];u=h@basis;state=FactorDeflatedMoments(basis,u)
        residual=h@(np.eye(12)-basis@basis.T)
        expected=np.sum(residual**2,axis=1)
        np.testing.assert_allclose(state.deterministic+expected,np.sum(h*h,axis=1),atol=1e-12)
        for start in range(0,len(fields),1024):state.update(fields[start:start+1024],xi[:,start:start+1024])
        se=np.sqrt(2/len(fields))*expected
        assert np.all(np.abs(state.diagonal()-np.sum(h*h,axis=1))<6*se+1e-12)
        if rank==0:np.testing.assert_allclose(state.diagonal(),np.mean(fields**2,axis=0),atol=1e-12)
        if rank==12:
            np.testing.assert_allclose(state.diagonal(),np.sum(h*h,axis=1),atol=1e-12)
            wrong=FactorDeflatedMoments(basis,u);wrong.update(fields,xi[:,::-1])
            assert np.linalg.norm(wrong.diagonal()-state.diagonal())>1  # pairing is essential
    assert np.all(.001-state.diagonal()<0)  # no posterior positivity clipping


def test_factor_projection_rejects_invalid_inputs_without_accumulation():
    v=np.eye(4)[:,:2];u=np.ones((3,2));state=FactorDeflatedMoments(v,u)
    with pytest.raises(ValueError):state.diagonal()
    for fields,xi in [(np.ones((2,4)),np.ones((4,2))), (np.ones((2,3)),np.ones((3,2))),
                      (np.ones((2,3),complex),np.ones((4,2))), (np.ones((2,3)),np.full((4,2),np.nan))]:
        with pytest.raises(ValueError):state.update(fields,xi)
    assert state.count==0
    for basis,action in [(2*v,u),(v,np.ones((3,3))),(v.astype(complex),u),(v,np.full((3,2),np.inf))]:
        with pytest.raises(ValueError):FactorDeflatedMoments(basis,action)
