import cupy as cp
import numpy as np
import pytest
from geospsr.compact_basis import fused_half_apply


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('shape',[(9,4,3),(2051,67,11),(1048581,2,1)])
def test_fused_half_matches_independent_double_application(dtype,shape):
    n,rank,channels=shape;rng=np.random.default_rng(305)
    u=rng.normal(size=(n,rank))/np.sqrt(n)
    basis=(u*32).astype('float16');decoded=basis.astype('d')/32
    damping=np.linspace(0,.7,rank).astype(dtype)
    host=rng.normal(size=(n,2*channels)).astype(dtype)
    host[:,::2]*=np.geomspace(1e-5,1e5,channels).astype(dtype)
    if channels>1:host[:,2]=0
    rhs=cp.asarray(host)[:,::2]
    expected=host[:,::2].astype('d')-decoded@(damping[:,None]*(decoded.T@host[:,::2].astype('d')))
    result=fused_half_apply(cp.asarray(basis),cp.asarray(damping),rhs,32)
    errors=np.linalg.norm(result.get()-expected,axis=0)/np.maximum(np.linalg.norm(expected,axis=0),1e-30)
    assert errors.max()<(3e-6 if dtype=='float32' else 2e-13)
    if channels>1:np.testing.assert_array_equal(result.get()[:,1],0)


def test_fused_validation_and_empty_channels():
    u=cp.ones((7,3),dtype='float16');d=cp.zeros(3,dtype='f');b=cp.ones((7,2),dtype='f')
    np.testing.assert_array_equal(fused_half_apply(u,d,b,1).get(),b.get())
    assert fused_half_apply(u,d,b[:,:0],1).shape==(7,0)
    with pytest.raises(ValueError,match='half basis'):fused_half_apply(u.astype('f'),d,b,1)
    with pytest.raises(ValueError,match='RHS'):fused_half_apply(u,d,b.astype('F'),1)
    with pytest.raises(ValueError,match='damping'):fused_half_apply(u,d.astype('d'),b,1)
    with pytest.raises(ValueError,match='scale'):fused_half_apply(u,d,b,0)
