import cupy as cp
import numpy as np
import pytest
from geospsr.spectral import FourierPoisson
from geospsr.packed_sources import PackedRealSourcePlan
from reproduction.factor_deflation import FourierCorrectionFactor
from reproduction.correction_sampling import DenseCorrectionSampler


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_packed_real_sources_against_explicit_sum_with_unequal_zero_odd_channels(dtype):
    rng=np.random.default_rng(17);x=rng.uniform(-np.pi,np.pi,(13,3)).astype(dtype)
    op=FourierPoisson(3,dtype=dtype,eps=1e-6 if dtype=='float32' else 1e-12)
    weights=(rng.normal(size=(5,13))*np.array([0,1e-10,1,1e10,2])[:,None]).astype(dtype)
    plan=PackedRealSourcePlan(op,x,5);actual=plan.execute(cp.asarray(weights)).get().reshape(5,-1)
    expected=weights.astype('d')@np.exp(-1j*x.astype('d')@op.ns.get().reshape(3,-1).astype('d'))
    tol=2e-5 if dtype=='float32' else 1e-10
    assert np.count_nonzero(actual[0])==0
    for i in range(1,5):assert np.linalg.norm(actual[i]-expected[i])/np.linalg.norm(expected[i])<tol
    with pytest.raises(ValueError):plan.execute(cp.ones((5,13),dtype=op.cdtype)*(1+1j))
    with pytest.raises(ValueError):plan.execute(cp.ones((4,13)))


def test_packed_factor_and_sampler_preserve_paired_fields():
    rng=np.random.default_rng(71);n=9;x=rng.normal(size=(n,3));q=rng.normal(size=(8,3));a=rng.normal(size=(n,n))
    chol=cp.asarray(np.linalg.cholesky(a@a.T+.3*np.eye(n)));op=FourierPoisson(3,dtype='float64',eps=1e-12)
    factory=lambda kind,points,count:PackedRealSourcePlan(op,points,count)
    native=FourierCorrectionFactor(x,q,op,chol,batch=3)
    packed=FourierCorrectionFactor(x,q,op,chol,batch=3,source_plan_factory=factory)
    z=cp.asarray(rng.normal(size=(3*n,5)))
    np.testing.assert_allclose(packed.forward(z).get(),native.forward(z).get(),rtol=1e-9,atol=1e-11)
    a=DenseCorrectionSampler(x,chol,op);b=DenseCorrectionSampler(x,chol,op,source_plan_factory=factory)
    for raw,new in zip(a.draw_batches(5,seed=71,batch=3,retain_standard_normal=True),b.draw_batches(5,seed=71,batch=3,retain_standard_normal=True)):
        np.testing.assert_array_equal(raw['standard_normal'].get(),new['standard_normal'].get())
        np.testing.assert_allclose(op.evaluate(q,new['coefficients']).get(),native.forward(new['standard_normal']).get().T,rtol=1e-9,atol=1e-11)
