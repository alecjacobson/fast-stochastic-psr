import numpy as np
import pytest
from geospsr.periodic_hybrid import HybridPeriodicRangeKernel
from geospsr.native_range import NativeCompactRangeKernel
from test_periodic_range import dense


@pytest.mark.parametrize('ell',[.001,.01,.3,3.])
@pytest.mark.parametrize('case',['random','ties','seams'])
def test_hybrid(ell,case):
    rng=np.random.default_rng(113);x=rng.uniform(-7,7,(337,3))
    if case=='ties':x[:120]=x[0];x[120:200,2]=x[0,2]
    if case=='seams':x[:120]=rng.choice([-2*np.pi,0,2*np.pi],(120,3))+rng.normal(0,1e-9,(120,3))
    w=rng.normal(size=(len(x),4))*[0,1e-5,1,1e5];op=HybridPeriodicRangeKernel(x,ell,.7,0,leaf_size=8)
    y=op(w).get();ref=dense(x,ell,.7)@w
    np.testing.assert_array_equal(y[:,0],np.zeros(len(x)))
    for j in [1,2,3]:assert np.linalg.norm(y[:,j]-ref[:,j])/np.linalg.norm(ref[:,j])<1e-7
    assert op.record['small_groups'] and op.record['large_groups']
    np.testing.assert_array_equal(np.sort(np.r_[op.small.get(),op.large.get()]),np.arange(op.record['far_groups']))


def test_hybrid_large_and_singleton():
    rng=np.random.default_rng(127);x=rng.uniform(-5,5,(2051,3));w=rng.normal(size=(len(x),3))
    op=HybridPeriodicRangeKernel(x,.3,.7,.02);ref=NativeCompactRangeKernel(x,.3,.7,.02)
    np.testing.assert_allclose(op(w).get(),ref(w).get(),rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(HybridPeriodicRangeKernel(x[:1],variance=.7)([2]).get(),[2*(.7**3+.02)])
