import numpy as np
import pytest
from geospsr.periodic_compact import CompactPeriodicRangeKernel
from geospsr.periodic_cached import CachedPeriodicRangeKernel
from test_periodic_range import dense


@pytest.mark.parametrize('ell',[.001,.01,.3,3.])
@pytest.mark.parametrize('case',['random','ties','seams'])
def test_compact(ell,case):
    rng=np.random.default_rng(97);x=rng.uniform(-np.pi,np.pi,(337,3))
    if case=='ties':x[:120]=x[0];x[120:200,2]=x[0,2]
    if case=='seams':x[:120]=rng.choice([-2*np.pi,0,2*np.pi],(120,3))+rng.normal(0,1e-9,(120,3))
    w=rng.normal(size=(len(x),4))*[0,1e-5,1,1e5]
    op=CompactPeriodicRangeKernel(x,ell,.7,0,leaf_size=8);actual=op(w).get();expected=dense(x,ell,.7)@w
    np.testing.assert_array_equal(actual[:,0],np.zeros(len(x)))
    for j in [1,2,3]:assert np.linalg.norm(actual[:,j]-expected[:,j])/np.linalg.norm(expected[:,j])<1e-7
    assert op.record['cache_bytes']==op.coeff.nbytes+op.decays.nbytes==152*op.record['far_entries']


def test_many_tiles_separable_cache_and_singleton():
    rng=np.random.default_rng(101);x=rng.uniform(-10,10,(2051,3));ell=.4
    w=rng.normal(size=(len(x),2));op=CompactPeriodicRangeKernel(x,ell,.7,.02)
    control=CachedPeriodicRangeKernel(x,ell,.7,.02,scan=True)
    np.testing.assert_allclose(op(w).get(),control(w).get(),rtol=1e-9,atol=1e-10)
    assert op.record['max_far_group_entries']>1024
    np.testing.assert_allclose(CompactPeriodicRangeKernel(x[:1],variance=.7)([2]).get(),[2*(.7**3+.02)])
    with pytest.raises(MemoryError):CompactPeriodicRangeKernel(x,max_cache_bytes=1)
