import cupy as cp
import numpy as np
import pytest
from geospsr.periodic_scan import PeriodicScanKernel
from geospsr.periodic_range import PeriodicRangeKernel
from test_periodic_range import dense


@pytest.mark.parametrize('ell',[.001,.01,.3,3.])
@pytest.mark.parametrize('case',['random','ties','seams'])
def test_scan(ell,case):
    rng=np.random.default_rng(83);x=rng.uniform(-np.pi,np.pi,(193,3))
    if case=='ties':x[:80]=x[0];x[80:120,2]=x[0,2]
    if case=='seams':x[:80]=rng.choice([-2*np.pi,0,2*np.pi],(80,3))+rng.normal(0,1e-9,(80,3))
    w=rng.normal(size=(len(x),3))*[1e-5,1,1e5];w[:,0]=0
    op=PeriodicScanKernel(x,ell,.7,0,leaf_size=8)
    actual=op(w).get();expected=dense(x,ell,.7)@w
    np.testing.assert_array_equal(actual[:,0],np.zeros(len(x)))
    for j in [1,2]:assert np.linalg.norm(actual[:,j]-expected[:,j])/np.linalg.norm(expected[:,j])<1e-7
    serial=PeriodicRangeKernel(x,ell,.7,0,leaf_size=8)(w).get()
    np.testing.assert_allclose(actual,serial,rtol=1e-8,atol=1e-9)


def test_many_tiles_and_periodic_closure():
    rng=np.random.default_rng(87);x=rng.uniform(-10,10,(1103,3));ell=.3
    op=PeriodicScanKernel(x,ell,.7,.02)
    w=rng.normal(size=len(x));reference=(dense(x,ell,.7)+.02*np.eye(len(x)))@w
    np.testing.assert_allclose(op(w).get(),reference,rtol=1e-9,atol=1e-10)
    assert op.record['max_far_group_entries']>512
