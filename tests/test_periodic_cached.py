import numpy as np
import pytest
from geospsr.periodic_cached import CachedPeriodicRangeKernel,_SERIAL_SOURCE,_SCAN_SOURCE
from test_periodic_range import dense


@pytest.mark.parametrize('scan',[False,True])
@pytest.mark.parametrize('ell',[.001,.3,3.])
def test_cached_coefficients(scan,ell):
    rng=np.random.default_rng(91);x=rng.uniform(-7,7,(257,3));x[:70]=x[0];x[70:90,2]=x[0,2]
    x[90:110]=rng.choice([-2*np.pi,0,2*np.pi],(20,3))+rng.normal(0,1e-9,(20,3))
    w=rng.normal(size=(len(x),3))*[1e-5,1,1e5]
    op=CachedPeriodicRangeKernel(x,ell,.7,0,leaf_size=8,scan=scan)
    actual=op(w).get();expected=dense(x,ell,.7)@w
    for j in range(3):assert np.linalg.norm(actual[:,j]-expected[:,j])/np.linalg.norm(expected[:,j])<1e-7
    assert op.record['cache_bytes']==op.coeff.nbytes+op.decays.nbytes


def test_cache_cap_and_no_exp_in_action():
    with pytest.raises(MemoryError):CachedPeriodicRangeKernel(np.random.default_rng(1).normal(size=(50,3)),max_cache_bytes=1)
    # Specialization must remove exponentials from the far action, not its unused helper.
    for source in [_SERIAL_SOURCE,_SCAN_SOURCE]:assert 'exp(' not in source.split('extern "C"')[1]
