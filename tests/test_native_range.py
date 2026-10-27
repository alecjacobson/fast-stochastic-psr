import numpy as np
import pytest
from geospsr.native_range import NativeRangeKernel,NativeCompactRangeKernel
from geospsr.periodic_range import PeriodicRangeKernel


@pytest.mark.parametrize('case',['random','ties','seams'])
@pytest.mark.parametrize('n',[1,17,311])
def test_native_plan(case,n):
    rng=np.random.default_rng(107);x=rng.uniform(-9,9,(n,3))
    if case=='ties':x[:n//3]=x[0];x[n//3:2*n//3,2]=x[0,2]
    if case=='seams':x[:n//3]=rng.choice([-2*np.pi,0,2*np.pi],(n//3,3))+rng.normal(0,1e-9,(n//3,3))
    ref=PeriodicRangeKernel(x,leaf_size=8);op=NativeCompactRangeKernel(x,leaf_size=8)
    for a,b in zip((ref.x,*ref.far,*ref.direct,ref.anchors),(op.x,*op.far,*op.direct,op.anchors)):
        np.testing.assert_array_equal(a.get(),b.get())
    w=rng.normal(size=(n,3));np.testing.assert_allclose(op(w).get(),ref(w).get(),rtol=1e-9,atol=1e-12)
    with pytest.raises(ValueError if n==1 else MemoryError):NativeRangeKernel(x,max_entries=n-1)
