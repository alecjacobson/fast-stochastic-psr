import numpy as np
import pytest
from geospsr.periodic_leaf_cache import LeafCachedRangeKernel
from test_periodic_range import dense


@pytest.mark.parametrize('leaf',[8,32,128])
@pytest.mark.parametrize('ell',[.001,.01,3.])
def test_cached_leaves(leaf,ell):
    rng=np.random.default_rng(131);x=rng.uniform(-7,7,(337,3));x[:65]=x[0];x[65:100,2]=x[0,2]
    x[100:130]=rng.choice([-2*np.pi,0,2*np.pi],(30,3))+rng.normal(0,1e-9,(30,3))
    w=rng.normal(size=(len(x),3))*[1e-5,1,1e5];op=LeafCachedRangeKernel(x,ell,.7,0,leaf_size=leaf)
    actual=op(w).get();ref=dense(x,ell,.7)@w
    for j in range(3):assert np.linalg.norm(actual[:,j]-ref[:,j])/np.linalg.norm(ref[:,j])<1e-7
    cover=np.zeros((len(x),len(x)),int)
    for groups,isfar in [(op.far,True),(op.direct,False)]:
        ids,labels,offsets=[a.get() for a in groups[:3]];full=np.zeros(len(offsets)-1,bool) if isfar else groups[3].get()
        for g,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
            ii=ids[a:b];ll=labels[a:b];cover[np.ix_(ii,ii)]+=np.ones((len(ii),len(ii)),int) if full[g] else ll[:,None]!=ll[None,:]
    np.testing.assert_array_equal(cover,np.ones_like(cover))
    assert op.record['leaf_cache_bytes']==op.matrices.nbytes


def test_leaf_cache_cap_and_singleton():
    x=np.random.default_rng(2).normal(size=(17,3))
    with pytest.raises(MemoryError):LeafCachedRangeKernel(x,max_cache_bytes=1)
    np.testing.assert_allclose(LeafCachedRangeKernel(x[:1],variance=.7)([2]).get(),[2*(.7**3+.02)])
