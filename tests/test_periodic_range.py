import cupy as cp
import numpy as np
import pytest
from geospsr.periodic_range import PeriodicRangeKernel


def dense(x,ell,variance):
    t=np.remainder(np.abs(x[:,None,:]-x[None,:,:])/(2*np.pi),1.)
    w=np.sqrt(3)*(t-.5)
    first=2*ell+np.sqrt(3)/np.tanh(np.sqrt(3)/(2*ell))
    norm=(1+np.exp(-np.sqrt(3)/ell))*(first-np.sqrt(3)*np.tanh(np.sqrt(3)/(2*ell)))
    k=(np.exp((w-np.sqrt(3)/2)/ell)+np.exp((-w-np.sqrt(3)/2)/ell))*(first-2*w*np.tanh(w/ell))/norm
    return variance**3*np.prod(k,axis=-1)


@pytest.mark.parametrize('ell',[.001,.01,.3,3.])
@pytest.mark.parametrize('case',['random','ties','seams'])
def test_full_action(ell,case):
    rng=np.random.default_rng(81);x=rng.uniform(-np.pi,np.pi,(97,3))
    if case=='ties':x[:25]=x[0];x[25:50,1:]=x[1,1:];x[50:75,0]=x[1,0]
    if case=='seams':x[:40]=rng.choice([-2*np.pi,0,2*np.pi],(40,3))+rng.normal(0,1e-9,(40,3))
    weights=rng.normal(size=(len(x),3))*[1e-5,1,1e5]
    op=PeriodicRangeKernel(x,ell,.7,.02,leaf_size=8)
    actual=op(weights).get();k=dense(x,ell,.7);expected=(k+.02*np.eye(len(x)))@weights
    for j in range(3):assert np.linalg.norm(actual[:,j]-expected[:,j])/np.linalg.norm(expected[:,j])<1e-7
    # Covariance alone matters too: noise must not conceal a poor kernel action.
    for j in range(3):assert np.linalg.norm(actual[:,j]-.02*weights[:,j]-k@weights[:,j])/np.linalg.norm(k@weights[:,j])<1e-7
    v=rng.normal(size=len(x));u=rng.normal(size=len(x))
    np.testing.assert_allclose(v@op(u).get(),u@op(v).get(),rtol=1e-9,atol=1e-11)
    assert v@op(v).get()>0
    # Explicit partition coverage, independent of moment evaluation.
    cover=np.zeros((len(x),len(x)),int)
    for groups,isfar in [(op.far,True),(op.direct,False)]:
        ids,labels,offsets=[a.get() for a in groups[:3]]
        full=np.zeros(len(offsets)-1,bool) if isfar else groups[3].get()
        for g,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
            ii=ids[a:b];ll=labels[a:b]
            cover[np.ix_(ii,ii)]+=np.ones((len(ii),len(ii)),int) if full[g] else (ll[:,None]!=ll[None,:])
    np.testing.assert_array_equal(cover,np.ones_like(cover))


def test_periodic_image_sum_and_singleton():
    rng=np.random.default_rng(73);x=rng.uniform(-5,5,(21,3));ell=.2;a=np.sqrt(3)/ell
    t=(x[:,None,:]-x[None,:,:])/(2*np.pi);k=np.zeros_like(t);norm=0.
    for image in range(-12,13):
        d=np.abs(t+image);k+=(1+a*d)*np.exp(-a*d)
        norm+=(1+a*abs(image))*np.exp(-a*abs(image))
    k=np.prod(k/norm,axis=-1)*.7**3
    op=PeriodicRangeKernel(x,ell,.7,0,leaf_size=4)
    np.testing.assert_allclose(op(np.eye(len(x))).get(),k,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(PeriodicRangeKernel(x[:1],variance=.7,noise_variance=.02)([2]).get(),[2*(.7**3+.02)])
    with pytest.raises(MemoryError):PeriodicRangeKernel(x,max_entries=2)
    with pytest.raises(ValueError):op(np.ones((21,1))*1j)
