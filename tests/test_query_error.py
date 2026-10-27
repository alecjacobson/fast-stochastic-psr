import numpy as np
import pytest
from reproduction.query_error import residual_envelopes,relative_covariance_budget


@pytest.mark.parametrize('scale',[1e-2,1e-5])
def test_quadratic_identity_and_all_bounds_against_independent_dense_truth(scale):
    rng=np.random.default_rng(4901);n,m=23,7;noise=.02
    u=rng.normal(size=(n,n));a=u@u.T+noise*np.eye(n)
    xs=[rng.normal(size=(n,m))*1e-2 for _ in range(3)]
    ws=[np.linalg.solve(a+scale*np.eye(n),x) for x in xs]
    rs=[x-a@w for x,w in zip(xs,ws)]
    target=np.eye(m)*1e-6
    prior=target+sum(x.T@np.linalg.solve(a,x) for x in xs)
    raw=prior-sum(x.T@w for x,w in zip(xs,ws));raw=(raw+raw.T)*.5
    shift=sum(w.T@r for w,r in zip(ws,rs));shift=(shift+shift.T)*.5
    corrected=raw-shift;error=sum(r.T@np.linalg.solve(a,r) for r in rs)
    upper=sum(r.T@r for r in rs)/noise
    np.testing.assert_allclose(corrected-target,error,atol=2e-17,rtol=1e-5)
    assert np.linalg.eigvalsh(upper-error).min()>=-1e-18
    c=np.stack([np.linalg.norm(x,axis=0) for x in xs],axis=1)
    r=np.stack([np.linalg.norm(v,axis=0) for v in rs],axis=1)
    first,second=residual_envelopes(c,r,noise)
    assert np.all(np.abs(raw-target)<=first+1e-17)
    assert np.all(np.abs(corrected-target)<=second+2e-17)
    budget=relative_covariance_budget(corrected,upper)
    if budget['informative']:
        assert np.linalg.norm(error,2)/1e-6<=budget['upper_relative_to_exact_covariance']+1e-12


def test_singular_covariance_has_no_relative_certificate_and_bad_norms_fail():
    assert not relative_covariance_budget(np.diag([1.,0.]),np.eye(2)*1e-5)['informative']
    for value in [-1,np.nan]:
        with pytest.raises(ValueError):residual_envelopes(np.full((2,3),value),np.ones((2,3)),.02)
    with pytest.raises(ValueError):residual_envelopes(np.ones((2,3)),np.ones((2,3)),0)


@pytest.mark.parametrize('separation',[0.,1e-7])
def test_duplicate_and_near_duplicate_low_variance_queries(separation):
    rng=np.random.default_rng(4903);n,m=13,4;noise=.02
    z=rng.normal(size=(n,n));a=z@z.T+noise*np.eye(n)
    xs=[]
    for scale in [1e-2,1e-15,0.]:
        x=rng.normal(size=(n,m))*scale
        x[:,1]=x[:,0]+separation*rng.normal(size=n)*scale
        xs.append(x)
    ws=[np.linalg.solve(a+1e-3*np.eye(n),x) for x in xs]
    rs=[x-a@w for x,w in zip(xs,ws)]
    t=np.eye(m)*1e-6;t[1]=t[0]+separation*t[1]
    target=t@t.T
    prior=target+sum(x.T@np.linalg.solve(a,x) for x in xs)
    raw=prior-sum(x.T@w for x,w in zip(xs,ws));raw=(raw+raw.T)*.5
    shift=sum(w.T@r for w,r in zip(ws,rs));corrected=raw-(shift+shift.T)*.5
    error=sum(r.T@np.linalg.solve(a,r) for r in rs)
    upper=sum(r.T@r for r in rs)/noise
    np.testing.assert_allclose(corrected-target,error,atol=1e-17,rtol=1e-6)
    assert np.linalg.eigvalsh(upper-error).min()>=-1e-17
    assert not relative_covariance_budget(corrected,upper)['informative']


def test_gpu_component_layout_and_retained_residual_correction():
    import cupy as cp
    from reproduction.query_error import QueryResidualAudit
    rng=np.random.default_rng(4902);n,m=17,5;noise=.01
    u=rng.normal(size=(n,n));a=u@u.T+noise*np.eye(n)
    rhs=rng.normal(size=(n,3*m))*np.tile([1e-15,1e-2,0.],m)
    aw=np.linalg.solve(a+.001*np.eye(n),rhs);res=rhs-a@aw
    ad=cp.asarray(a);bd=cp.asarray(a+.001*np.eye(n))
    audit=QueryResidualAudit(lambda w:ad@w,lambda b:cp.linalg.solve(bd,b),noise)
    for start in range(0,3*m,6):audit(cp.asarray(rhs[:,start:start+6]))
    def product(left,right):return sum(left[:,d::3].T@right[:,d::3] for d in range(3))
    exact=np.linalg.solve(a,rhs);target=np.eye(m)*.001
    prior=target+product(rhs,exact);raw=prior-product(rhs,aw)
    corrected=audit.correct(cp.asarray(raw)).get()
    np.testing.assert_allclose(corrected-target,product(res,np.linalg.solve(a,res)),rtol=1e-6,atol=1e-15)
    np.testing.assert_allclose(audit.upper.get(),product(res,res)/noise,rtol=1e-7,atol=1e-18)
    np.testing.assert_allclose(audit.residual_norms,np.linalg.norm(res,axis=0).reshape(m,3),rtol=1e-7,atol=1e-18)
    audit.clear_vectors();assert audit.w is None and audit.r is None
