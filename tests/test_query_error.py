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


@pytest.mark.parametrize('row_chunk',[1,7,100])
def test_host_products_match_gpu_and_dense_with_partial_blocks(row_chunk):
    import cupy as cp
    from reproduction.query_error import QueryResidualAudit,HostQueryResidualAudit
    rng=np.random.default_rng(5001);n,m=19,7;noise=.02
    u=rng.normal(size=(n,n));a=u@u.T+noise*np.eye(n)
    rhs=rng.normal(size=(n,3*m))*np.tile([1.,1e-18,0.],m)
    rhs[:,3:6]=rhs[:,:3]  # Duplicate query, with zero and tiny components.
    ad=cp.asarray(a);approx=cp.asarray(a+.01*np.eye(n))
    action=lambda w:ad@w;solve=lambda b:cp.linalg.solve(approx,b)
    gpu=QueryResidualAudit(action,solve,noise)
    host=HostQueryResidualAudit(action,solve,noise,n,m,row_chunk=row_chunk)
    for start in range(0,3*m,9):
        b=cp.asarray(rhs[:,start:start+9]);gpu(b);host(b)
    raw=cp.eye(m,dtype='d')*100
    expected=gpu.correct(raw);actual=host.correct(raw)
    np.testing.assert_allclose(actual.get(),expected.get(),rtol=1e-13,atol=1e-13)
    for name in ['upper','shift']:
        np.testing.assert_allclose(getattr(host,name).get(),getattr(gpu,name).get(),rtol=1e-12,atol=1e-14)
    np.testing.assert_array_equal(host.cross_norms,gpu.cross_norms)
    np.testing.assert_array_equal(host.residual_norms,gpu.residual_norms)
    np.testing.assert_array_equal(host.host_w.transpose(2,1,0).reshape(n,3*m),gpu.w.get())
    assert not host.weights and not host.residuals
    assert host.record['gpu_product_vector_bytes_upper']==2*min(n,row_chunk)*m*8
    host.clear_vectors();assert host.host_w is None and host.host_r is None
    with pytest.raises(RuntimeError):host(cp.asarray(rhs))


def test_host_storage_cap_and_incomplete_capture_fail_explicitly():
    import cupy as cp
    from reproduction.query_error import HostQueryResidualAudit
    with pytest.raises(MemoryError):HostQueryResidualAudit(lambda x:x,lambda x:x,.1,1000,1000,max_host_bytes=1)
    host=HostQueryResidualAudit(lambda x:x,lambda x:x,.1,2,2)
    host(cp.ones((2,3)))
    with pytest.raises(ValueError):host.correct(cp.eye(1))


def test_adaptive_marginal_budget_matches_independent_dense_covariance():
    import cupy as cp
    from reproduction.query_error import DiagonalBudgetSolve
    rng=np.random.default_rng(5002);n,m=17,4;noise=.2
    u=rng.normal(size=(n,n));a=u@u.T+noise*np.eye(n)
    rhs=rng.normal(size=(n,3*m))*.02;rhs[:,2]=0
    ad=cp.asarray(a);approx=cp.asarray(a+.1*np.eye(n))
    solver=DiagonalBudgetSolve(lambda w:ad@w,lambda b:(cp.linalg.solve(approx,b),{}),noise,1.,rtol=1e-7,initial_corrections=0,max_corrections=12)
    w=solver(cp.asarray(rhs)).get();r=rhs-a@w
    def product(x,y):return sum(x[:,d::3].T@y[:,d::3] for d in range(3))
    target=np.eye(m)-product(rhs,np.linalg.solve(a,rhs))
    raw=np.eye(m)-product(rhs,w);shift=product(w,r);actual=raw-(shift+shift.T)*.5
    error=np.abs(np.diag(actual-target))/np.diag(target)
    assert error.max()<=1e-7
    assert error.max()<=solver.records[-1]['budget_history'][-1]['maximum_relative_marginal_bound']+1e-14
    assert solver.records[-1]['completed_corrections']>0
    fail=DiagonalBudgetSolve(lambda w:ad@w,lambda b:(cp.zeros_like(b),{}),noise,1.,rtol=1e-12,initial_corrections=0,max_corrections=0)
    with pytest.raises(RuntimeError):fail(cp.asarray(rhs))
