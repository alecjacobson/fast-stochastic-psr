import cupy as cp
import numpy as np
import pytest
from geospsr.spectral import FourierPoisson
from reproduction.planar_fourier import PlanarFourierPlan
from reproduction.factor_deflation import FourierCorrectionFactor


@pytest.mark.parametrize('axis',[0,1,2])
@pytest.mark.parametrize('dtype',['float32','float64'])
def test_planar_forward_adjoint_against_explicit_fourier(axis,dtype):
    rng=np.random.default_rng(152);op=FourierPoisson(3,dtype=dtype,eps=1e-6 if dtype=='float32' else 1e-12)
    q=rng.uniform(-np.pi,np.pi,(17,3)).astype(dtype);q[:,axis]=np.array([-.75,-.35,0])[np.arange(17)%3]
    q[0,(axis+1)%3]=np.pi-1e-5;q[1,(axis+1)%3]=-np.pi+1e-5
    c=(rng.normal(size=(3,*op.shape))+1j*rng.normal(size=(3,*op.shape))).astype(op.cdtype)
    strengths=(rng.normal(size=(3,len(q)))+1j*rng.normal(size=(3,len(q)))).astype(op.cdtype)
    ns=op.ns.get().reshape(3,-1);phase=np.exp(1j*q.astype('d')@ns.astype('d'))
    expected2=c.reshape(3,-1)@phase.T;expected1=strengths@phase.conj()
    f=PlanarFourierPlan(2,op,q,3,axis=axis);a=PlanarFourierPlan(1,op,q,3,axis=axis)
    actual2=f.execute(cp.asarray(c)).get();actual1=a.execute(cp.asarray(strengths)).get().reshape(3,-1)
    tol=2e-5 if dtype=='float32' else 1e-10
    assert np.linalg.norm(actual2-expected2)/np.linalg.norm(expected2)<tol
    assert np.linalg.norm(actual1-expected1)/np.linalg.norm(expected1)<tol
    generic=op.plan(2,q,3).execute(cp.asarray(c)).get()
    assert np.linalg.norm(actual2-generic)/np.linalg.norm(generic)<tol
    assert abs(np.vdot(actual2,strengths)-np.vdot(c.reshape(3,-1),actual1))<tol*np.linalg.norm(actual2)*np.linalg.norm(strengths)
    zero=cp.zeros_like(cp.asarray(c));zero[:,3,3,3]=1+2j
    np.testing.assert_allclose(f.execute(zero).get(),1+2j,rtol=tol,atol=tol)
    with pytest.raises(ValueError):f.execute(cp.asarray(c[:2]))
    with pytest.raises(ValueError):PlanarFourierPlan(2,op,q,3,axis=axis,max_planes=2)


def test_planar_factor_partial_batches_and_coordinate_ownership():
    rng=np.random.default_rng(98);x=rng.normal(size=(7,3));q=rng.normal(size=(11,3));q[:,1]=np.arange(11)%2-.75
    op=FourierPoisson(3,dtype='float64',eps=1e-12);a=rng.normal(size=(7,7));chol=cp.asarray(np.linalg.cholesky(a@a.T+.3*np.eye(7)))
    planar=FourierCorrectionFactor(x,q,op,chol,batch=2,query_plan_factory=lambda kind,points,count:PlanarFourierPlan(kind,op,points,count))
    generic=FourierCorrectionFactor(x,q,op,chol,batch=2)
    z=cp.asarray(rng.normal(size=(21,5)));w=cp.asarray(rng.normal(size=(11,5)))
    for a,b in [(planar.forward(z),generic.forward(z)),(planar.adjoint(w),generic.adjoint(w))]:
        assert float(cp.linalg.norm(a-b)/cp.linalg.norm(b))<1e-10
    assert set(planar.plans)=={1,2}
    points=cp.asarray(q);plan=PlanarFourierPlan(2,op,points,1);coef=cp.ones((1,*op.shape),dtype=op.cdtype)
    before=plan.execute(coef).copy();points[:]=0
    np.testing.assert_allclose(plan.execute(coef).get(),before.get(),rtol=1e-12,atol=1e-12)
