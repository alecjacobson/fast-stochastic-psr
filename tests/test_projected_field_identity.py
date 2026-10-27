"""Independent finite-feature proof controls for future projected field samples.

This is not a Fourier implementation or a performance benchmark. It tests the
joint Gaussian construction at reference AND unseen query features, including
their cross-covariances, before investing in compressed spectral preparation.
"""
import numpy as np
import pytest
from reproduction.query_projection import assemble_projected


@pytest.mark.parametrize('rank',[0,4,11])
@pytest.mark.parametrize('shift',[0.,.03])
def test_compressed_joint_field_and_unseen_query_error_identity(rank,shift):
    rng=np.random.default_rng(5401);n,m,f=23,17,11;noise=.02
    phi_z=rng.normal(size=(m,f));v=rng.normal(size=(n,f))
    tail=rng.normal(size=(n,7))*.2
    a=v@v.T+tail@tail.T+noise*np.eye(n)
    kz=phi_z@phi_z.T;lam,q=np.linalg.eigh(kz)
    lam=lam[::-1][:rank];q=q[:,::-1][:,:rank]
    # The reference prior eigenvectors define functions, not just point values.
    d=(phi_z.T@q)/lam if rank else np.empty((f,0))
    xz=v@phi_z.T;w=np.linalg.solve(a+shift*np.eye(n),xz@q)
    residual=xz@q-a@w
    j=v.T@w;g=w.T@a@w
    h=(g-j.T@j);h=(h+h.T)*.5
    # The omitted data spectrum and observation noise remain in a SMALL
    # compressed covariance. They are not replaced by a diagonal.
    independent_h=w.T@(tail@tail.T+noise*np.eye(n))@w
    np.testing.assert_allclose(h,independent_h,rtol=1e-10,atol=1e-12)
    if rank:
        eig,vec=np.linalg.eigh(h)
        assert eig.min()>=-100*np.finfo(float).eps*rank*np.linalg.norm(h)
        hroot=(vec*np.sqrt(np.maximum(eig,0)))@vec.T
    else:hroot=np.empty((0,0))
    # t=J.T z + Hroot e; posterior coefficients c=z-D t.
    coefficient_factor=np.column_stack([np.eye(f)-d@j.T,-d@hroot])
    coefficient_covariance=coefficient_factor@coefficient_factor.T
    phi_new=rng.normal(size=(8,f));phi_new[1]=phi_new[0]
    phi=np.vstack([phi_z,phi_new]);b=phi@d;cross=phi@v.T
    actual=phi@coefficient_covariance@phi.T
    exact=phi@phi.T-cross@np.linalg.solve(a,cross.T)
    rq=cross.T-a@w@b.T
    identity=rq.T@np.linalg.solve(a,rq)
    np.testing.assert_allclose(actual-exact,identity,rtol=1e-10,atol=1e-11)
    # Nyström prior innovation controls new-query truncation, including its
    # cross-covariance with the reference panel. Orthogonality at new queries
    # is not assumed in this PSD inequality.
    innovation=phi@phi.T-(b*lam)@b.T
    u=residual.T@residual/noise
    upper=2*innovation+2*b@u@b.T
    assert np.linalg.eigvalsh(upper-(actual-exact)).min()>=-1e-10*np.linalg.norm(phi@phi.T)
    # On the reference panel, the function lift agrees with Q and exactly
    # recovers the existing projected-query covariance formula.
    np.testing.assert_allclose(phi_z@d,q,atol=1e-13,rtol=1e-12)
    projected,_,_=assemble_projected(kz,q,phi_z@j,w.T@residual,kz-(q*lam)@q.T,u)
    np.testing.assert_allclose(actual[:m,:m],projected,rtol=1e-10,atol=1e-11)
    np.testing.assert_allclose(actual[m],actual[m+1],rtol=0,atol=1e-13)
