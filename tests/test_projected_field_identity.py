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


@pytest.mark.parametrize('cutoff',[1,3])
@pytest.mark.parametrize('rank',[4,12])
def test_poisson_fourier_coefficients_with_full_analytic_training(cutoff,rank):
    # Independent CPU Fourier/image-sum construction: no NUFFT or GPU calls.
    rng=np.random.default_rng(5402);n,m=19,17;ell=.2;variance=.1;noise=.0002
    x=rng.uniform(-np.pi,np.pi,(n,3));z=rng.uniform(-np.pi,np.pi,(m,3))
    query=np.vstack([z,rng.uniform(-np.pi,np.pi,(8,3))])
    axis=np.arange(-cutoff,cutoff+1);a=np.sqrt(3)/ell;r=np.exp(-a)
    normalization=1+2*r/(1-r)+2*a*r/(1-r)**2
    # Fourier transform of (1+a|t|) exp(-a|t|), periodized with period one.
    rho1=variance*4*a**3/(a*a+(2*np.pi*axis)**2)**2/normalization
    freq=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    rho=(rho1[:,None,None]*rho1[None,:,None]*rho1[None,None,:]).ravel()
    norm2=np.sum(freq**2,axis=1);positive=norm2>0
    scalar=np.zeros_like(rho);scalar[positive]=rho[positive]/norm2[positive]
    sigma=np.sqrt(scalar);safe=np.where(positive,sigma,1.)
    def features(q):
        angle=q@freq.T
        return np.concatenate([np.cos(angle)*sigma,np.sin(angle)*sigma],axis=1)
    def cross(q):
        phase=(q[:,None,:]-x[None,:,:])@freq.T
        return np.concatenate([np.sin(phase)@(scalar*freq[:,d]) for d in range(3)],axis=1)
    # Analytic vector covariance from a rapidly converged PERIODIC image sum,
    # independent of both the finite Fourier features and the range backend.
    delta=(x[:,None,:]-x[None,:,:])/(2*np.pi);periodic=np.zeros_like(delta)
    for image in range(-20,21):
        distance=np.abs(delta+image);periodic+=(1+a*distance)*np.exp(-a*distance)
    kernel=variance**3*np.prod(periodic/normalization,axis=2)
    matrix=np.kron(np.eye(3),kernel+noise*np.eye(n))
    phi_z=features(z);kz=phi_z@phi_z.T;lam,q=np.linalg.eigh(kz)
    lam=lam[::-1][:rank];q=q[:,::-1][:,:rank]
    w=np.linalg.solve(matrix+.005*np.eye(3*n),cross(z).T@q)
    source=np.exp(-1j*(x@freq.T)).T
    l_coeff=-1j*sum((scalar*freq[:,d])[:,None]*(source@w[d*n:(d+1)*n]) for d in range(3))
    b_coeff=scalar[:,None]*(np.exp(-1j*(z@freq.T)).T@(q/lam))
    j=np.concatenate([l_coeff.real/safe[:,None],-l_coeff.imag/safe[:,None]])
    d=np.concatenate([b_coeff.real/safe[:,None],-b_coeff.imag/safe[:,None]])
    phi=features(query);b=phi@d;l=cross(query)@w
    np.testing.assert_allclose(phi@j,l,rtol=1e-10,atol=1e-13)
    np.testing.assert_allclose(phi_z@d,q,rtol=1e-10,atol=1e-12)
    # Scalar-whitened feature coupling of each observed vector component.
    projected=[]
    for component in range(3):
        factor=(scalar*freq[:,component])/safe
        projected.append(np.concatenate([-np.sin(x@freq.T)*factor,np.cos(x@freq.T)*factor],axis=1))
    projected=np.concatenate(projected,axis=0)
    np.testing.assert_allclose(j,projected.T@w,rtol=1e-10,atol=1e-13)
    remainder=matrix-projected@projected.T
    assert np.linalg.eigvalsh(remainder).min()>0
    g=w.T@matrix@w;h=g-j.T@j;h=(h+h.T)*.5
    np.testing.assert_allclose(h,w.T@remainder@w,rtol=1e-10,atol=1e-13)
    eig,vec=np.linalg.eigh(h);assert eig.min()>0
    root=(vec*np.sqrt(eig))@vec.T
    field_factor=phi-b@j.T;extra=b@root
    covariance=field_factor@field_factor.T+extra@extra.T
    target=phi@phi.T-cross(query)@np.linalg.solve(matrix,cross(query).T)
    residual=cross(query).T-matrix@w@b.T
    np.testing.assert_allclose(covariance-target,residual.T@np.linalg.solve(matrix,residual),rtol=1e-9,atol=1e-13)
    white=rng.normal(size=(2*len(freq),7));other=rng.normal(size=(rank,7))
    t=j.T@white+root@other
    coefficients=sigma[:,None]*(white[:len(freq)]-1j*white[len(freq):])-b_coeff@t
    np.testing.assert_array_equal(coefficients[~positive],0)
    via_coefficients=(np.exp(1j*(query@freq.T))@coefficients).real
    np.testing.assert_allclose(via_coefficients,phi@(white-d@t),rtol=1e-10,atol=1e-13)
    # Exercise the implementation against the independently derived coefficients.
    from geospsr.projected_field import CompressedJointField
    sampler=CompressedJointField(scalar,l_coeff.T,b_coeff.T,g,frequency_chunk=23)
    cached=sampler.draw(7,white=(white[:len(freq)]-1j*white[len(freq):]).T,extra_white=other.T)
    np.testing.assert_allclose(cached,coefficients.T,rtol=1e-9,atol=1e-12)
