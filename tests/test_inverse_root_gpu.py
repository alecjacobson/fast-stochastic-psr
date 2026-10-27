import cupy as cp
import numpy as np
import pytest
from geospsr.inverse_root import FixedChebyshevInverseRoot,spatial_box_spectral_bound
from geospsr.half_spectral import HalfCubeKernel,HalfCubeSpectrum
from geospsr.spectral import FourierPoisson,spectral_axis
from reproduction.correction_sampling import PolynomialCorrectionSampler


def dense_finite(points,cutoff,ell,noise):
    modes,rho=spectral_axis(cutoff,ell)
    matrix=np.ones((len(points),len(points)))
    for d in range(3):
        delta=points[:,None,d]-points[None,:,d]
        matrix*=np.cos(delta[...,None]*modes)@rho
    return matrix+noise*np.eye(len(points))


@pytest.mark.parametrize('ell,noise',[ (.003,.02),(.03,1e-5),(.3,1e-4)])
def test_complete_finite_covariance_with_seams_and_duplicates(ell,noise):
    rng=np.random.default_rng(19)
    x=rng.normal(0,.2,(19,3))
    x[:3]=[[-np.pi+1e-6,0,0],[np.pi-1e-6,0,0],[0,0,0]]
    x[3]=x[2]
    matrix=dense_finite(x,8,ell,noise)
    bound=spatial_box_spectral_bound(x,noise_variance=noise,grid=16,lengthscale=ell)
    eig,vec=np.linalg.eigh(matrix)
    assert eig.min()>=noise*(1-1e-9) and eig.max()<=bound['upper']
    polynomial=FixedChebyshevInverseRoot(bound['lower'],bound['upper'],1e-7)
    spectrum=HalfCubeSpectrum(8,lengthscale=ell,dtype='float64',eps=1e-12,upsampfac=2.)
    kernel=HalfCubeKernel(x,spectrum,noise,rhs_count=len(x))
    root=polynomial.apply(kernel,cp.eye(len(x),dtype='d'),xp=cp).get()
    expected=(vec/np.sqrt(eig))@vec.T
    assert np.linalg.norm(root-expected)/np.linalg.norm(expected)<1e-7
    assert np.linalg.norm(root@root.T-np.linalg.inv(matrix))/np.linalg.norm(np.linalg.inv(matrix))<2e-7


def test_streamed_correction_fields_match_dense_symmetric_root_and_partial_batch():
    rng=np.random.default_rng(91);x=rng.uniform(-.5,.5,(11,3));q=rng.uniform(-.5,.5,(7,3))
    matrix=dense_finite(x,5,.2,.001)
    eig,vec=np.linalg.eigh(matrix);root=(vec/np.sqrt(eig))@vec.T
    op=FourierPoisson(2,lengthscale=.2,dtype='float64',eps=1e-12)
    spectrum=HalfCubeSpectrum(5,lengthscale=.2,dtype='float64',eps=1e-12,upsampfac=2.)
    polynomial=FixedChebyshevInverseRoot(.001,len(x)*.001+.001,1e-7)
    sampler=PolynomialCorrectionSampler(x,op,polynomial,lambda r:HalfCubeKernel(x,spectrum,.001,rhs_count=r))
    counts=[]
    for block in sampler.draw_batches(5,seed=3,batch=2,retain_weights=True):
        expected=root@block['standard_normal'].get()
        actual=block['weights'].transpose(1,0,2).reshape(len(x),-1).get()
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)
        expected_fields=op.evaluate(q,op.aggregate(x,cp.asarray(expected.reshape(len(x),-1,3).transpose(1,0,2)))).get()
        np.testing.assert_allclose(op.evaluate(q,block['coefficients']).get(),expected_fields,rtol=1e-6,atol=1e-9)
        counts.append(block['stop']-block['start'])
    assert counts==[2,2,1]


def test_float32_high_condition_polynomial_roundoff_is_measured():
    rng=np.random.default_rng(11);q=np.linalg.qr(rng.normal(size=(41,41)))[0]
    eig=np.geomspace(.02,32.,41);matrix=(q*eig)@q.T
    polynomial=FixedChebyshevInverseRoot(.02,32.,1e-4)
    value=polynomial.apply(lambda x:cp.asarray(matrix,dtype='f')@x,cp.eye(41,dtype='f'),xp=cp).get()
    expected=(q/np.sqrt(eig))@q.T
    assert np.linalg.norm(value-expected)/np.linalg.norm(expected)<2e-4


def test_mixed_precision_defect_correction_matches_independent_dense_target():
    from geospsr.refinement import refine_solution
    rng=np.random.default_rng(924);x=rng.uniform(-.1,.1,(17,3));rhs=rng.normal(size=(17,3)).astype('f')
    # Deliberately bias the native action; acceptance must use the target below.
    native=HalfCubeKernel(x,HalfCubeSpectrum(6,lengthscale=.2),.00001001)
    exact=HalfCubeKernel(x,HalfCubeSpectrum(6,lengthscale=.2,dtype='float64',eps=1e-12,upsampfac=2.),.00001)
    initial,_=native.solve(cp.asarray(rhs),rtol=1e-5)
    result,info=refine_solution(lambda r:native.solve(r,rtol=.01),exact,cp.asarray(rhs),initial,xp=cp)
    matrix=dense_finite(x,6,.2,.00001)
    assert info['converged'] and info['corrections']>=1
    assert result.dtype==cp.float32
    assert np.max(np.linalg.norm(matrix@result.get()-rhs,axis=0)/np.linalg.norm(rhs,axis=0))<=2e-5
