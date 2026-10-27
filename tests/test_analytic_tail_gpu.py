import cupy as cp
import numpy as np
import pytest
from geospsr.analytic_tail import AnalyticTailAction
from geospsr.periodic_leaf_cache import LeafCachedRangeKernel
from geospsr.spectral import FourierKernel
from geospsr.training_spectrum import FourierCovariance, finite_covariance_rows
from geospsr.spectral_tail import analytic_covariance_rows64
from geospsr.square_root import FixedChebyshevRoot, analytic_tail_spectral_bound


@pytest.mark.parametrize('ell,noise', [(.01,.02),(.005,.0004),(1.,.00001)])
def test_full_tail_covariance_including_seams_duplicates_and_posterior(ell,noise):
    rng = np.random.default_rng(4623)
    x = rng.normal(0,.03,(23,3)).astype('f')
    x[:3] = [[-np.pi+1e-6,0,0],[np.pi-1e-6,0,0],[0,0,0]]
    x[3] = x[2]
    spectrum = FourierCovariance(8,ell,dtype='float64',eps=1e-12,upsampfac=2.)
    finite = FourierKernel(x,spectrum,0,rhs_count=len(x))
    analytic = LeafCachedRangeKernel(x,ell,.1,noise,leaf_size=8)
    action = AnalyticTailAction(analytic,finite)
    a = analytic_covariance_rows64(x,x,ell).get()+noise*np.eye(len(x))
    low = finite_covariance_rows(x,x,spectrum,dtype='float64').get()
    target = a-low
    bound = analytic_tail_spectral_bound(x,8,ell,noise_variance=noise,grid=16)
    eig,vec = np.linalg.eigh(target)
    assert eig.min() >= noise*(1-1e-9)
    assert eig.max() <= bound['upper']*(1+1e-12)
    np.testing.assert_allclose(action(cp.eye(len(x),dtype='d')).get(),target,rtol=1e-8,atol=1e-12)
    root = FixedChebyshevRoot(bound['lower'],bound['upper'],1e-7)
    actual = root.apply(action,cp.eye(len(x),dtype='d'),xp=cp).get()
    expected = (vec*np.sqrt(eig))@vec.T
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 1e-7
    white = (vec/np.sqrt(eig))@vec.T
    assert np.linalg.norm(white@(actual@actual.T-target)@white,2) < 2.1e-7

    # Independent finite joint prior features at tiny cutoff 2. Test covariance
    # conditioning deterministically, with every feature, not Monte Carlo noise.
    # The tail above uses cutoff8: construct a compatible scalar feature whose
    # nonzero cross spectrum is supported inside that cube.
    from geospsr.spectral import spectral_axis
    modes, rho = spectral_axis(2,ell)
    ns = np.stack(np.meshgrid(modes,modes,modes,indexing='ij')).reshape(3,-1).T
    rho3 = (rho[:,None,None]*rho[None,:,None]*rho[None,None,:]).ravel()
    q = rng.normal(0,.02,(7,3))
    norm2 = (ns*ns).sum(1); denom = np.where(norm2==0,1,norm2)
    phaseq,phasex = q@ns.T,x.astype('d')@ns.T
    vf = np.concatenate([np.cos(phasex),np.sin(phasex)],axis=1)*np.tile(np.sqrt(rho3),2)
    sf = [np.concatenate([np.sin(phaseq),-np.cos(phaseq)],axis=1)*np.tile(np.sqrt(rho3)*ns[:,d]/denom,2) for d in range(3)]
    cff = sum(f@f.T for f in sf)
    cross = [f@vf.T for f in sf]
    solves = [np.linalg.solve(a,c.T).T for c in cross]
    wanted = cff-sum(c@h.T for c,h in zip(cross,solves))
    # Intermediate modes 2<|n|<=8 are independent of these scalar features.
    extra = low-vf@vf.T
    got = sum((f-h@vf)@(f-h@vf).T+h@(extra+actual@actual.T)@h.T for f,h in zip(sf,solves))
    np.testing.assert_allclose(got,wanted,rtol=2e-6,atol=1e-13)


def test_tail_contract_rejects_released_normalization_and_noise_double_counting():
    x = np.zeros((3,3),dtype='f')
    with pytest.raises(ValueError, match='analytic normalization'):
        analytic_tail_spectral_bound(x,8,normalization='released')
    analytic = LeafCachedRangeKernel(x,noise_variance=.02,leaf_size=8)
    for dtype,normalization,noise,match in [('float64','analytic',.02,'exclude'),
                                          ('float32','analytic',0,'double'),
                                          ('float64','released',0,'normalization')]:
        finite = FourierKernel(x,FourierCovariance(2,dtype=dtype,normalization=normalization),noise)
        with pytest.raises(ValueError, match=match): AnalyticTailAction(analytic,finite)
