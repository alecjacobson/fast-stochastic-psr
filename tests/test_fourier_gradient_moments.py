import numpy as np
import pytest
from geospsr.fourier_gradient_moments import FourierGradientSquareAccumulator
from tests.test_fourier_moments import evaluate


@pytest.mark.parametrize('cutoff',[0,1,3,4])
def test_gradient_identity_against_independent_derivative_sums(cutoff):
    rng=np.random.default_rng(926)
    c=rng.normal(size=(3,)+(2*cutoff+1,)*3)+1j*rng.normal(size=(3,)+(2*cutoff+1,)*3)
    c[:,cutoff,cutoff,cutoff]+=1e8 # irrelevant constant must not contaminate gradients
    q=rng.uniform(-np.pi,np.pi,(37,3))
    n=np.stack(np.meshgrid(*([np.arange(-cutoff,cutoff+1)]*3),indexing='ij'))
    expected=sum((evaluate(1j*c*n[d],q)**2).sum(0) for d in range(3))
    acc=FourierGradientSquareAccumulator(cutoff)
    acc.add(c[:2]);acc.add(c[2:])
    actual=evaluate(acc.coefficients(.5,-1),q)[0]
    np.testing.assert_allclose(actual,.5-expected,rtol=1e-11,atol=1e-10)
    assert acc.count==3


def test_single_precision_packed_fields_against_derivative_sums():
    import cupy as cp
    rng=np.random.default_rng(927);k=4
    c=(rng.normal(size=(3,9,9,9))+1j*rng.normal(size=(3,9,9,9)))*1e-5
    q=rng.uniform(-np.pi,np.pi,(29,3))
    n=np.stack(np.meshgrid(*([np.arange(-k,k+1)]*3),indexing='ij'))
    expected=sum((evaluate(1j*c*n[d],q)**2).sum(0) for d in range(3))
    acc=FourierGradientSquareAccumulator(k,xp=cp,working_dtype='float32');acc.add(cp.asarray(c))
    actual=evaluate(acc.coefficients().get(),q)[0]
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<2e-6
