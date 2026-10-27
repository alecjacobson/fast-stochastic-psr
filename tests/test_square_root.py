import numpy as np
import pytest
from geospsr.square_root import FixedChebyshevRoot


@pytest.mark.parametrize('condition', [1., 2., 100., 10000.])
def test_root_and_complete_covariance_match_independent_eigendecomposition(condition):
    rng = np.random.default_rng(4621)
    q = np.linalg.qr(rng.normal(size=(29,29)))[0]
    eig = np.geomspace(.0004, .0004*condition, 29)
    matrix = (q*eig)@q.T
    root = FixedChebyshevRoot(eig[0], eig[-1], 1e-7)
    counts = []
    actual = root.apply(lambda x: matrix@x, np.eye(29), progress=lambda i,n: counts.append(i))
    expected = (q*np.sqrt(eig))@q.T
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 1e-7
    whiten = (q/np.sqrt(eig))@q.T
    assert np.linalg.norm(whiten@(actual@actual.T-matrix)@whiten, 2) < 2.1e-7
    assert counts == list(range(1, root.degree+1))
    a, b = rng.normal(size=(2,29,3))
    np.testing.assert_allclose(root.apply(lambda x: matrix@x, 2*a-b),
                               2*root.apply(lambda x: matrix@x, a)-root.apply(lambda x: matrix@x,b),
                               rtol=1e-9, atol=1e-9)


def test_fixed_root_limits_and_nonfinite_action():
    for args in [(0,1),(2,1),(1,np.inf)]:
        with pytest.raises(ValueError): FixedChebyshevRoot(*args)
    with pytest.raises(ValueError, match='required degree'):
        FixedChebyshevRoot(.00001,100,max_degree=2)
    root = FixedChebyshevRoot(1,2)
    with pytest.raises(FloatingPointError):
        root.apply(lambda x:np.full_like(x,np.nan),np.ones(4))
