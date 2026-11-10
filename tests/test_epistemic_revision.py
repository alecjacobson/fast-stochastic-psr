import numpy as np
import pytest
from scipy.special import ndtri

from reproduction.epistemic_revision import probabilities, summary


def test_zero_mean_is_half_probability_at_every_positive_variance():
    np.testing.assert_array_equal(probabilities(np.zeros(4),np.array([1e-12,1e-6,1.,1e6])),np.full(4,.5))
    # Equal zero contours do not imply equal confidence away from the contour.
    p=probabilities(np.ones(2),np.array([.01,100.]))
    assert p[0]<1e-10 and .45<p[1]<.5


def test_uncertain_errors_and_confident_errors_are_distinct():
    p=np.array([.99,.01,.55,.45]);m=-ndtri(p);v=np.ones(4)
    r=summary(m,v,1.,np.array([True,False,False,True]))
    assert r['wrong_count']==2 and r['confident_wrong_count']==0
    assert r['fraction_confident_95']==.5
    np.testing.assert_allclose(r['truth_probability_on_errors'],[.45,.45,.45])
    assert sum(b['count'] for b in r['reliability_bins'])==4
    r=summary(m,v,1.,np.array([False,False,False,True]))
    assert r['wrong_count']==3 and r['confident_wrong_count']==1


@pytest.mark.parametrize('variance',[0.,-1.,np.nan])
def test_no_clipping_of_invalid_uncertainty(variance):
    with pytest.raises(ValueError,match='strictly positive'):
        probabilities(np.zeros(1),np.array([variance]))
