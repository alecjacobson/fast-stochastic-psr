import numpy as np
import pytest
from geospsr.refinement import refine_solution


def test_double_defect_corrects_a_biased_native_operator_at_float32_storage():
    rng=np.random.default_rng(541);q=np.linalg.qr(rng.normal(size=(31,31)))[0]
    matrix=(q*np.geomspace(.02,3.,31))@q.T
    # An intentionally biased approximate action mimics error that restarting
    # the same native system cannot remove from the actual target residual.
    approximate=matrix+2e-4*np.eye(31)
    rhs=rng.normal(size=(31,3)).astype('f')
    initial=np.linalg.solve(approximate,rhs).astype('f');original=initial.copy()
    def solve(residual):return np.linalg.solve(approximate,residual).astype('f'),{'converged':True}
    value,info=refine_solution(solve,lambda x:matrix@x,rhs,initial,target_rtol=2e-5,max_corrections=3)
    assert info['converged'] and 1<=info['corrections']<=3
    assert value.dtype==np.dtype('float32')
    np.testing.assert_array_equal(initial,original)
    assert np.max(np.linalg.norm(matrix@value-rhs,axis=0)/np.linalg.norm(rhs,axis=0))<=2e-5


def test_failed_corrections_cannot_override_independent_residual():
    rhs=np.ones((4,2),dtype='f');initial=np.zeros_like(rhs)
    value,info=refine_solution(lambda r:(np.zeros_like(r),{'converged':True}),lambda x:x,
                               rhs,initial,max_corrections=2)
    assert not info['converged'] and info['corrections']==2
    np.testing.assert_array_equal(info['relative_residual_history'],np.ones((3,2)))


def test_zero_rhs_and_precision_contract():
    rhs=np.zeros((4,2),dtype='f')
    def unexpected(r):raise AssertionError('no correction needed')
    _,info=refine_solution(unexpected,lambda x:x,rhs,rhs)
    assert info['converged'] and info['corrections']==0
    with pytest.raises(ValueError,match='float64'):
        refine_solution(unexpected,lambda x:x.astype('f'),rhs,rhs)
