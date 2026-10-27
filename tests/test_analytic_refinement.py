import cupy as cp
import numpy as np
import pytest
from geospsr.analytic_refinement import refine_analytic


def test_defect_correction_original_rhs_and_input_ownership():
    rng=np.random.default_rng(139);v=rng.normal(size=(17,17));base=v@v.T+5*np.eye(17)
    a=base+.05*np.eye(17);b=rng.normal(size=(17,4))*[0,1e-5,1,1e5]
    ad,bd=cp.asarray(a),cp.asarray(b);inverse=cp.asarray(np.linalg.inv(base))
    x0=inverse@bd;copy=x0.copy()
    solve=lambda r:(inverse@r,{'converged':True})
    x,info=refine_analytic(lambda x:ad@x,solve,bd,x0,rtol=1e-11)
    assert info['converged'] and info['corrections']>1 and info['analytic_actions']==info['corrections']+1
    np.testing.assert_array_equal(x0.get(),copy.get())
    np.testing.assert_allclose(x.get(),np.linalg.solve(a,b),rtol=1e-8,atol=1e-10)
    assert max(info['relative_residual'])<=1e-11
    np.testing.assert_array_equal(x[:,0].get(),np.zeros(17))


def test_exact_start_and_nonconvergence_are_reported():
    b=cp.ones((5,2));x,info=refine_analytic(lambda x:x,lambda r:(_ for _ in ()).throw(AssertionError()),b,b)
    assert info['converged'] and info['corrections']==0
    x,info=refine_analytic(lambda x:2*x,lambda r:(2*r,{}),b,cp.zeros_like(b),max_corrections=2)
    assert not info['converged'] and info['corrections']==2 and info['relative_residual']==[9.,9.]
    with pytest.raises(ArithmeticError):refine_analytic(lambda x:x,lambda r:(r*float('nan'),{}),b,cp.zeros_like(b))
    with pytest.raises(ValueError):refine_analytic(lambda x:x,lambda r:(r,{}),b,b*1j)
