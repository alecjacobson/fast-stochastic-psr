import cupy as cp
import numpy as np
import pytest
from geospsr.packed_queries import PackedDirectEvaluator


@pytest.mark.parametrize('count,dtype',[(1,'float64'),(2,'float32'),(5,'float64')])
def test_direct_packed_real_fields_match_explicit_fourier_sum(count,dtype):
    rng=np.random.default_rng(31)
    x=rng.uniform(-np.pi,np.pi,(17,3)).astype(dtype)
    coefficients=(rng.normal(size=(count,5,5,5))+1j*rng.normal(size=(count,5,5,5)))
    coefficients*=np.geomspace(1e-12,1e12,count)[:,None,None,None]
    coefficients=coefficients.astype('complex64' if dtype=='float32' else 'complex128')
    if count>2: coefficients[2]=0
    ns=np.stack(np.meshgrid(*[np.arange(-2,3)]*3,indexing='ij'),-1).reshape(-1,3)
    expected=(coefficients.reshape(count,-1)@np.exp(1j*ns@x.astype('d').T)).real
    evaluator=PackedDirectEvaluator(x,(5,5,5),dtype=dtype,eps=1e-6 if dtype=='float32' else 1e-11,upsampfac=1.25 if dtype=='float32' else 2.)
    actual=cp.asnumpy(evaluator.evaluate(coefficients));active=np.linalg.norm(expected,axis=1)>0
    assert (np.linalg.norm(actual-expected,axis=1)[active]/np.linalg.norm(expected,axis=1)[active]).max()<(4e-6 if dtype=='float32' else 2e-10)
    np.testing.assert_array_equal(actual[~active],0.)
    plan=evaluator.prepare(count)
    evaluator.evaluate(coefficients)
    assert evaluator.prepare(count) is plan
