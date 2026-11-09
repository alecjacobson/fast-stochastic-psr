import numpy as np
import pytest


@pytest.mark.parametrize('budget',[None,64*1024])
def test_mixed_hierarchy_refines_against_float64_and_matches_dense_probability(budget):
    pytest.importorskip('psr_higp_kd_cext')
    cp=pytest.importorskip('cupy')
    from geospsr.hierarchical import HierarchicalMaternSystem
    from geospsr.hierarchical_gpu import GPUHierarchicalSystem
    from geospsr.nonperiodic import FreeSpacePoissonMatern,DenseFreeSpacePosterior
    from geospsr.scalar_contacts import ContactConditionedDensePosterior
    from reproduction.partial_direct_gpu import DirectContactQueries
    from scipy.special import ndtr
    rng=np.random.default_rng(145)
    x=rng.normal(size=(384,3))*.2;y=rng.normal(size=x.shape);q=rng.normal(size=(17,3))*.3
    model=FreeSpacePoissonMatern(.48,1/3)
    native=HierarchicalMaternSystem(model,x,.04,tolerance=1e-9,leaf_size=30,
        preconditioner_rank=12,neighbors=24,threads=2,preconditioner_backend='prefix-kd')
    double=GPUHierarchicalSystem(native,interaction_budget_bytes=budget)
    mixed=GPUHierarchicalSystem(native,interaction_budget_bytes=budget,dtype='float32',rhs_block_size=8,release_native=True)
    b=rng.normal(size=(len(x),5))
    np.testing.assert_allclose(cp.asnumpy(mixed.accurate_matmul(b)),cp.asnumpy(double.matmul(b)),atol=1e-12)
    assert np.linalg.norm(cp.asnumpy(mixed.matmul(b)-double.matmul(b)))/np.linalg.norm(cp.asnumpy(double.matmul(b)))<2e-6
    z=mixed.solve(b,tolerance=1e-9)
    assert np.max(np.linalg.norm(cp.asnumpy(double.matmul(z))-b,axis=0)/np.linalg.norm(b,axis=0))<1e-9
    zv,info,rv=mixed.variance_solve(b,6)
    np.testing.assert_allclose(cp.asnumpy(rv),b-cp.asnumpy(double.matmul(zv)),atol=1e-12)
    np.testing.assert_allclose(info['relative_residuals'],np.linalg.norm(cp.asnumpy(rv),axis=0)/np.linalg.norm(b,axis=0))
    assert mixed.preconditioner_gpu.coupling.dtype==cp.float32
    assert mixed.reference_workspace_bytes>0
    with pytest.raises(ValueError,match='stopping'):
        mixed.solve(b,tolerance=0)
    with pytest.raises(RuntimeError,match='converge'):
        mixed.solve(b,tolerance=1e-12,max_iterations=1)
    assert mixed.last_solve['max_iterations'] == 1
    np.testing.assert_array_equal(mixed.solve(np.zeros(len(x))),np.zeros(len(x)))
    exact=ContactConditionedDensePosterior(DenseFreeSpacePosterior(model,x,y,.04),x[:4],np.zeros(4),.0001)
    posterior=DirectContactQueries(model,x,y,x[:4],mixed,iterations=12,solve_tolerance=1e-9)
    m,v=posterior.predict(q,batch_size=7);rm,rv=exact.predict(q)
    assert np.all(v>0)
    assert np.max(abs(ndtr(-m/np.sqrt(v))-ndtr(-rm/np.sqrt(rv))))<1e-4
    assert np.max(abs(v-rv))/model.prior_variance<1e-6
    # A float32 caller basis must not demote blocked residuals or Gram sums.
    basis=rng.normal(size=(len(x),11)).astype(np.float32)
    gram,projected=mixed.project(basis,y,variance_iterations=12,mean_tolerance=1e-9)
    a=model.vector_covariance(x,x)+.04*np.eye(len(x))
    expected=basis.T@np.linalg.solve(a,basis)
    assert np.linalg.norm(gram-expected)/np.linalg.norm(expected)<1e-6
    expected=basis.T@np.linalg.solve(a,y)
    assert np.linalg.norm(projected-expected)/np.linalg.norm(expected)<1e-6
