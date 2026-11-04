import numpy as np
import pytest

from geospsr.nonperiodic import FreeSpacePoissonMatern
from geospsr.hierarchical import HierarchicalMaternSystem


@pytest.mark.parametrize('neighbors', [1, 7, 24])
def test_prefix_predecessor_neighbors_against_exhaustive_oracle(neighbors):
    pytest.importorskip('psr_higp_kd_cext')
    x = np.random.default_rng(77).normal(size=(257, 3))
    x[10:20] = x[0]
    system = HierarchicalMaternSystem(FreeSpacePoissonMatern(.2, .7), x, .05,
                                     preconditioner_rank=8, neighbors=neighbors, threads=2,
                                     build_hierarchy=False, preconditioner_backend='prefix-kd')
    snap = system.preconditioner.snapshot()
    rank = len(snap['L11_transpose'])
    xx = x[snap['permutation'].ravel()][rank:]
    ptr, columns = snap['G_indptr'].ravel(), snap['G_columns'].ravel()
    for i in range(len(xx)):
        distances = np.sum((xx[:i]-xx[i])**2, axis=1)
        nearest = np.lexsort((np.arange(i), distances))[:neighbors-1]
        expected = np.sort(np.r_[nearest, i])
        np.testing.assert_array_equal(np.sort(columns[ptr[i]:ptr[i+1]]), expected)
    b = np.random.default_rng(31).normal(size=(len(x), 3))
    assert np.sum(b*system.precondition(b)) > 0


def test_streamed_hierarchy_and_blocked_variance_match_cached_actions():
    pytest.importorskip('psr_higp_kd_cext')
    cp = pytest.importorskip('cupy')
    from geospsr.hierarchical_gpu import GPUHierarchicalSystem
    rng = np.random.default_rng(8)
    x = rng.normal(size=(512, 3))*.3
    model = FreeSpacePoissonMatern(.15, .7)
    native = HierarchicalMaternSystem(model, x, .03, tolerance=1e-7, leaf_size=30,
                                     preconditioner_rank=8, neighbors=24, threads=2,
                                     preconditioner_backend='prefix-kd')
    cached = GPUHierarchicalSystem(native)
    streamed = GPUHierarchicalSystem(native, interaction_budget_bytes=64*1024, rhs_block_size=3)
    assert len(streamed.interactions.chunks) > 1
    assert streamed.interactions.workspace_bytes <= 64*1024
    b = rng.normal(size=(len(x), 7))
    a = model.vector_covariance(x, x)+.03*np.eye(len(x))
    np.testing.assert_allclose(cp.asnumpy(streamed.matmul(b)), cp.asnumpy(cached.matmul(b)), atol=2e-13)
    assert np.linalg.norm(cp.asnumpy(streamed.matmul(b))-a@b)/np.linalg.norm(a@b) < 1e-6
    y = rng.normal(size=(len(x), 3))
    for actual, expected in zip(streamed.project(b, y, variance_iterations=5, mean_tolerance=1e-8),
                               cached.project(b, y, variance_iterations=5, mean_tolerance=1e-8)):
        np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=1e-9)


def test_streamed_radius_action_against_full_and_cached_kernel():
    pytest.importorskip('psr_higp_kd_cext')
    cp = pytest.importorskip('cupy')
    from geospsr.nonperiodic_local_gpu import GPUStreamedLocalSystem
    from geospsr.nonperiodic_sparse import GPUSparseMaternSystem
    from scipy.spatial.distance import cdist
    x = np.random.default_rng(17).normal(size=(384, 3))*.2
    model = FreeSpacePoissonMatern(.1, .7)
    for cutoff in [6., 100.]:
        local = GPUStreamedLocalSystem(model, x, .05, cutoff=cutoff, budget_bytes=48*1024,
                                       preconditioner_backend='prefix-kd', threads=2, rhs_block_size=3)
        cached = GPUSparseMaternSystem(model, x, .05, cutoff=cutoff,
                                       preconditioner_backend='prefix-kd', threads=2)
        assert len(local.chunks) > 1
        assert local.interaction_workspace_bytes <= 48*1024
        a = model.vector_covariance(x, x)*(cdist(x, x)<=cutoff/model.kappa)+.05*np.eye(len(x))
        b = np.random.default_rng(18).normal(size=(len(x), 7))
        np.testing.assert_allclose(cp.asnumpy(local.matmul(b)), a@b, atol=1e-13)
        np.testing.assert_allclose(cp.asnumpy(local.matmul(b[:, 0])), a@b[:, 0], atol=1e-13)
        np.testing.assert_allclose(cp.asnumpy(local.matmul(b)), cp.asnumpy(cached.matmul(b)), atol=1e-13)
        z = local.solve(b, tolerance=1e-9)
        assert np.linalg.norm(a@z-b)/np.linalg.norm(b) < 1e-9
