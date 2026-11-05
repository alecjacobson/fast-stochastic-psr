import numpy as np
import pytest

from geospsr.nonperiodic import FreeSpacePoissonMatern
from reproduction.benchmark_hierarchical_variance import data, data_from_arguments


def test_partial_noisy_scan_contract_and_nested_observations():
    x, y, axes = data(128, 42, 'ellipsoid', 'front', .005)
    xx, yy, _ = data(256, 42, 'ellipsoid', 'front', .005)
    np.testing.assert_array_equal(x, xx[:128]); np.testing.assert_array_equal(y, yy[:128])
    assert np.all(x[:, 0] >= 0)
    np.testing.assert_allclose(np.sum((x/axes)**2, axis=1), 1.)
    _, clean, _ = data(128, 42, 'ellipsoid', 'front')
    np.testing.assert_allclose(y-clean, .005*np.random.default_rng(1042).normal(size=(128, 3)), atol=1e-16)
    for a, b in zip(data(128, 42, 'ellipsoid'), data_from_arguments(dict(n=128, seed=42, geometry='ellipsoid'))):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        data(128, 42, 'ellipsoid', normal_noise=-1.)


def test_gpu_radius_graph_at_cell_boundaries_and_duplicate_points():
    cp = pytest.importorskip('cupy')
    from scipy.sparse import csr_matrix
    from geospsr.nonperiodic_sparse import _gpu_graph
    x = np.array([[u, v, 0.] for u in [-1., 0., 1., 2.] for v in [-1., 0., 1.]])
    x = np.vstack((x, x[0], [np.nextafter(1., 2.), 0., 0.]))
    _, permutation, ptr, columns, counts = _gpu_graph(x, 1.)
    p = cp.asnumpy(permutation)
    graph = csr_matrix((np.ones(len(columns)), cp.asnumpy(columns), cp.asnumpy(ptr)), shape=(len(x), len(x)))
    expected = np.sum((x[:, None]-x[None, :])**2, axis=-1) <= 1.
    assert graph.has_canonical_format
    np.testing.assert_array_equal(graph.toarray(), expected[np.ix_(p, p)])
    np.testing.assert_array_equal(counts, expected.sum(axis=1)[p])


@pytest.mark.parametrize('cutoff', [6., 100.])
@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_sparse_control_against_independent_truncated_dense(cutoff, backend):
    pytest.importorskip('psr_higp_cext')
    cp = pytest.importorskip('cupy')
    from scipy.spatial.distance import cdist
    from scipy.sparse import csr_matrix
    from geospsr.nonperiodic_sparse import GPUSparseMaternSystem
    x, _, _ = data(256, 37, 'ellipsoid', 'front', .005)
    model = FreeSpacePoissonMatern(.08, .7)
    system = GPUSparseMaternSystem(model, x, .1, cutoff=cutoff, threads=2, graph_backend=backend)
    full = model.vector_covariance(x, x)
    keep = cdist(x, x) <= cutoff/model.kappa
    a = full*keep+.1*np.eye(len(x))
    graph = csr_matrix((cp.asnumpy(system.matrix.data), cp.asnumpy(system.matrix.indices),
                        cp.asnumpy(system.matrix.indptr)), shape=a.shape)
    assert graph.has_canonical_format
    permutation = cp.asnumpy(system.permutation)
    np.testing.assert_allclose(graph.toarray(), a[np.ix_(permutation, permutation)], atol=3e-16)
    np.testing.assert_array_equal(graph.toarray(), graph.toarray().T)
    assert system.diagnostics['nnz'] == np.count_nonzero(keep)
    b = np.random.default_rng(9).normal(size=(len(x), 3))
    np.testing.assert_allclose(cp.asnumpy(system.matmul(b)), a@b, atol=2e-15)
    solution = system.solve(b, tolerance=1e-10)
    assert np.linalg.norm(a@solution-b)/np.linalg.norm(b) < 1e-10
    if cutoff == 100.:
        np.testing.assert_allclose(a, full+.1*np.eye(len(x)))
    else:
        assert np.linalg.norm(full-full*keep) > 0
    with pytest.raises(ValueError):
        system.matmul(np.ones(len(x)+1))
