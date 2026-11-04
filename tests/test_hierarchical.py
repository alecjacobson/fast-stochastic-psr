import numpy as np
import pytest
from scipy.linalg import cho_solve

from geospsr.hierarchical import HierarchicalMaternSystem, batched_pcg
from geospsr.nonperiodic import DenseFreeSpacePosterior, FreeSpacePoissonMatern, RegionalVarianceCache


def test_batched_pcg_dependent_zero_and_strided_rhs():
    rng = np.random.default_rng(22)
    q = rng.normal(size=(53, 53))
    a = q @ q.T + 3*np.eye(53)
    v = rng.normal(size=53)
    b = np.column_stack([v, v, np.zeros(53), 2*v])[:, ::-1]
    actual, info = batched_pcg(lambda x: a@x, lambda x: x/np.diag(a)[:, None], b,
                               tolerance=1e-10)
    assert info['converged']
    np.testing.assert_allclose(actual, np.linalg.solve(a, b), atol=1e-10)
    assert info['iterations'][1] == 0


def test_pcg_reports_failure_and_nonpositive_curvature():
    a = np.diag(np.arange(1., 11.))
    _, info = batched_pcg(lambda x: a@x, lambda x: x, np.ones(10), max_iterations=1)
    assert not info['converged']
    with pytest.raises(np.linalg.LinAlgError):
        batched_pcg(lambda x: -x, lambda x: x, np.ones(10))


def test_residual_corrected_quadratic_form_identity_and_bound():
    rng = np.random.default_rng(29)
    v = rng.normal(size=(40, 40))
    a = v@v.T+2*np.eye(40)
    u = rng.normal(size=(40, 4))
    z, _, r = batched_pcg(lambda x: a@x, lambda x: x, u, max_iterations=3,
                          return_residual=True)
    g = u.T@z+z.T@r
    g = .5*(g+g.T)
    error = u.T@np.linalg.solve(a, u)-g
    np.testing.assert_allclose(error, r.T@np.linalg.solve(a, r), atol=2e-13)
    assert np.linalg.eigvalsh(error).min() >= -1e-12
    assert np.linalg.eigvalsh(r.T@r/2-error).min() >= -1e-12


def test_solver_cache_matches_dense_and_compresses_rhs():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(150, 3)); x /= np.linalg.norm(x, axis=1)[:, None]
    model = FreeSpacePoissonMatern(.3)
    p = DenseFreeSpacePosterior(model, x, x, .01)
    direct = RegionalVarianceCache(p, [1, 0, 0], .1, 5)
    compressed = RegionalVarianceCache.from_solver(
        model, x, x, lambda b: cho_solve((p.factor, True), b), [1, 0, 0], .1, 5,
        compression_tolerance=1e-9)
    q = [1, 0, 0] + rng.uniform(-.1, .1, (80, 3))
    for a, b in zip(direct.predict(q), compressed.predict(q)):
        np.testing.assert_allclose(a, b, atol=3e-10)
    assert compressed.solve_rank < 100


def test_native_h2_action_and_solve_against_independent_dense():
    pytest.importorskip('psr_higp_cext')
    rng = np.random.default_rng(24)
    x = rng.normal(size=(768, 3)); x /= np.linalg.norm(x, axis=1)[:, None]
    model = FreeSpacePoissonMatern(.3, 1.7)
    system = HierarchicalMaternSystem(model, x, .02, preconditioner_rank=100,
                                      leaf_size=80, threads=2)
    a = model.vector_covariance(x, x) + .02*np.eye(len(x))
    b = rng.normal(size=(len(x), 6))[:, ::2]
    np.testing.assert_allclose(system.matmul(b), a@b, atol=3e-6, rtol=1e-7)
    np.testing.assert_allclose(system.matmul(b[:, 0]), a@b[:, 0], atol=3e-6, rtol=1e-7)
    actual = system.solve(b, tolerance=1e-8)
    assert system.last_solve['converged']
    assert system.last_solve['max_iterations'] < 30
    assert np.linalg.norm(a@actual-b)/np.linalg.norm(b) < 2e-6
    np.testing.assert_allclose(actual, np.linalg.solve(a, b), rtol=1e-4, atol=2e-5)
    with pytest.raises(ValueError):
        system.matmul(np.ones(3))


def test_nonperiodic_h2_variance_without_dense_factor():
    pytest.importorskip('psr_higp_cext')
    rng = np.random.default_rng(25)
    x = rng.normal(size=(512, 3)); x /= np.linalg.norm(x, axis=1)[:, None]
    model = FreeSpacePoissonMatern(.3)
    system = HierarchicalMaternSystem(model, x, .01, threads=2, leaf_size=80)
    cache = RegionalVarianceCache.from_solver(model, x, x, system.solve, [1, 0, 0], .1, 5)
    dense = DenseFreeSpacePosterior(model, x, x, .01)
    queries = [1, 0, 0] + rng.uniform(-.1, .1, (60, 3))
    mean, variance = cache.predict(queries)
    mean_ref, var_ref = dense.predict(queries)
    assert np.max(np.abs(variance-var_ref))/model.prior_variance < 1e-4
    assert np.max(np.abs(mean-mean_ref)) < 1e-4
    assert np.all(variance > 0)


@pytest.mark.parametrize('geometry', ['sphere', 'clustered'])
def test_gpu_nested_actions_match_cpu_and_dense(geometry):
    pytest.importorskip('psr_higp_cext')
    cp = pytest.importorskip('cupy')
    from geospsr.hierarchical_gpu import GPUHierarchicalSystem
    rng = np.random.default_rng(26)
    x = rng.normal(size=(768, 3))
    if geometry == 'sphere':
        x /= np.linalg.norm(x, axis=1)[:, None]
    else:
        x[:600] *= .05
        x[600:] += [2, 0, 0]
    model = FreeSpacePoissonMatern(.3, 1.7)
    native = HierarchicalMaternSystem(model, x, .02, threads=2, leaf_size=60)
    gpu = GPUHierarchicalSystem(native)
    b = rng.normal(size=(len(x), 3))
    actual = cp.asnumpy(gpu.matmul(b))
    np.testing.assert_allclose(cp.asnumpy(gpu.precondition(b)), native.precondition(b),
                               atol=1e-8, rtol=1e-9)
    np.testing.assert_allclose(actual, native.matmul(b), atol=2e-12, rtol=2e-12)
    a = model.vector_covariance(x, x)+.02*np.eye(len(x))
    np.testing.assert_allclose(actual, a@b, atol=5e-6, rtol=1e-6)
    solution = gpu.solve(b, tolerance=1e-8)
    assert np.linalg.norm(a@solution-b)/np.linalg.norm(b) < 3e-6


@pytest.mark.parametrize('order', [3, 5])
def test_collapsed_scalar_fields_match_quadratic_form(order):
    cp = pytest.importorskip('cupy')
    from geospsr.regional_field import PolynomialMomentField
    rng = np.random.default_rng(27)
    x = rng.normal(size=(100, 3))
    p = DenseFreeSpacePosterior(FreeSpacePoissonMatern(.3), x, x, .01)
    cache = RegionalVarianceCache(p, [1, 0, 0], .1, order)
    field = PolynomialMomentField(cache)
    q = [1, 0, 0]+rng.uniform(-.1, .1, (150, 3))
    for actual, truth in zip(field.predict(q), cache.predict(q)):
        np.testing.assert_allclose(actual, truth, atol=2e-14)
    for actual, truth in zip(field.predict_gpu(q), cache.predict(q)):
        np.testing.assert_allclose(cp.asnumpy(actual), truth, atol=2e-14)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_inplace_gpu_dense_baseline_and_projection(dtype):
    pytest.importorskip('cupy')
    from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem
    rng = np.random.default_rng(28)
    x = rng.normal(size=(128, 3))
    model = FreeSpacePoissonMatern(.3, .0025)
    dense = GPUDenseMaternSystem(model, x, .000025, dtype=dtype)
    a = model.vector_covariance(x, x)+.000025*np.eye(len(x))
    rhs = rng.normal(size=(len(x), 5))
    np.testing.assert_allclose(dense.exact_matmul(rhs, block_size=31), a@rhs, atol=1e-16)
    tolerance = 1e-3 if dtype == 'float32' else 1e-10
    np.testing.assert_allclose(dense.solve(rhs), np.linalg.solve(a, rhs), atol=tolerance, rtol=tolerance)
    gram, projected = dense.project(rhs, x)
    np.testing.assert_allclose(gram, rhs.T@np.linalg.solve(a, rhs), atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(projected, rhs.T@np.linalg.solve(a, x), atol=tolerance, rtol=tolerance)
    refined = dense.refined_solve(rhs)
    assert np.linalg.norm(a@refined-rhs)/np.linalg.norm(rhs) < 1e-9


@pytest.mark.parametrize('cached', [True, False])
def test_dense_gpu_pcg_control(cached):
    pytest.importorskip('psr_higp_cext')
    cp = pytest.importorskip('cupy')
    from geospsr.nonperiodic_dense_cg import GPUDenseCGSystem
    rng = np.random.default_rng(30)
    x = rng.normal(size=(128, 3))
    model = FreeSpacePoissonMatern(.3, .0025)
    system = GPUDenseCGSystem(model, x, .000025, cache=cached, block_size=31, threads=2)
    a = model.vector_covariance(x, x)+.000025*np.eye(len(x))
    rhs = rng.normal(size=(len(x), 5))
    np.testing.assert_allclose(cp.asnumpy(system.matmul(rhs)), a@rhs, atol=1e-8, rtol=1e-4)
    gram, mean = system.project(rhs, x, variance_iterations=8, mean_tolerance=1e-7)
    np.testing.assert_allclose(gram, rhs.T@np.linalg.solve(a, rhs), atol=1e-2, rtol=2e-5)
    np.testing.assert_allclose(mean, rhs.T@np.linalg.solve(a, x), atol=1e-2, rtol=2e-5)
