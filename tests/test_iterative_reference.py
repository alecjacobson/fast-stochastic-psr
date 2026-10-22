"""Verify the stronger compiled baseline preserves released SDD updates."""
import importlib.util
import os
from pathlib import Path

import jax
import numpy as np
import pytest

from geospsr.kernels import Matern32Kernel, ProductKernel
from reproduction.iterative_reference import StochasticDualDescent
from geospsr.sdd import sdd_solve


def test_compiled_sdd_matches_released_loop_and_resumes():
    path = Path(os.environ.get('GEOSPSR_RUNTIME', '/opt/geospsr-runtime')) / 'upstream/geospsr/sdd.py'
    if not path.exists():
        pytest.skip('pinned upstream checkout required for untouched-code comparison')
    spec = importlib.util.spec_from_file_location('released_sdd', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(21)
    x = rng.uniform(-.5, .5, (37, 3)).astype('f')
    y = rng.normal(size=x.shape).astype('f')
    k = Matern32Kernel(.03, .1)
    kernel = ProductKernel(k, k, k)
    reference = module.sdd_solve(jax.numpy.asarray(x), jax.numpy.asarray(y), kernel,
                                 np.sqrt(.02), jax.random.key(7), bs=13, lr=.1, iterations=251)
    solver = StochasticDualDescent(x, y, kernel, batch=13, lr=.1, seed=7)
    solver.compile()
    solver.advance(100)
    result = solver.advance(151)
    assert solver.iterations == 251
    np.testing.assert_allclose(np.asarray(result), np.asarray(reference), rtol=2e-5, atol=3e-5)


def test_compiled_sdd_api_preserves_vectorized_solves():
    rng = np.random.default_rng(15)
    x = jax.numpy.asarray(rng.uniform(-1, 1, (31, 3)).astype('f'))
    targets = jax.numpy.asarray(rng.normal(size=(2, 31, 3)).astype('f'))
    k = Matern32Kernel(.03, .1)
    kernel = ProductKernel(k, k, k)
    def solve(y, compiled):
        return sdd_solve(x, y, kernel, .02, jax.random.key(3), bs=11,
                         lr=.1, iterations=137, compiled_loop=compiled)
    reference = jax.vmap(lambda y: solve(y, False))(targets)
    result = jax.jit(jax.vmap(lambda y: solve(y, True)))(targets)
    np.testing.assert_allclose(np.asarray(result), np.asarray(reference), rtol=2e-5, atol=3e-5)


def test_sdd_initial_solution_matches_explicit_residual_equation():
    rng = np.random.default_rng(23)
    x = jax.numpy.asarray(rng.uniform(-.2,.2,(29,3)).astype('f'))
    y = jax.numpy.asarray(rng.normal(size=(29,6)).astype('f'))
    initial = jax.numpy.asarray(rng.normal(size=y.shape).astype('f'))
    k = Matern32Kernel(.05,.1)
    kernel = ProductKernel(k,k,k)
    matrix = jax.vmap(jax.vmap(kernel,(None,0)),(0,None))(x,x)+.02*jax.numpy.eye(len(x))
    residual = y-matrix@initial
    direct = StochasticDualDescent(x,y,kernel,batch=11,lr=.1,seed=19,initial=initial)
    shifted = StochasticDualDescent(x,residual,kernel,batch=11,lr=.1,seed=19)
    direct.advance(70)
    result = direct.advance(67)
    expected = initial+shifted.advance(137)
    np.testing.assert_allclose(np.asarray(result),np.asarray(expected),rtol=3e-5,atol=4e-5)
