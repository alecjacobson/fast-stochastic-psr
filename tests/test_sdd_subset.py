import jax
import jax.numpy as jnp
import numpy as np

from geospsr.iterative import rejection_subset


def test_rejection_minibatches_are_distinct_reproducible_and_label_symmetric():
    n, batch, count = 97, 8, 4096
    keys = jax.random.split(jax.random.key(219), count)
    draw = jax.jit(jax.vmap(lambda key: rejection_subset(key, n, batch)))
    samples = np.asarray(draw(keys))
    np.testing.assert_array_equal(samples, np.asarray(draw(keys)))
    assert np.all((samples >= 0) & (samples < n))
    assert np.all(np.diff(np.sort(samples, axis=1), axis=1) > 0)
    # An independent distribution check catches sorted-prefix or label bias;
    # deterministic seed and a broad threshold avoid fragile statistical tests.
    counts = np.bincount(samples.ravel(), minlength=n)
    expected = count*batch/n
    assert np.sum((counts-expected)**2/expected) < 2*n


def test_dense_minibatches_fall_back_to_original_permutation():
    key = jax.random.key(11)
    np.testing.assert_array_equal(np.asarray(rejection_subset(key, 13, 9)),
                                  np.asarray(jax.random.choice(key, 13, shape=(9,), replace=False)))
