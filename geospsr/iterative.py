"""Compiled stochastic dual descent with resumable checkpoints.

The update algebra and sequential fold_in/random.choice sequence are unchanged.
Noise is accepted as variance here; upstream must receive its square root.
This is an optimized reference, not the untouched Python-loop implementation.
"""
import jax
import jax.numpy as jnp


class StochasticDualDescent:
    def __init__(self, x, y, kernel, noise_variance=.02, batch=128, lr=.001,
                 momentum=.9, polyak=.01, seed=0, initial=None):
        self.x, self.y = jnp.asarray(x), jnp.asarray(y)
        n = len(x)
        if not 1 <= batch <= n:
            raise ValueError('batch must be between one and the observation count')
        zero = jnp.zeros_like(self.y)
        initial = zero if initial is None else jnp.asarray(initial,dtype=self.y.dtype)
        if initial.shape != self.y.shape:
            raise ValueError('initial solution must have the same shape as y')
        # Starting the iterate AND its Polyak average at initial is equivalent
        # to solving for a correction to that fixed solution, with zero velocity.
        self.state = (jnp.asarray(0, dtype=jnp.int32), jax.random.key(seed), initial, initial, zero)

        def advance(state, x, y, steps):
            def step(_, state):
                iteration, key, alpha, average, velocity = state
                key = jax.random.fold_in(key, iteration)
                ids = jax.random.choice(key, n, shape=(batch,), replace=False)
                rows = jax.vmap(jax.vmap(kernel, (None, 0)), (0, None))(x[ids], x)
                gradient = jnp.zeros_like(y).at[ids].set(rows @ alpha - y[ids] + noise_variance * alpha[ids])
                velocity = momentum * velocity - lr * (n / batch) * gradient
                alpha = alpha + velocity
                average = polyak * alpha + (1 - polyak) * average
                return iteration + 1, key, alpha, average, velocity
            return jax.lax.fori_loop(0, steps, step, state)

        self.advance_jit = jax.jit(advance)
        self.compiled = None

    def compile(self):
        self.compiled = self.advance_jit.lower(self.state, self.x, self.y, 1).compile()

    def advance(self, steps):
        if self.compiled is None:
            self.compile()
        self.state = self.compiled(self.state, self.x, self.y, steps)
        self.state[3].block_until_ready()
        return self.state[3]

    @property
    def iterations(self):
        return int(self.state[0])
