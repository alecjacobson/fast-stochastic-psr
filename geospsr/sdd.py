import jax.numpy as jnp
from jax import jit, random, vmap
from tqdm.auto import tqdm


def sdd_solve(
    x_data,
    v_data,
    kernel_v,
    sigma,
    key,
    bs=128,
    lr=0.001,
    momentum=0.9,
    polyak=1e-2,
    iterations=1500,
    verbose=None,
    compiled_loop=False,
):
    """
    Implementation of Stochastic Dual Descent adapted from
    https://github.com/cambridge-mlg/sgd-gp

    sigma is the observation noise variance, as in the Cholesky solver.
    compiled_loop=True compiles the unchanged update sequence as one JAX loop.
    It supports the existing vmap-based GP sampling/variance calls; progress
    bars are available only in the default Python-loop mode.
    """
    if compiled_loop:
        from .iterative import StochasticDualDescent
        solver = StochasticDualDescent(x_data, v_data, kernel_v, sigma, bs,
                                       lr, momentum, polyak)
        # Preserve the caller's random key and stay traceable under vmap/jit.
        solver.state = (solver.state[0], key, *solver.state[2:])
        return solver.advance_jit(solver.state, solver.x, solver.y, iterations)[3]
    num_data = x_data.shape[0]
    dim = v_data.shape[1]

    alpha = jnp.zeros((num_data, dim))
    alpha_polyak = jnp.zeros((num_data, dim))

    v = jnp.zeros((num_data, dim))

    @jit
    def gr(params, idx):
        K_batch = vmap(vmap(kernel_v, in_axes=(None, 0)), in_axes=(0, None))(
            x_data[idx], x_data
        )
        grad = jnp.zeros((num_data, dim))
        grad = grad.at[idx].set(
            K_batch @ params - v_data[idx] + sigma * params[idx]
        )
        return (num_data / bs) * grad

    @jit
    def update(params, params_polyak, velocity, idx):
        grad = gr(params, idx)
        velocity = momentum * velocity - lr * grad
        params = params + velocity
        params_polyak = polyak * params + (1.0 - polyak) * params_polyak
        return params, params_polyak, velocity

    # TODO: Replace with scan for faster jit?
    for i in tqdm(range(iterations), disable=not verbose, desc="SDD"):
        key = random.fold_in(key, i)
        idx = random.choice(key, num_data, shape=(bs,), replace=False)
        alpha, alpha_polyak, v = update(alpha, alpha_polyak, v, idx)

    return alpha_polyak
