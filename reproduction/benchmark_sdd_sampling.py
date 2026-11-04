"""Isolate compiled minibatch-sampling cost; this is not an SDD solve benchmark."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp

from geospsr.iterative import rejection_subset


def main(args):
    records = []
    for n in args.n:
        for name in ['permutation', 'rejection']:
            def run(key):
                def step(i, state):
                    key, checksum = state
                    key = jax.random.fold_in(key, i)
                    ids = (jax.random.choice(key, n, shape=(args.batch,), replace=False)
                           if name == 'permutation' else rejection_subset(key, n, args.batch))
                    return key, checksum+jnp.sum(ids)
                return jax.lax.fori_loop(0, args.iterations, step, (key, jnp.int32(0)))[1]
            compiled = jax.jit(run)
            compiled(jax.random.key(11)).block_until_ready()
            times = []
            for seed in [12, 13, 14]:
                start = perf_counter()
                value = compiled(jax.random.key(seed))
                value.block_until_ready()
                times.append(perf_counter()-start)
            records.append(dict(n=n, batch=args.batch, sampler=name, iterations=args.iterations, seconds=times))
    result = dict(scope='Compiled sampler microbenchmark only; different streams, same target subset distribution',
                  device=str(jax.devices()[0]), records=records)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n', nargs='+', type=int, default=[196608, 796825])
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--iterations', type=int, default=128)
    p.add_argument('--output', required=True)
    main(p.parse_args())
