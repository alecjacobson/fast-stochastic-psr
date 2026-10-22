"""Validate the matched reproduction against literal APIs and Fourier sums."""
from functools import partial
import json
from pathlib import Path
import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np
from reproduction.matched_teaser import original, Matern32Kernel, ProductKernel, trunc_Zd
from geospsr.spectral import FourierPoisson

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reproduction/results/matched'


def main():
    data = np.load(OUT / 'A/input.npz')
    rng = np.random.default_rng(33)
    diff = (data['query'][rng.choice(len(data['query']), 64)]
            - data['points'][rng.choice(len(data['points']), 64)])
    op = FourierPoisson()
    freq = trunc_Zd(50)
    k = Matern32Kernel(.01, .1)
    kv = ProductKernel(k, k, k)
    ev = jax.vmap(kv.spectral_density)(freq) ** .5
    fn = jax.jit(jax.vmap(partial(original.poisson_cross_covariances,
                                eigenvectors=freq, eigenvalues=ev)))
    reference = np.concatenate([np.asarray(fn(jnp.asarray(diff[i:i + 8]))) for i in range(0, len(diff), 8)])
    fast = cp.asnumpy(op.cross_values(diff))
    direct = {'pairs': len(diff), 'relative_l2': float(np.linalg.norm(fast - reference) / np.linalg.norm(reference)),
              'max_abs': float(np.max(np.abs(fast - reference)))}
    assert direct['relative_l2'] < 1e-4, direct
    (OUT / 'direct_fourier_validation.json').write_text(json.dumps(direct, indent=2))
    for case in ['A', 'B']:
        out = OUT / case
        meta = json.loads((out / 'original.json').read_text())
        comp = json.loads((out / 'comparison.json').read_text())
        matched = json.loads((out / 'compatible.json').read_text())
        for key in ['points_sha256', 'normals_sha256', 'queries_sha256']:
            assert meta[key] == matched[key], (case, key)
        assert all(v['relative_l2'] < 3e-6 for v in meta['cached_vs_literal_api'].values())
        assert comp['nufft_backend'] == 'compatible'
        assert comp['fields']['mean']['relative_l2'] < .002
        assert comp['fields']['variance']['relative_l2'] < .0002
        assert comp['fields']['probability']['max_abs'] < .02
        assert comp['fields']['probability']['rms_abs'] < .001
        print(case, 'matched fields, literal API agreement, and input hashes passed')
    print('Direct Fourier validation', direct)


if __name__ == '__main__':
    main()
