"""Compare scalar and warp CSR fill on a real full-scan streaming chunk."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cupy as cp
import numpy as np

from geospsr.nonperiodic import FreeSpacePoissonMatern
from geospsr.nonperiodic_local_gpu import GPUStreamedLocalSystem, _FILL_SCALAR
from reproduction.poissonrecon_data import load_cloud


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--point-cloud', required=True)
    p.add_argument('--n', type=int, default=796825)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    x, _, metadata = load_cloud(a.point_cloud, a.n)
    model = FreeSpacePoissonMatern(.03, .0025)
    s = GPUStreamedLocalSystem(model, x, .000025, cutoff=16,
                              rank=50, neighbors=40, preconditioner_backend='prefix-kd')
    first, last, nnz, ptr = s.chunks[0]
    scalar = cp.RawKernel(_FILL_SCALAR, 'fill_local')
    scalar.compile()
    args = (s.coordinates, s.ids, s.offsets, s.ny, s.nz, s.nx,
            s.radius**2, first, last, ptr, s.columns, s.values,
            model.kappa, model.amplitude, s.noise_variance)
    timings = {}
    expected = None
    for name, kernel, blocks in [('scalar', scalar, (last-first+127)//128),
                                  ('warp', s.kernel, (last-first+3)//4)]:
        elapsed = []
        for _ in range(4):
            cp.cuda.get_current_stream().synchronize()
            start = perf_counter()
            kernel((blocks,), (128,), args)
            cp.cuda.get_current_stream().synchronize()
            elapsed.append(perf_counter()-start)
        timings[name] = elapsed
        if expected is None:
            expected = cp.asnumpy(s.columns[:nnz]), cp.asnumpy(s.values[:nnz])
        else:
            np.testing.assert_array_equal(cp.asnumpy(s.columns[:nnz]), expected[0])
            np.testing.assert_array_equal(cp.asnumpy(s.values[:nnz]), expected[1])
    Path(a.output).write_text(json.dumps(dict(n=a.n, point_cloud_sha256=metadata['normalized_sha256'],
                         rows=last-first, nnz=nnz, seconds=timings,
                         exact_columns_and_values=True,
                         scope='first chunk fill only; excludes setup, SpMM and solves'), indent=2)+'\n')


if __name__ == '__main__':
    main()
