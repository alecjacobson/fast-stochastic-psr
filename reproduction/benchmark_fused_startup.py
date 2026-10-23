"""Measure cold fused-kernel compilation separately from warmed action timings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import cupy as cp
import numpy as np
from geospsr.compact_basis import fused_half_apply
from reproduction.benchmark_nufft_batches import measure


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cache=os.environ.get('CUPY_CACHE_DIR')
    if not cache:raise ValueError('set CUPY_CACHE_DIR to a new empty directory')
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    if any(cache.iterdir()):raise ValueError('cold startup diagnostic requires an empty cache')
    rng=np.random.default_rng(41);basis=cp.asarray((rng.normal(size=(257,17))*2).astype('float16'))
    record={'scope':'cold fused compile/first use on small input; excludes process/CUDA initialization',
            'cache_initially_empty':True,'seconds':{},
            'kernel_source_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'geospsr/compact_basis.py').read_bytes()).hexdigest()}
    for dtype in ['float32','float64']:
        rhs=cp.asarray(rng.normal(size=(257,6)).astype(dtype));damping=cp.asarray(np.linspace(0,.7,17).astype(dtype))
        fn=lambda:fused_half_apply(basis,damping,rhs,32)
        first,record['seconds'][dtype+'_cold']=measure(fn)
        again,record['seconds'][dtype+'_warm']=measure(fn)
        if not bool(cp.array_equal(first,again)):raise RuntimeError('startup replay changed output')
    record['complete']=True;a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))


if __name__=='__main__':main()
