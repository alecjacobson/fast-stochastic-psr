"""Run a periodic benchmark/audit with one vector component per FFT plan.

The Fourier coefficients, cutoff and accuracy are unchanged. This optional
benchmark wrapper keeps the primary production sources and raw trials intact.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys

import cupy as cp
from geospsr.spectral import FourierPoisson
from reproduction.run_matched_lengthscales import digest


def cross_values_single(self, differences):
    coefficients=cp.ascontiguousarray(-1j*self.multiplier,dtype=self.cdtype)
    return self.evaluate_batched(differences,coefficients,batch=1).T


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--module',default='reproduction.benchmark_eagle_comparison',
                   choices=['reproduction.benchmark_eagle_comparison','reproduction.audit_periodic_scaling'])
    a,arguments=p.parse_known_args()
    FourierPoisson.cross_values=cross_values_single
    if a.module.endswith('benchmark_eagle_comparison'):
        out=Path(arguments[arguments.index('--output')+1])
    else:
        record=Path(arguments[0]);out=record.with_name(record.stem+'.audit.json')
    sys.argv=[a.module,*arguments]
    try:runpy.run_module(a.module,run_name='__main__')
    finally:
        if out.exists():
            d=json.loads(out.read_text())
            d['execution_control']=dict(cross_transform_batch=1,source=__file__,source_sha256=digest(__file__))
            out.write_text(json.dumps(d,indent=2)+'\n')


if __name__=='__main__':main()
