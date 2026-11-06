"""Retry strict validation PCG with a recurrence stopping safety margin.

The requested true-residual tolerance stays unchanged. Primary attempts and
their logs remain intact, including roundoff-level stopping failures.
"""
import json
from pathlib import Path
import runpy
import sys
import cupy as cp
from geospsr.hierarchical import batched_pcg
from geospsr.hierarchical_gpu import GPUHierarchicalSystem
from reproduction.run_matched_lengthscales import digest


def solve_with_margin(self,rhs,tolerance=1e-6,max_iterations=500):
    x,info=batched_pcg(self.matmul,self.precondition,rhs,tolerance=tolerance*.25,
                     max_iterations=max_iterations,xp=cp)
    info['recurrence_stopping_tolerance']=tolerance*.25
    info['tolerance']=tolerance
    info['converged']=bool(max(info['relative_residuals'])<=tolerance)
    self.last_solve=info
    if not info['converged']:raise RuntimeError('Requested true-residual tolerance not reached: '+str(info))
    return cp.asnumpy(x)


if __name__=='__main__':
    out=Path(sys.argv[sys.argv.index('--output')+1])
    GPUHierarchicalSystem.solve=solve_with_margin
    try:runpy.run_module('reproduction.noise_selection',run_name='__main__')
    finally:
        if out.exists():
            d=json.loads(out.read_text());d['source_sha256'][__file__]=digest(__file__)
            d['validation_solve_control']='Quarter-tolerance recurrence stopping, unchanged true-residual acceptance'
            out.write_text(json.dumps(d,indent=2)+'\n')
