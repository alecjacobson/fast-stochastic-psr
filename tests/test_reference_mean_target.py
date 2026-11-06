"""A tight mean target must refine beyond probability/variance-only stopping."""
import json
import numpy as np

import reproduction.audit_hierarchical_variance as audit_module
from geospsr.nonperiodic import FreeSpacePoissonMatern


def test_absolute_mean_target_forces_reference_refinement(tmp_path,monkeypatch):
    class ApproximateSolve:
        dtype=np.dtype('float64')

        def __init__(self,model,x,noise,dtype):
            self.matrix=model.vector_covariance(x,x)+noise*np.eye(len(x))

        def solve(self,rhs):
            # Deliberately inexact initializer, with a predictable contraction.
            return .95*np.linalg.solve(self.matrix,rhs)

    def action(model,x,noise,z):
        return (model.vector_covariance(x,x)+noise*np.eye(len(x)))@z

    monkeypatch.setattr(audit_module,'GPUDenseMaternSystem',ApproximateSolve)
    monkeypatch.setattr(audit_module,'full_kernel_matmul',action)
    model=FreeSpacePoissonMatern(.96,1/3)
    args=dict(n=8,seed=111,geometry='ellipsoid',lengthscale=.96,amplitude=1/3,
              relative_noise=.12,halfwidth=.005)
    field=dict(center=[1,0,0],halfwidth=[.005]*3,order=2,prior_variance=model.prior_variance,
               mean=np.zeros((2,2,2)).tolist(),correction=np.zeros((2,2,2)).tolist())
    record=tmp_path/'record.json';record.write_text(json.dumps(dict(arguments=args,field=field,level=0)))
    normal=tmp_path/'normal.json';strict=tmp_path/'strict.json'
    audit_module.audit(record,normal,random_queries=0)
    audit_module.audit(record,strict,random_queries=0,reference_mean_bound=1e-12)
    before=json.loads(normal.read_text());after=json.loads(strict.read_text())
    assert max(before['mean_error_bound'])>1e-12
    assert max(after['mean_error_bound'])<=1e-12
    assert after['refinement']['steps']>before['refinement']['steps']
    assert after['contract']['reference_mean_bound']==1e-12
