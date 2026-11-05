"""Explicit-query audits preserve their input records and contact gauge."""
import json
import numpy as np
import pytest
from reproduction.audit_hierarchical_variance import audit


def test_cached_targeted_audit_is_separate_and_query_bound(tmp_path):
    args=dict(n=8,seed=12,geometry='sphere',lengthscale=.3,amplitude=.0025,relative_noise=.01,halfwidth=.1)
    queries=[[.99,0,0],[1,0,0]]
    field=dict(center=[1,0,0],halfwidth=[.1]*3,prior_variance=1.,order=3,
               mean=np.zeros((3,)*3).tolist(),correction=np.zeros((3,)*3).tolist())
    record=tmp_path/'record.json';record.write_text(json.dumps(dict(arguments=args,field=field,level=0.)))
    before=record.read_bytes();reference=tmp_path/'reference.json';result=tmp_path/'audit.json'
    reference.write_text(json.dumps(dict(contract=dict(args,random_queries=32,custom_queries=queries),
        queries=queries,mean=[0,0],variance=[1,1],mean_error_bound=[0,0],variance_error_bound=[0,0],prior_variance=1.)))
    audit(record,reference,queries=queries,result_path=result)
    assert record.read_bytes()==before
    assert json.loads(result.read_text())['passes']
    with pytest.raises(ValueError,match='contract mismatch'):
        audit(record,reference,queries=[[.98,0,0],[1,0,0]],result_path=result)
    with pytest.raises(ValueError,match='contact last'):
        audit(record,reference,queries=[[.99,0,0]],result_path=result)
