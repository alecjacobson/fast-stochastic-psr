"""Resolve the old cached-locality/new fields' worst full-query discrepancy."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
from cupyx.scipy.special import ndtr
import numpy as np

from reproduction.audit_hierarchical_variance import audit
from reproduction.benchmark_hierarchical_variance import read_field,query_batches


def main(root):
    root=Path(root);out=root/'targeted';out.mkdir(exist_ok=True)
    old=Path('reproduction/results/poissonrecon_large/eagle/cached-control-15/sparse-gpu-196608-0.json')
    paths=[root/'free-space/h2-streamed-196608.json',root/'free-space/sparse-streamed-196608.json',old]
    data=[json.loads(p.read_text()) for p in paths];fields=[read_field(d['field']) for d in data]
    worst=[(-1.,None),(-1.,None)]
    for q in query_batches(SimpleNamespace(**data[0]['arguments'])):
        probabilities=[]
        for f,d in zip(fields,data):
            m,v=f.predict_gpu(q);probabilities.append(ndtr((d['level']-m)/cp.sqrt(v)))
        for i in range(2):
            error=abs(probabilities[i]-probabilities[-1]);idx=int(cp.argmax(error));value=float(error[idx])
            if value>worst[i][0]:worst[i]=(value,q[idx].tolist())
    q=[v[1] for v in worst]+[[1.,0,0]]
    (out/'queries.json').write_text(json.dumps(q,indent=2)+'\n')
    del fields,probabilities,m,v,error;cp.get_default_memory_pool().free_all_blocks()
    for i,path in enumerate(paths):
        audit(str(path),str(out/'reference.json'),reference_preconditioner='prefix-kd',
              queries=q,result_path=str(out/f'audit-{i}.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='reproduction/results/asymptotic_regimes')
    main(p.parse_args().root)
