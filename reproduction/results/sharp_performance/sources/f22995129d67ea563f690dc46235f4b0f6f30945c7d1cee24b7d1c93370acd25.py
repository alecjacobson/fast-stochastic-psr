"""Fresh performance measurements for the frozen PR127/129 posterior."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import ndtr

from geospsr.nonperiodic import FreeSpacePoissonMatern
from reproduction.partial_occupancy import ROOT, load
from reproduction.partial_competitor_common import gates, store_result
from reproduction.stanford_partial_data import digest

OUT=Path('reproduction/results/sharp_performance')
FROZEN=Path('reproduction/results/observed_fidelity')


def snapshot():
    folder=OUT/'sources';folder.mkdir(exist_ok=True);result={}
    paths=[__file__,'reproduction/partial_competitor_common.py','reproduction/partial_direct_gpu.py',
           'reproduction/observed_fidelity_large.py','reproduction/partial_quality_large.py',
           'reproduction/gpu_audit.py','reproduction/partial_occupancy.py',
           'geospsr/nonperiodic.py','geospsr/nonperiodic_dense_gpu.py','geospsr/scalar_contacts.py',
           'geospsr/hierarchical.py','geospsr/hierarchical_gpu.py','geospsr/hierarchical_interactions.py',
           'geospsr/nonperiodic_local_gpu.py','geospsr/nonperiodic_dense_cg.py']
    for path in paths:
        sha=digest(path);(folder/(sha+'.py')).write_bytes(Path(path).read_bytes());result[path]=sha
    return result


def problem(n,group='test_side',validation_panel=False):
    cfg=json.loads((FROZEN/'selected.json').read_text())['configuration']
    x,y,_,old=load(group,n);panel=np.load(FROZEN/(group+'-panel.npz'))
    source=panel['queries'] if validation_panel else old['queries']
    ids=np.linspace(0,len(source)-1,128,dtype=int);q=source[ids]
    c=panel['contacts'][:cfg['contacts']];model=FreeSpacePoissonMatern(cfg['ell'],1/3)
    contract=dict(n=n,group=group,configuration=cfg,queries=128,query_indices=ids.tolist(),validation_panel=validation_panel,
                  selected_sha256=digest(FROZEN/'selected.json'),panel_sha256=digest(FROZEN/(group+'-panel.npz')),
                  dataset_sha256=digest(ROOT/'dataset.json'),query_sha256=digest(ROOT/(group+'-queries.npz')),
                  protocol_sha256=digest(OUT/'PROTOCOL.md'))
    return x,y,c,q,model,cfg,contract


def reference(args):
    import cupy as cp
    from reproduction.observed_fidelity_large import StreamingReference
    from reproduction.gpu_audit import GpuProcessAudit
    path=Path(args.output)
    if path.exists():raise FileExistsError(path)
    x,y,c,q,model,cfg,contract=problem(args.n,args.group,args.validation_panel)
    check=args.n==70652 and args.group=='test_side'
    if check:
        oldpanel=np.load(FROZEN/'test_side-panel.npz');ids=np.linspace(0,511,32,dtype=int)
        q=np.concatenate((q,oldpanel['queries'][ids]))
    record=dict(arguments=vars(args),contract=contract,source_sha256=snapshot(),scope='Independent full-kernel reference, excluded from competitive timings')
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id,interval=.5,cuda_device=cp.cuda.Device().id).start()
    start=perf_counter();m=v=None
    try:
        posterior=StreamingReference(model,x,y,c,cfg['sigma']**2,cfg['scalar_variance'],block=192)
        m,v=posterior.predict(q)
        assert np.isfinite(m).all() and np.isfinite(v).all() and np.all(v>0)
        record['refinement_history']=posterior.system.histories
        if check:
            old=np.load(FROZEN/'test-side-selected.npz')
            error=dict(mean_max=float(np.max(abs(m[128:]-old['mean'][ids]))),
                       variance_over_prior_max=float(np.max(abs(v[128:]-old['variance'][ids]))/model.prior_variance))
            assert error['mean_max']<=1e-9 and error['variance_over_prior_max']<=1e-7
            record['retained_agreement']=error
        m,v=m[:128],v[:128];record['status']='completed'
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        record.update(seconds=perf_counter()-start,gpu_audit=audit.stop())
        record['accepted']=record['status']=='completed' and record['gpu_audit']['sampled_isolation_pass']
        store_result(path,record,m,v)
    print('REFERENCE',args.n,record['seconds'],record['accepted'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=['reference'],required=True)
    p.add_argument('--n',type=int,default=70652);p.add_argument('--group',default='test_side')
    p.add_argument('--validation-panel',action='store_true');p.add_argument('--output',required=True)
    a=p.parse_args();reference(a)
