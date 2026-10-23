"""Compare identical half-basis applications; this is not a posterior benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cupy as cp
import numpy as np
from geospsr.nystrom import LandmarkNystrom
from geospsr.compact_basis import fused_half_apply
from reproduction.benchmark_nufft_batches import measure
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit
from reproduction.pool_memory import PoolLivePeak


def double_reference(pre,rhs):
    """Blocked float64 action of exactly the stored basis and damping."""
    projection=cp.zeros((pre.rank,rhs.shape[1]),dtype='d')
    for i in range(0,pre.n,32768):
        u=pre.basis[i:i+32768].astype('d')/pre.basis_scale
        projection+=u.T@rhs[i:i+32768].astype('d')
    projection*=pre.damping.astype('d')[:,None]
    result=cp.empty(rhs.shape,dtype='d')
    for i in range(0,pre.n,32768):
        u=pre.basis[i:i+32768].astype('d')/pre.basis_scale
        result[i:i+32768]=rhs[i:i+32768].astype('d')-u@projection
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--points',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=7)
    a=p.parse_args()
    if a.repeats<1:p.error('require positive repeats')
    a.output.mkdir(parents=True,exist_ok=True)
    record={'complete':False,'scope':'same stored preconditioner action only; no posterior or upstream timing ratio',
            'args':{**vars(a),'points':str(a.points),'output':str(a.output)},'seconds':{},'pool_live_memory':{}}
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    def timed(name,fn):
        hook=PoolLivePeak()
        with hook:result,seconds=measure(fn)
        record['seconds'][name]=seconds;record['pool_live_memory'][name]=hook.record()
        return result
    try:
        x=np.load(a.points,allow_pickle=False);record['N']=len(x)
        record['points_sha256']=array_digest(x)
        record['kernel_source_sha256']=hashlib.sha256((Path(__file__).resolve().parents[1]/'geospsr/compact_basis.py').read_bytes()).hexdigest()
        pre=timed('setup',lambda:LandmarkNystrom(x,.02,rank=1024,basis_dtype='float16',recompute_features=True,apply_row_batch=32768))
        record['preconditioner']=pre.record
        cp.get_default_memory_pool().free_all_blocks()
        rng=cp.random.RandomState(717)
        rhs=rng.standard_normal((len(x),6),dtype='f')*cp.asarray(np.geomspace(1e-5,1e5,6).astype('f'))[None]
        rhs[:,1]=0
        # Exercise cancellation in a coarse-space direction, not only random
        # RHS that mostly lie outside the low-rank subspace.
        rhs[:,-1]=pre.basis[:,-1].astype('f')/pre.basis_scale
        reference=timed('double_reference',lambda:double_reference(pre,rhs))
        functions={'decoded':lambda:pre(rhs),'fused':lambda:fused_half_apply(pre.basis,pre.damping,rhs,pre.basis_scale)}
        record['quality']={};record['samples']={k:[] for k in functions}
        for name,fn in functions.items():
            result=timed(name+'_first_use',fn)
            error=cp.linalg.norm(result.astype('d')-reference,axis=0)
            input_norm=cp.maximum(cp.linalg.norm(rhs.astype('d'),axis=0),1e-30)
            output_norm=cp.maximum(cp.linalg.norm(reference,axis=0),1e-30)
            record['quality'][name]={'error_over_input':cp.asnumpy(error/input_norm).tolist(),
                                     'relative_output_error':cp.asnumpy(error/output_norm).tolist(),
                                     'pass':bool(cp.all(error/input_norm<5e-6)),
                                     'zero_channel_exact':bool(cp.all(result[:,1]==0))}
        for repeat in range(a.repeats):
            for name in (['decoded','fused'] if repeat%2==0 else ['fused','decoded']):
                result=timed(f'{name}_{repeat}',functions[name])
                record['samples'][name].append(record['seconds'][f'{name}_{repeat}'])
        record['medians']={k:float(np.median(v)) for k,v in record['samples'].items()}
        if not all(v['pass'] and v['zero_channel_exact'] for v in record['quality'].values()):
            raise RuntimeError('preconditioner application accuracy gate failed')
        if not all(v['end_state_matches'] for v in record['pool_live_memory'].values()):
            raise RuntimeError('pool memory event accounting did not reconcile')
        record['complete']=True
    except Exception as exc:
        record['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:
        record['gpu_process_audit']=audit.stop()
        (a.output/'results.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ['gpu_process_audit','pool_live_memory']}))


if __name__=='__main__':main()
