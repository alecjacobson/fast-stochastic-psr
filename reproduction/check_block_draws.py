"""Production-band sampler smoke check with independent direct residual rows.

This checks executable sampling and conditioning, not Monte Carlo convergence.
Small-model prior/posterior law checks live in tests/test_block_spectral.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cupy as cp
import numpy as np
from reproduction.block_posterior import BlockPosterior
from reproduction.block_reference import block_covariance_rows
from reproduction.gpu_audit import GpuProcessAudit

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case',choices=['A','B'],default='B')
    p.add_argument('--core',type=int,default=150)
    p.add_argument('--extent',type=int,default=800)
    p.add_argument('--lengthscale',type=float,default=.005)
    p.add_argument('--noise',type=float,default=.0004)
    p.add_argument('--draws',type=int,default=4)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    data = np.load(ROOT/f'reproduction/results/matched/{a.case}/input.npz')
    panel = np.load(ROOT/f'reproduction/results/spectral_tail/{a.case}/fields.npz')
    x,y,q = data['points'].astype('f'),data['normals'].astype('f'),panel['query'].astype('f')
    ids = np.sort(np.random.default_rng(1271).choice(len(x),min(512,len(x)),replace=False))
    record = {'args':{**vars(a),'output':str(a.output)},'complete':False,
              'scope':'production-band sampling/finite-model residual smoke test; not uncertainty convergence or competitive timing',
              'seed':51,'row_seed':1271,'normalization':'analytic','N':len(x),
              'hashes':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in [('points',x),('normals',y),('queries',q)]},
              'seconds':{},'device_used_bytes_at_stage':{}}
    def save():
        (a.output/'results.json').write_text(json.dumps(record,indent=2)+'\n')
    def timed(name,fn):
        cp.cuda.runtime.deviceSynchronize();begin=time.perf_counter()
        result=fn();cp.cuda.runtime.deviceSynchronize()
        record['seconds'][name]=time.perf_counter()-begin
        free,total = cp.cuda.runtime.memGetInfo()
        record['device_used_bytes_at_stage'][name]=total-free
        print(name,record['seconds'][name],flush=True);save()
        return result
    audit = GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    try:
        model = timed('fit',lambda:BlockPosterior(x,y,core=a.core,extent=a.extent,lengthscale=a.lengthscale,noise_variance=a.noise))
        timed('draw',lambda:model.draw(a.draws,seed=51,batch=1,retain_components=True,retain_rhs=True))
        rows = timed('independent_rows',lambda:block_covariance_rows(x[ids],x,a.core,a.extent,a.lengthscale))
        weights = model.sample_weights.transpose(1,0,2).reshape(len(x),-1).astype('d')
        rhs = model.sample_rhs[:,ids].transpose(1,0,2).reshape(len(ids),-1).astype('d')
        residual = rows@weights+a.noise*weights[ids]-rhs
        relative = cp.linalg.norm(residual,axis=0)/cp.maximum(cp.linalg.norm(rhs,axis=0),1e-30)
        record['finite_residual_per_component'] = cp.asnumpy(relative).tolist()
        record['residual_gate'] = 2e-5
        record['residual_gate_pass'] = bool(cp.all(relative<=record['residual_gate']))
        record['sample_solves'] = model.sample_solve_records
        fields = model.samples(q)
        np.savez_compressed(a.output/'fields.npz',query=q,mean=cp.asnumpy(model.mean(q)),samples=cp.asnumpy(fields),validation_ids=ids)
        if not record['residual_gate_pass']:
            raise RuntimeError('independent finite residual gate failed')
        record['complete'] = True
    except Exception as exc:
        record['failure'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        record['gpu_process_audit'] = audit.stop();save()


if __name__=='__main__':
    main()
