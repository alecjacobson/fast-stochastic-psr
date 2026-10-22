"""Large-input folded posterior smoke check against independent finite rows.

This verifies production-band conditioning and execution, not MC convergence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import cupy as cp
import numpy as np
from geospsr.spectral_blocks import axis_weights
from geospsr.training_spectrum import finite_covariance_rows,FourierCovariance
from reproduction.half_posterior import HalfCubePosterior
from reproduction.gpu_audit import GpuProcessAudit

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['B','1000000'],default='1000000')
    p.add_argument('--draws',type=int,default=4)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.dataset=='B':
        data=np.load(ROOT/'reproduction/results/matched/B/input.npz')
        x,y=data['points'].astype('f'),data['normals'].astype('f')
    else:
        data=np.load(ROOT/'reproduction/results/teaser/cloud_1000000.npz')
        ids=np.random.default_rng(19).permutation(len(data['points']))
        x,y=data['points'][ids].astype('f'),data['normals'][ids].astype('f')
    q=np.load(ROOT/f'reproduction/results/spectral_tail/{a.dataset}/fields.npz')['query'].astype('f')
    ids=np.sort(np.random.default_rng(1271).choice(len(x),min(512,len(x)),replace=False))
    a.output.mkdir(parents=True,exist_ok=True)
    record={'args':{**vars(a),'output':str(a.output)},'complete':False,'N':len(x),'seconds':{},
            'training_cutoff':200,'scalar_cutoff':50,'lengthscale':.01,'noise_variance':.02,
            'seed':51,'row_seed':1271,'normalization':'analytic',
            'scope':'coherent finite-cube posterior smoke check; not sample convergence or a paired speed benchmark',
            'hashes':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in [('points',x),('normals',y),('queries',q)]}}
    def save():
        (a.output/'results.json').write_text(json.dumps(record,indent=2)+'\n')
    def timed(name,fn):
        cp.cuda.runtime.deviceSynchronize();begin=time.perf_counter()
        value=fn();cp.cuda.runtime.deviceSynchronize()
        record['seconds'][name]=time.perf_counter()-begin
        print(name,record['seconds'][name],flush=True);save()
        return value
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    try:
        model=timed('mean_fit',lambda:HalfCubePosterior(x,y))
        timed('draw',lambda:model.draw(a.draws,seed=51,batch=1,retain_components=True,retain_rhs=True))
        # Recreate the first canonical prior using a full 3D grid, without the
        # shifted positive-half or zero-plane evaluation. Identical Gaussian
        # coefficients here are a numerical control, not a seed-based pairing
        # assertion between independent samplers.
        rng=cp.random.RandomState(51)
        plane=rng.standard_normal((3,401,401),dtype='f').astype('F')
        plane.imag=-rng.standard_normal(plane.shape,dtype='f')
        plane*=cp.sqrt(model.training.plane_rho)[None]
        positive=rng.standard_normal((3,200,401,401),dtype='f').astype('F')
        positive.imag=-rng.standard_normal(positive.shape,dtype='f')
        positive*=cp.sqrt(2*model.training.positive_rho)[None]
        full=cp.zeros((3,401,401,401),dtype='F')
        full[:,200]=plane;full[:,201:]=positive
        del plane,positive
        spectrum=FourierCovariance(200)
        vector=timed('full_grid_prior_control',lambda:spectrum.evaluate(x,full).T)
        noise=rng.standard_normal(vector.shape,dtype='f')*np.sqrt(.02)
        rhs=model.normals-vector-noise
        record['paired_prior_rhs_relative_error']=float(cp.linalg.norm(rhs-model.sample_rhs[0])/cp.linalg.norm(rhs))
        low=full[:,150:251,150:251,150:251]
        denom=cp.maximum((model.op.ns**2).sum(0),1)
        scalar=-1j*(low*model.op.ns).sum(0)/denom
        record['paired_scalar_coefficients_relative_error']=float(cp.linalg.norm(scalar-model.sample_prior_coeff[0])/cp.linalg.norm(scalar))
        record['paired_prior_gate_pass']=max(record['paired_prior_rhs_relative_error'],record['paired_scalar_coefficients_relative_error'])<5e-5
        del full,low,spectrum,vector,noise,rhs,scalar
        cp.get_default_memory_pool().free_all_blocks()
        axis=SimpleNamespace(cutoff=200,axis64=cp.asarray(axis_weights(np.arange(-200,201))))
        weights=model.sample_weights.transpose(1,0,2).reshape(len(x),-1).astype('d')
        expected=model.sample_rhs[:,ids].transpose(1,0,2).reshape(len(ids),-1).astype('d')
        residuals=[]
        # Independent covariance rows stay bounded even at a million sources.
        for start in range(0,len(ids),64):
            subset=ids[start:start+64]
            rows=finite_covariance_rows(x[subset],x,axis,dtype='d')
            residuals.append(rows@weights+.02*weights[subset]-expected[start:start+len(subset)])
        residual=cp.concatenate(residuals)
        relative=cp.linalg.norm(residual,axis=0)/cp.maximum(cp.linalg.norm(expected,axis=0),1e-30)
        record['finite_residual_per_component']=cp.asnumpy(relative).tolist()
        record['residual_gate']=2e-5
        record['residual_gate_pass']=bool(cp.all(relative<=record['residual_gate']))
        record['sample_solves']=model.sample_solve_records
        fields=model.samples(q)
        np.savez_compressed(a.output/'fields.npz',query=q,mean=cp.asnumpy(model.mean(q)),samples=cp.asnumpy(fields),validation_ids=ids)
        if not record['residual_gate_pass'] or not record['paired_prior_gate_pass']:
            raise RuntimeError('independent finite residual gate failed')
        record['complete']=True
    except Exception as exc:
        record['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:
        record['gpu_process_audit']=audit.stop();save()


if __name__=='__main__':
    main()
