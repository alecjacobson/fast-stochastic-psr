"""Selected-mode direct-sum diagnosis after the first large planar mismatch."""
import json
from pathlib import Path
import cupy as cp
import numpy as np
from geospsr.spectral import FourierPoisson
from reproduction.planar_fourier import PlanarFourierPlan


def main():
    root=Path(__file__).resolve().parents[1];q=np.load(root/'reproduction/results/matched/B/input.npz')['query']
    w=cp.random.RandomState(43).standard_normal((16,len(q)),dtype='float32')
    rng=np.random.default_rng(43);ns=np.concatenate([rng.integers(-50,51,(64,3)),np.array([[0,0,0],[-50,0,0],[0,-50,0],[0,0,-50],[50,50,50]])])
    truth=w.astype('complex128')@cp.exp(-1j*cp.asarray(q,dtype='d')@cp.asarray(ns,dtype='d').T)
    results={}
    for name,dtype,up,planar in [('generic32','float32',1.25,False),('planar32','float32',1.25,True),('planar32_up2','float32',2.,True),('planar64','float64',2.,True)]:
        op=FourierPoisson(50,dtype=dtype,eps=1e-12 if dtype=='float64' else 1e-6,upsampfac=up)
        plan=PlanarFourierPlan(1,op,q,16) if planar else op.plan(1,q,16)
        out=plan.execute(cp.asarray(w,dtype=op.cdtype));values=out[:,ns[:,0]+50,ns[:,1]+50,ns[:,2]+50]
        error=cp.asnumpy(cp.linalg.norm(values-truth,axis=0)/cp.maximum(cp.linalg.norm(truth,axis=0),1e-300))
        results[name]={'relative_l2':float(cp.linalg.norm(values-truth)/cp.linalg.norm(truth)),
                       'max_mode_relative_l2':float(error.max()),'mode_relative_l2':error.tolist()}
        del op,plan,out,values
    p=root/'reproduction/results/planar_queries/precision_diagnosis.json';p.write_text(json.dumps({'modes':ns.tolist(),'methods':results},indent=2)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='mode_relative_l2'} for k,v in results.items()},indent=2))


if __name__=='__main__':main()
