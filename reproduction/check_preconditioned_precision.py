"""Retain a small low-noise case where solver tolerance alone was insufficient."""
import argparse
import json
from pathlib import Path
import cupy as cp
import numpy as np
from reproduction.half_posterior import HalfCubePosterior
from reproduction.paired_applications import field_quality


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(222);x=rng.uniform(-.5,.5,(23,3)).astype('f');y=rng.normal(size=x.shape).astype('f')
    q=rng.uniform(-.5,.5,(11,3)).astype('f')
    records=[];arrays={'points':x,'normals':y,'query':q}
    for eps in [1e-6,1e-7]:
        for rtol in [1e-5,3e-6,1e-6]:
            for packed,initial in [(True,False),(False,True)]:
                item={'eps':eps,'rtol':rtol,'packed':packed,'initial_mean':initial,'complete':False}
                model=None
                try:
                    model=HalfCubePosterior(x,y,training_cutoff=5,cutoff=2,lengthscale=.12,noise_variance=.0001,rtol=rtol,eps=eps)
                    model.draw(5,seed=19,batch=2,pack_rhs=packed,initial_mean=initial)
                    baseline=model.samples(q);item['baseline_solves']=model.sample_solve_records
                    model.prepare_conditioning_preconditioner(rank=9,seed=41,row_batch=7)
                    model.draw(5,seed=19,batch=2,pack_rhs=packed,initial_mean=initial,preconditioned=True)
                    candidate=model.samples(q);item['preconditioned_solves']=model.sample_solve_records
                    item['fields']=field_quality(baseline,candidate)
                    arrays[f'baseline_{len(records)}']=cp.asnumpy(baseline)
                    arrays[f'candidate_{len(records)}']=cp.asnumpy(candidate)
                    item['complete']=True
                except Exception as exc:item['failure']=f'{type(exc).__name__}: {exc}'
                finally:
                    del model
                    cp.get_default_memory_pool().free_all_blocks()
                records.append(item)
                print(json.dumps({k:v for k,v in item.items() if k not in ['baseline_solves','preconditioned_solves']}),flush=True)
    (a.output/'results.json').write_text(json.dumps({'scope':'fixed low-noise float32 numerical stress case; unchanged field/covariance gates','cases':records},indent=2)+'\n')
    np.savez_compressed(a.output/'fields.npz',**arrays)


if __name__=='__main__':main()
