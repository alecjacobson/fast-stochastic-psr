"""Regenerate old common-RHS data without rewriting it; audit storage casts."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import cupy as cp
import numpy as np
from geospsr.spectral import FourierPoisson
from geospsr.training_spectrum import FourierCovariance
from reproduction.benchmark_nufft_batches import prior_rhs


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scratch',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();manifest=json.loads((a.scratch/'manifest.json').read_text())
    x=np.load(a.scratch/'points.npy');y=np.load(a.scratch/'normals.npy')
    source=np.load(a.scratch/'rhs.npy',mmap_mode='r');priors=np.load(a.scratch/'prior.npy',mmap_mode='r')
    record={'complete':False,'source_manifest_sha256':hashlib.sha256((a.scratch/'manifest.json').read_bytes()).hexdigest(),
            'scope':'read-only regeneration of legacy full-cube200/scalar50 common RHS and prior arrays',
            'source_hashes':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in [('points',x),('normals',y)]}}
    def save():
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(record,indent=2)+'\n')
    try:
        if not manifest['complete'] or len(source)!=manifest['args']['samples'] or len(priors)!=len(source):
            raise ValueError('incomplete source')
        if any(record['source_hashes'][k]!=manifest['hashes'][k] for k in record['source_hashes']):
            raise ValueError('points or normals differ from preparation')
        training=FourierCovariance(200);op=FourierPoisson(50)
        model=SimpleNamespace(training=training,op=op,normals=cp.asarray(y),noise_variance=.02,
                              K=SimpleNamespace(backward=training.plan(2,x,3)))
        rng=cp.random.RandomState(manifest['args']['seed'])
        root=cp.sqrt(training.rho);denominator=cp.maximum((op.ns**2).sum(0),1)
        before=hashlib.sha256();stored=hashlib.sha256();prior_hash=hashlib.sha256()
        record['draws']=[]
        for i in range(len(source)):
            rhs,prior=prior_rhs(model,rng,root,denominator)
            host=cp.asnumpy(rhs);prior_host=cp.asnumpy(prior)
            before.update(host.tobytes())
            cast=host.astype(source.dtype);stored.update(cast.tobytes());prior_hash.update(prior_host.tobytes())
            item={'index':i,'rhs_exact_after_storage_cast':bool(np.array_equal(cast,source[i])),
                  'prior_exact':bool(np.array_equal(prior_host,priors[i]))}
            record['draws'].append(item)
            if not item['rhs_exact_after_storage_cast'] or not item['prior_exact']:
                raise RuntimeError('regenerated data differs from stored arrays')
            if (i+1)%8==0: print('verified',i+1,flush=True)
        record.update(raw_rhs_dtype=host.dtype.str,stored_rhs_dtype=source.dtype.str,
                      raw_rhs_sha256=before.hexdigest(),stored_rhs_sha256=stored.hexdigest(),prior_sha256=prior_hash.hexdigest())
        record['declared_matches_raw_rhs']=before.hexdigest()==manifest['hashes']['all_rhs']
        record['declared_matches_stored_rhs']=stored.hexdigest()==manifest['hashes']['all_rhs']
        if not record['declared_matches_raw_rhs'] and not record['declared_matches_stored_rhs']:
            raise RuntimeError('neither regenerated representation matches manifest')
        record['complete']=True
    except Exception as exc:
        record['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:save()
    print(json.dumps({k:v for k,v in record.items() if k!='draws'}))


if __name__=='__main__':main()
