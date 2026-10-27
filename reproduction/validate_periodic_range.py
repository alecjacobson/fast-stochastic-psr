"""Replay saved action-quality checks, without rerunning GPU benchmarks."""
import hashlib
import json
from pathlib import Path
import numpy as np
from reproduction.common_rhs_integrity import array_digest


def main():
    root=Path(__file__).resolve().parents[1];count=0
    for case in 'AB':
        out=root/'reproduction/results/periodic_range'/case
        r=json.loads((out/'results.json').read_text());a=np.load(out/'actions.npz')
        assert r['complete'] and r['case']==case
        for path,digest in r['source_sha256'].items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest
        x=np.load(root/f'reproduction/results/matched/{case}/input.npz')['points']
        assert array_digest(x)==r['hashes']['points']
        w=np.random.default_rng(r['seed']).normal(size=(len(x),r['rhs_count']))
        np.testing.assert_array_equal(w,a['weights']);assert array_digest(w)==r['hashes']['weights']
        ref=a['reference'];cov=ref-r['model']['noise_variance']*w
        for name in ['range','packed_nufft200']:
            y=a[name];v=r['methods'][name]['quality'][-1]
            errors=[float(np.linalg.norm(y[:,j]-ref[:,j])/np.linalg.norm(ref[:,j])) for j in range(3)]
            ke=[float(np.linalg.norm(y[:,j]-r['model']['noise_variance']*w[:,j]-cov[:,j])/np.linalg.norm(cov[:,j])) for j in range(3)]
            np.testing.assert_allclose(errors,v['action_relative_l2_per_rhs'],rtol=1e-12,atol=1e-30)
            np.testing.assert_allclose(ke,v['covariance_relative_l2_per_rhs'],rtol=1e-12,atol=1e-30)
            assert (max(errors+ke)<=r['relative_l2_gate'])==v['quality_pass']
            assert v['quality_pass']==(name=='range');count+=1
    print(f'Replayed {count} saved action comparisons; range passes, cube200 failures retained.')


if __name__=='__main__':main()
