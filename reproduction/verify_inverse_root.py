"""Check saved pilot artifacts and replay scalar fields from their coefficients."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from reproduction.common_rhs_integrity import array_digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    import cupy as cp
    from geospsr.spectral import FourierPoisson
    raw=json.loads((a.run/'results.json').read_text())
    if not raw['complete']:raise ValueError('incomplete run')
    fields=np.load(a.run/'fields.npz');q=fields['query']
    assert array_digest(q)==raw['query_sha256']
    op=FourierPoisson(50);rows=[]
    for d in raw['polynomial_draws']:
        i=d['draw'];row={'draw':i}
        for name,key in [('xi','xi_sha256'),('weights','weights_sha256'),('coefficients','coefficients_sha256')]:
            value=np.load(a.run/f'{name}_{i}.npy',mmap_mode='r')
            assert np.isfinite(value).all() and array_digest(value)==d[key]
        coeff=np.load(a.run/f'coefficients_{i}.npy')
        replay=cp.asnumpy(op.evaluate(q,cp.asarray(coeff))[0])
        expected=fields[f'polynomial_{i}']
        row['field_replay_relative_error']=float(np.linalg.norm(replay.astype('d')-expected)/np.linalg.norm(expected.astype('d')))
        assert row['field_replay_relative_error']<1e-5
        passed=max(d['roundtrip_relative_errors'])<=.001
        if raw['case']=='A':
            reference=fields[f'dense_{i}']
            error=float(np.linalg.norm(expected-reference)/np.linalg.norm(reference))
            np.testing.assert_allclose(error,d['dense_fields_relative_error'],rtol=1e-12)
            passed &= error<=.001 and d['dense_weights_relative_error']<=.001
        assert passed==d['quality_pass']
        np.testing.assert_allclose(d['draw_through_query_seconds'],sum(d[k] for k in
            ['gaussian_seconds','inverse_root_seconds','aggregation_seconds','query_seconds']),rtol=1e-12)
        rows.append(row)
    for d in raw['pathwise_draws']:
        assert d['quality_pass']==d['solve']['converged']
        np.testing.assert_allclose(d['draw_through_query_seconds'],sum(d[k] for k in
            ['joint_prior_noise_seconds','solve_seconds','aggregation_seconds','query_seconds']),rtol=1e-12)
    result={'complete':True,'draws':rows,
            'results_sha256':hashlib.sha256((a.run/'results.json').read_bytes()).hexdigest(),
            'fields_sha256':hashlib.sha256((a.run/'fields.npz').read_bytes()).hexdigest(),
            'scope':'artifact hashes, field replay, stored quality decisions and cost accounting; roundtrip is the original-run diagnostic, not independently recomputed here'}
    (a.run/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
