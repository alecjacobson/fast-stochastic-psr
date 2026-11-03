"""Check published record gates and bind their exact bytes; not a rerun of numerics."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args()
    records=[]
    for kind in ('solver','variance'):
        for path in sorted((args.results/kind).glob('*/results.json')):
            r=json.loads(path.read_text())
            assert r['complete'],path
            audit=r['gpu_audit']
            assert audit['samples']>0 and audit['observations'] and not audit['errors'] and not audit['foreign_process_samples'],path
            assert audit['sampled_isolation_pass'] and audit['process_query_backend']=='nvml_cuda_device',path
            assert r['source_sha256'] and r['hashes'],path
            count=0
            if kind=='solver':
                assert r['direct_action_relative_error']<=r['gates']['direct_action_relative_error']
                assert r['reference_solve']['converged']
                for name in ('analytic_cg','nufft_refined','sdd','cholesky','cholesky_float64'):
                    method=r['methods'].get(name,{})
                    runs=method.get('runs',[method] if 'quality' in method else [])
                    if name in ('analytic_cg','nufft_refined','sdd'):assert len(runs)==r['args']['repeats']
                    for run in runs:
                        q=run['quality']
                        assert q['pass'] and q['analytic_relative_residual']<=r['gates']['full_analytic_relative_residual']
                        assert q['mean_relative_l2']<=r['gates']['mean_relative_l2']
                        assert math.isfinite(run['solve_seconds']) and run['solve_seconds']>0
                        count+=1
            else:
                assert len(r['measurements'])==r['args']['repeats']*len(r['args']['queries'])
                for row in r['measurements']:
                    assert set(row['seconds'])==set(row['quality'])==set(row['heldout_exact_quality'])
                    assert all(math.isfinite(t) and t>0 for t in row['seconds'].values())
                    for group in ('quality','heldout_exact_quality'):
                        for q in row[group].values():
                            assert q['pass'] and q['positive']
                            assert q['mean_relative_l2']<=.002 and q['variance_relative_l2']<=.0002
                            assert q['contrast_relative_l2']<=.01 and q['probability_rms']<=.0005 and q['probability_max']<=.005
                            count+=1
            records.append({'path':str(path.relative_to(args.results)),
                            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'quality_records_checked':count})
    assert records
    result={'pass':True,'scope':'Checks stored accuracy/timing/audit gates and record hashes; numerical arrays must be regenerated with the benchmark to independently recompute errors.',
            'records':records}
    (args.results/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'{len(records)} promoted runs pass record verification')


if __name__=='__main__':main()
