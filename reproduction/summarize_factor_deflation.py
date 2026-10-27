"""Report only fixed-setting independent confirmation timing ratios."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'results/factor_deflation'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--confirmation-root',type=Path,default=ROOT/'confirmation')
    p.add_argument('--output',type=Path,default=ROOT/'summary.json')
    a=p.parse_args()
    result={}
    for case in 'AB':
        pilot=json.loads((ROOT/f'pilot/{case}/results.json').read_text())
        path=a.confirmation_root/case/'results.json'
        if not path.exists():
            result[case]={'pilot_selected':pilot['selected'],'confirmation_complete':False};continue
        r=json.loads(path.read_text())
        if not r['complete']:
            result[case]={'confirmation_complete':False,'failure':r.get('failure')};continue
        rank,count=r['ranks'][0],r['looks'][0]
        same=bool(pilot['selected'] and (rank,count)==(pilot['selected']['rank'],pilot['selected']['count']) and
                  r['hashes']==pilot['hashes'] and r['estimator']==pilot['estimator'])
        checks=[next(c for c in run['checkpoints'] if (c['rank'],c['count'])==(rank,count)) for run in r['runs']]
        paired=True
        if r.get('factor_batch',4)==16:
            comparison=json.loads((ROOT/'batch_comparison.json').read_text())[case]
            paired=bool(comparison['pass'] and comparison['gate']==.001 and comparison['batch16_result_sha256']==hashlib.sha256(path.read_bytes()).hexdigest())
        eligible=bool(same and paired and all(c['quality_pass'] for c in checks) and r['deterministic_quality_pass'] and r['gpu_process_audit']['sampled_isolation_pass'])
        times=[c['first_estimate_seconds'] for c in checks];baseline=r['deterministic_median_seconds'];common=r['seconds']['fit']+r['seconds']['mean_queries']
        result[case]=out={'confirmation_complete':True,'factor_batch':r.get('factor_batch',4),'sketch_rank':rank,'evaluation_draws':count,
            'factor_action_columns':2*rank,'all_seeds_quality_pass':all(c['quality_pass'] for c in checks),
            'settings_match_pilot':same,'paired_batch_check_pass':paired,'contrast_error_percent':[100*c['metrics']['contrast_relative_l2'] for c in checks],
            'probability_rms_pp':[100*c['metrics']['probability_rms'] for c in checks],
            'probability_max_pp':[100*c['metrics']['probability_max'] for c in checks],
            'first_estimate_seconds':times,'median_first_estimate_seconds':float(np.median(times)),
            'preparation_seconds':[c['setup_seconds']+r['seconds']['query_plan'] for c in checks],
            'retained_projection_bytes':r['runs'][0]['settings'][str(rank)]['record']['retained_bytes'],
            'common_fit_mean_seconds':common,'deterministic_median_seconds':baseline,
            'sampled_isolation_pass':r['gpu_process_audit']['sampled_isolation_pass'],'timing_ratio_eligible':eligible}
        if eligible:
            out['variance_stage_speedup']=float(baseline/np.median(times))
            out['fit_mean_variance_stage_sum_speedup']=float((common+baseline)/(common+np.median(times)))
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
