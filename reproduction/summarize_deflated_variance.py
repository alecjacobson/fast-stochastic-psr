"""Only independently confirmed, quality-gated current costs produce ratios."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'results/deflated_variance'


def main():
    result={}
    for variant,prefix in [('unfiltered',''),('filtered','filtered/')]:
        result[variant]={}
        for case in ['A','B']:
            pilot=json.loads((ROOT/f'{prefix}pilot/{case}/results.json').read_text())
            path=ROOT/f'{prefix}confirmation/{case}/results.json'
            if not path.exists():
                result[variant][case]={'pilot_selected':pilot['selected'],'confirmation_complete':False};continue
            r=json.loads(path.read_text())
            if not r['complete']:result[variant][case]={'confirmation_complete':False,'failure':r.get('failure')};continue
            rank,count=r['ranks'][0],r['looks'][0]
            same=bool(pilot['selected'] and (rank,count)==(pilot['selected']['rank'],pilot['selected']['count']) and
                      r['hashes']==pilot['hashes'] and r['power_steps']==pilot.get('power_steps',0))
            checks=[next(c for c in run['checkpoints'] if (c['rank'],c['count'])==(rank,count)) for run in r['runs']]
            eligible=bool(same and all(c['quality_pass'] for c in checks) and r['deterministic_quality_pass'] and r['gpu_process_audit']['sampled_isolation_pass'])
            times=[c['first_estimate_seconds'] for c in checks];baseline=r['deterministic_median_seconds'];common=r['seconds']['fit']+r['seconds']['mean_queries']
            out={'confirmation_complete':True,'rank':rank,'evaluation_draws':count,'pilot_plus_evaluation_draws':rank+count,
                 'covariance_action_columns':rank*(1+r['power_steps']),
                 'all_seeds_quality_pass':all(c['quality_pass'] for c in checks),'settings_match_pilot':same,
                 'contrast_error_percent':[100*c['metrics']['contrast_relative_l2'] for c in checks],
                 'probability_rms_pp':[100*c['metrics']['probability_rms'] for c in checks],
                 'probability_max_pp':[100*c['metrics']['probability_max'] for c in checks],
                 'first_estimate_seconds':times,'median_first_estimate_seconds':float(np.median(times)),
                 'preparation_seconds':[c['setup_seconds']+r['seconds']['query_plan'] for c in checks],
                 'common_fit_mean_seconds':common,'deterministic_median_seconds':baseline,
                 'sampled_isolation_pass':r['gpu_process_audit']['sampled_isolation_pass'],
                 'sampled_process_peak_GiB':max((o['own_memory_bytes'] or 0 for o in r['gpu_process_audit']['observations']),default=0)/2**30,
                 'timing_ratio_eligible':eligible}
            if eligible:
                out['variance_stage_speedup']=baseline/np.median(times)
                out['fit_mean_variance_stage_sum_speedup']=(common+baseline)/(common+np.median(times))
            result[variant][case]=out
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
