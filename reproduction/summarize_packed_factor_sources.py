"""Quality/isolation-gated fixed-setting source-packing comparisons."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'results/packed_factor_sources'


def main():
    result={}
    for case in 'AB':
        gpath=ROOT/f'generic/{case}';g=json.loads((gpath/'results.json').read_text())
        pilot=json.loads((ROOT.parent/f'factor_deflation/pilot/{case}/results.json').read_text())
        truth=np.load(gpath/'reference.npz')['variance'];contrast=float(np.linalg.norm(truth-truth.mean()))
        rank,count=g['ranks'][0],g['looks'][0];key=f'variance_{rank}_{count}'
        def passes(r):
            checks=[next(c for c in row['checkpoints'] if (c['rank'],c['count'])==(rank,count)) for row in r['runs']]
            good=bool(r['complete'] and r['selected'] and all(c['quality_pass'] for c in checks) and
                      r['deterministic_quality_pass'] and r['gpu_process_audit']['sampled_isolation_pass'])
            return checks,good
        gc,ggood=passes(g);gmedian=float(np.median([c['first_estimate_seconds'] for c in gc]))
        result[case]={}
        for method in ['generic','packed','packed_planar']:
            path=ROOT/method/case;r=json.loads((path/'results.json').read_text())
            expected={'source_packing':method!='generic','planar_queries':method=='packed_planar'}
            if any(r[k]!=v for k,v in expected.items()):raise ValueError('method flags mismatch')
            if any(r[k]!=g[k] for k in ['hashes','seeds','ranks','looks','factor_batch','model','gates']):raise ValueError('settings mismatch')
            if (rank,count)!=(pilot['selected']['rank'],pilot['selected']['count']):raise ValueError('pilot setting mismatch')
            checks,good=passes(r);paired=[]
            for row in r['runs']:
                filename=f"seed_{row['seed']}.npz"
                paired.append(float(np.linalg.norm(np.load(path/filename)[key]-np.load(gpath/filename)[key])/contrast))
            eligible=bool(good and ggood and max(paired)<=.001)
            times=[c['first_estimate_seconds'] for c in checks];median=float(np.median(times));common=r['seconds']['fit']+r['seconds']['mean_queries']
            result[case][method]=out={'rank':rank,'evaluation_draws':count,'all_seed_quality_and_isolation_pass':good,
                'paired_discrepancy_over_contrast':paired,'paired_gate':.001,'timing_eligible':eligible,
                'contrast_error_percent':[100*c['metrics']['contrast_relative_l2'] for c in checks],
                'probability_rms_pp':[100*c['metrics']['probability_rms'] for c in checks],
                'probability_max_pp':[100*c['metrics']['probability_max'] for c in checks],
                'first_estimate_seconds':times,'median_seconds':median,
                'preparation_seconds':[c['setup_seconds']+r['seconds']['query_plan'] for c in checks],
                'deterministic_seconds':r['deterministic_seconds'],'deterministic_median_seconds':r['deterministic_median_seconds'],
                'common_fit_mean_seconds':common}
            if eligible:out.update(speedup_vs_deterministic=r['deterministic_median_seconds']/median,
                speedup_vs_generic=gmedian/median,fit_mean_variance_stage_sum_speedup=(common+r['deterministic_median_seconds'])/(common+median))
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
