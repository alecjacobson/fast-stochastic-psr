"""Report prepared-weight aggregation costs only when all numerical gates pass."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'results/packed_factor_sources/three_million'


def main():
    r=json.loads((ROOT/'results.json').read_text());result={'complete':r['complete'],'N':r['N'],'draws':r['draws'],
        'scope':r['scope'],'query_setup_seconds':r['query_setup_seconds'],'transfer_seconds':r['transfer_seconds'],'methods':{}}
    eligible=bool(r['complete'] and r['gpu_process_audit']['sampled_isolation_pass'] and
                  all(p['quality_pass'] for m in r['methods'].values() for p in m['passes']))
    result['timing_eligible']=eligible
    for name,m in r['methods'].items():
        aggregate=[p['aggregation_seconds'] for p in m['passes']];combined=[p['aggregation_seconds']+p['query_seconds'] for p in m['passes']]
        result['methods'][name]={'source_setup_seconds':m['source_setup_seconds'],'aggregation_seconds':aggregate,
            'aggregation_query_seconds':combined,'median_aggregation_seconds':float(np.median(aggregate)),
            'median_aggregation_query_seconds':float(np.median(combined)),
            'first_aggregation_with_setup_seconds':m['source_setup_seconds']+aggregate[0],
            'first_aggregation_query_with_setup_seconds':m['source_setup_seconds']+r['query_setup_seconds']+combined[0],
            'max_empirical_covariance_error_percent':100*max(p['field_quality']['empirical_covariance_relative_l2'] for p in m['passes']),
            'all_events_exact':all(p['events_exact'] for p in m['passes'])}
    if eligible:
        g,p=result['methods']['generic'],result['methods']['packed']
        result['speedups']={key:g[key]/p[key] for key in ['median_aggregation_seconds','median_aggregation_query_seconds','first_aggregation_with_setup_seconds','first_aggregation_query_with_setup_seconds']}
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
