"""Compact measured numerical controls; never infer a speedup ratio."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent/'results/verified_posterior'


def main():
    summary={}
    for case in ['A','three_million']:
        r=json.loads((ROOT/case/'results.json').read_text())
        if not r['complete']:raise ValueError('incomplete run')
        draws=r['draws'];p=[d['production'] for d in draws]
        peak=max((v['own_memory_bytes'] or 0 for v in r['gpu_process_audit']['observations']),default=0)
        summary[case]={
            'N':r['N'],'samples':r['samples'],'numerical_gate_pass':r['numerical_gate_pass'],
            'native_initial_failures':sum(not d['native_solve']['converged'] for d in p),
            'production_corrections':[d['refinement']['corrections'] for d in p],
            'maximum_production_actual_residual':max(max(d['refinement']['relative_residual_history'][-1]) for d in p),
            'maximum_reference_actual_residual':max(max(d['reference_solve'].get('relative_residual',
                d['reference_solve'].get('relative_residual_history',[[]])[-1])) for d in draws),
            'field_quality':r['field_quality'],'event_quality':r['event_quality'],
            'seconds':{'fit':r['seconds']['production_fit'],
                **{k:sum(d[k+'_seconds'] for d in p) for k in ['kernel_setup','prior','solve','aggregation']},
                'queries':r['seconds']['production_queries'],
                'fit_draws_queries_sum':sum(r['seconds'][k] for k in ['production_fit','production_draw_components','production_queries']),
                'geometry':r['seconds']['geometry'],'instrumented_work_wall':r['seconds']['instrumented_work_wall']},
            'sampled_process_peak_GiB':peak/2**30,'sampled_isolation_pass':r['gpu_process_audit']['sampled_isolation_pass']}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
