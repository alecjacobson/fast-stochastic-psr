"""Recompute numerical gates before permitting scoped timing ratios."""
import argparse,json
from pathlib import Path
import numpy as np
from reproduction.paired_applications import field_quality,quality

ROOT=Path(__file__).resolve().parent/'results/verified_preconditioning'
SOURCE=ROOT.parent/'verified_posterior/three_million'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--one-pass-baseline',action='store_true');a=p.parse_args();root=a.root
    methods=['plain','float32','fused16']+(['float32_onepass'] if a.one_pass_baseline else [])
    records={k:json.loads((root/k/'results.json').read_text()) for k in methods}
    output={};arrays={};pairs={}
    distance=np.load(SOURCE/'geometry.npz')['distance']
    reference=np.load(SOURCE/'fields.npz')
    reference_events={k:reference['reference_'+k] for k in ['rays','collision']}
    for name,r in records.items():
        if not r['complete']:
            output[name]={'complete':False,'failure':r.get('failure')};continue
        a=np.load(root/name/'fields.npz');arrays[name]={k:a[k] for k in a.files}
        fq=field_quality(reference['reference_fields'],a['fields']);eq=quality(reference_events,a,distance)
        if fq!=r['field_quality'] or eq!=r['event_quality']:raise ValueError('stored quality differs from replay')
        d=r['draws'];solve=sum(v['solve_seconds'] for v in d)
        downstream=sum(v[k] for v in d for k in ['aggregation_seconds','query_seconds'])
        t=r['seconds'];coarse=t['coarse_setup']
        checks=[r['mean_solve']]+[v['solve'] for v in d]
        threshold=r['model']['actual_residual_threshold']
        verified=all(c['refinement']['target_rtol']==threshold and c['converged'] and c['refinement']['converged'] and max(c['refinement']['relative_residual_history'][-1])<=threshold for c in checks)
        quality_pass=verified and fq['pass'] and eq['compatibility_pass'] and eq['uncertainty_coverage_pass']
        memory=max((o['own_memory_bytes'] or 0 for o in r['gpu_process_audit']['observations']),default=0)/2**30
        output[name]={'complete':True,'quality_pass':bool(quality_pass),'field_quality':fq,'event_quality':eq,
            'maximum_verified_residual':max(max(c['refinement']['relative_residual_history'][-1]) for c in checks),
            'native_initial_failures':sum(not c['native_solve']['converged'] for c in checks[1:]),
            'corrections':[c['refinement']['corrections'] for c in checks[1:]],
            'initial_iterations':[c['native_solve']['iterations'] for c in checks[1:]],
            'seconds':{'fit':t['fit'],'coarse_setup':coarse,'eight_solves':solve,
                'first_solve':d[0]['solve_seconds'],'later_solve_median':float(np.median([v['solve_seconds'] for v in d[1:]])),
                'setup_plus_eight_solves':coarse+solve,'aggregation_and_queries':downstream,
                'fit_setup_solves_queries':t['fit']+coarse+solve+downstream,
                'input_transfers':sum(v['transfer_seconds'] for v in d),'work_wall':t['instrumented_work_wall']},
            'preconditioner':r.get('preconditioner'),
            'sampled_process_peak_GiB':memory,'sampled_isolation_pass':r['gpu_process_audit']['sampled_isolation_pass']}
    complete=all(v['complete'] for v in output.values())
    common=False
    if complete:
        for i,name in enumerate(methods):
            for baseline in methods[:i]:
                key=name+'_vs_'+baseline
                pairs[key]={'fields':field_quality(arrays[baseline]['fields'],arrays[name]['fields']),
                            'events':quality(arrays[baseline],arrays[name],distance)}
        sources=[(r['model'],r['source_hashes'],r['source_record_sha256'],r['source_geometry_hashes'],
                  [(d['index'],d['rhs_sha256'],d['prior_sha256']) for d in r['draws']]) for r in records.values()]
        common=all(v==sources[0] for v in sources) and len(sources[0][-1])==8
    pair_pass=complete and all(p['fields']['pass'] and p['events']['compatibility_pass'] for p in pairs.values())
    eligible=bool(complete and common and pair_pass and all(v['quality_pass'] and v['sampled_isolation_pass'] for v in output.values()))
    summary={'runs':output,'pairwise':pairs,'common_source_and_draw_hashes':common,'timing_ratios_eligible':eligible,
             'scope':'single eight-draw pilot of verified conditional solves; prior generation excluded; no upstream or complete-application speedup'}
    if eligible:
        f=output['fused16']['seconds'];ratios={}
        for baseline in [k for k in methods if k!='fused16']:
            b=output[baseline]['seconds']
            ratios[baseline]={k:b[k]/f[k] for k in ['eight_solves','setup_plus_eight_solves','fit_setup_solves_queries']}
            saving_per_draw=(b['eight_solves']-f['eight_solves'])/8
            incremental_setup=f['coarse_setup']-b['coarse_setup']
            ratios[baseline]['linear_setup_crossover_draws']=max(0.,incremental_setup/saving_per_draw) if saving_per_draw>0 else None
        summary['fused_ratios']=ratios
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
