"""Recompute quality and numerical gates from retained issue124 fields."""
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.special import ndtr

from geospsr.nonperiodic import FreeSpacePoissonMatern
from reproduction.partial_quality import OUT, CONTACTS, occupancy, roots
from reproduction.partial_occupancy import ROOT
from reproduction.partial_competitor_common import gates
from reproduction.analyze_partial_quality import balanced, qualifies
from reproduction.stanford_partial_data import digest


def read(path):return json.loads(Path(path).read_text())
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def report():
    selected=read(OUT/'selected.json');fit=read(ROOT/'controls.json')['fits']['spatial']
    model=FreeSpacePoissonMatern(selected['ell'],1/3)
    for group,h in read(OUT/'inputs.json')['panels'].items():assert digest(OUT/(group+'-panel.npz'))==h
    audited=[]
    for path in sorted(OUT.glob('*.json')):
        r=read(path)
        if not isinstance(r,dict) or 'source_sha256' not in r or not isinstance(r['source_sha256'],dict):continue
        for h in r['source_sha256'].values():assert digest(OUT/'sources'/(h+'.py'))==h
        if 'field_sha256' in r:assert digest(path.with_suffix('.npz'))==r['field_sha256']
        if 'gpu_audit' in r:assert r['gpu_audit']['sampled_isolation_pass'],path
        for h in r.get('refinement_history',[]):assert h[-1]['relative_residual']<=1e-10,path
        if 'geometry' in r and r.get('status')=='completed':
            fields=np.load(path.with_suffix('.npz'));panel=np.load(OUT/(r['arguments']['group']+'-panel.npz'))
            assert occupancy(fields['mean'],fields['variance'],panel)==r['occupancy']
            assert roots(fields['probe_mean'],panel)[0]==r['geometry']
            assert r['folded_mean_max_difference']<=1e-10
        audited.append(path.name)
    # Independent full-float64 screening oracle at the selected 16k setting.
    exact=np.load(OUT/f"screen-n16384-ell{selected['ell']}-sigma{selected['sigma']}.npz")
    refined=np.load(OUT/'reference-check-16384.npz');k=CONTACTS.index(selected['contacts'])
    reference_errors=gates(refined['mean'],refined['variance'],dict(mean=exact['mean'][:,k],variance=exact['variance'][:,k]),model,fit)
    assert reference_errors['cdf']<=1e-6 and reference_errors['calibrated_cdf']<=1e-6
    comparisons=[]
    for group,n,method,tag in [('validation',70652,'reference','candidate'+str(selected['decisions'][-1]['candidate'])),('test_side',70652,'reference','selected'),
                              ('test_back',16384,'float64','selected'),('test_other_side',16384,'float64','selected')]:
        old=read(OUT/f'{group}-old-{n}-{method}.json');new=read(OUT/f'{group}-{tag}-{n}-{method}.json')
        from reproduction.evaluate_partial_controls import contact_only_variance
        panel=np.load(OUT/(group+'-panel.npz'));fields=np.load(OUT/f'{group}-{tag}-{n}-{method}.npz')
        cv=contact_only_variance(model,panel['contacts'][:selected['contacts']],panel['queries'])
        assert np.max(fields['variance']-cv)<1e-8*model.prior_variance
        probability_difference=abs(ndtr(-fields['mean']/np.sqrt(fields['variance']))-ndtr(-fields['mean']/np.sqrt(cv)))
        comparisons.append(dict(group=group,n=n,contact_only_max_cdf_difference=float(probability_difference.max()),
            contact_only_mean_cdf_difference=float(probability_difference.mean()),
            full_to_contact_only_variance_ratio_quantiles=np.quantile(fields['variance']/cv,[0,.5,1]).tolist(),
            old_balanced_root_mae=float(balanced(old)),new_balanced_root_mae=float(balanced(new)),
            qualification=qualifies(new,old),old_geometry=old['geometry'],new_geometry=new['geometry'],
            old_occupancy=old['occupancy'],new_occupancy=new['occupancy'],new_contact_only=new['contact_only']))
    trials=[]
    for path in sorted(OUT.glob('benchmark-*-r*.json')):
        r=read(path);fields=np.load(path.with_suffix('.npz')) if 'field_sha256' in r else None
        if fields is not None:
            refpath=Path(r['reference_path']);assert digest(refpath)==r['reference_sha256']
            reference=np.load(refpath.with_suffix('.npz'))
            truth=dict(mean=reference['benchmark_mean'],variance=reference['benchmark_variance']) if 'benchmark_mean' in reference else dict(reference)
            error=gates(fields['mean'],fields['variance'],truth,model,fit)
            assert error==r['errors'],path
            assert r['accepted']==bool(r['status']=='completed' and error['passes'] and r['gpu_audit']['sampled_isolation_pass'])
        memory=[o['own_memory_bytes'] for o in r['gpu_audit']['observations'] if o['own_memory_bytes'] is not None]
        host=[o['own_host_rss_bytes'] for o in r['gpu_audit']['observations']]
        trials.append(dict(file=path.name,n=r['arguments']['n'],method=r['arguments']['method'],iterations=r['arguments']['iterations'],tolerance=r['arguments']['tolerance'],
            seconds=r.get('total_seconds'),accepted=r['accepted'],errors=r.get('errors'),status=r['status'],
            gpu_gib=max(memory)/2**30,host_gib=max(host)/2**30,setup_seconds=r.get('setup_seconds'),mean_contact_seconds=r.get('mean_contact_seconds')))
    grids={}
    if (OUT/'display-selected-120.json').exists():
        import igl
        for tag in ['old','selected']:
            pair=[]
            for resolution in [80,120]:
                p=OUT/f'display-{tag}-{resolution}.json';r=read(p)
                assert r['field_sha256']==digest(p.with_suffix('.npz'))
                assert r['gpu_audit']['sampled_isolation_pass']
                pair.append(np.load(p.with_suffix('.npz')))
            distances=[np.sqrt(igl.point_mesh_squared_distance(a['vertices'],b['vertices'],b['faces'])[0]) for a,b in [pair,pair[::-1]]]
            grids[tag]=[dict(count=len(d),median=float(np.median(d)),p95=float(np.quantile(d,.95)),maximum=float(d.max())) for d in distances]
    summary=dict(selected=selected,quality=comparisons,reference_errors=reference_errors,trials=trials,audited_records=audited,
        display_mesh_distances_80_to_120_and_reverse=grids)
    primary=OUT/'primary-runs.json'
    if primary.exists():
        summary['primary']={}
        for label,names in read(primary).items():
            rows=[next(t for t in trials if t['file']==name) for name in names]
            assert len(rows)==3 and all(r['accepted'] for r in rows)
            assert len({(r['method'],r['iterations'],r['tolerance']) for r in rows})==1
            summary['primary'][label]=dict(seconds_median=float(np.median([r['seconds'] for r in rows])),
                seconds_range=[min(r['seconds'] for r in rows),max(r['seconds'] for r in rows)],
                gpu_gib_max=max(r['gpu_gib'] for r in rows),runs=names)
    scaling={}
    for method in ['h2-joint','corrected-wide']:
        rows=[r for r in trials if r['method']==method and r['accepted']]
        assert len({(r['iterations'],r['tolerance']) for r in rows})==1
        points=[]
        for n in sorted({r['n'] for r in rows}):
            group=[r for r in rows if r['n']==n]
            points.append(dict(n=n,seconds=float(np.median([r['seconds'] for r in group])),runs=len(group),
                seconds_range=[min(r['seconds'] for r in group),max(r['seconds'] for r in group)],gpu_gib=max(r['gpu_gib'] for r in group)))
        slope=float(np.polyfit(np.log([r['n'] for r in points]),np.log([r['seconds'] for r in points]),1)[0]) if len(points)>=2 else None
        scaling[method]=dict(points=points,empirical_loglog_slope=slope)
    summary['scaling']=scaling
    subprocess.run(['git','diff','--exit-code','0a80da1ccbfa3613f62ca082fb160eac6dbae91e','--',
        'reproduction/results/mixed_h2','reproduction/results/partial_renderings'],check=True)
    old_display=np.load(OUT/'display-old-80.npz');frozen=np.load(OUT.parent/'partial_renderings/mean-volume.npz')
    for axis in 'xyz':np.testing.assert_array_equal(old_display['axis_'+axis],frozen['axis_'+axis])
    summary['frozen_h2_vs_full_reference_mean_grid_max_difference']=float(np.max(abs(old_display['mean']-frozen['mean'])))
    for folder in OUT.glob('render-*'):
        for p in folder.glob('*.json'):
            render=read(p)
            if 'scene_sha256' not in render:continue
            assert digest(folder/'scene.npz')==render['scene_sha256']
            geometry=folder/('mean-mesh.npz' if render['kind']=='mean' else 'scene.npz')
            assert digest(geometry)==render['geometry_sha256']
            assert digest(folder/'sources'/(render['source_sha256']+'.py'))==render['source_sha256']
    summary['historical_artifacts_unchanged']=True
    write(OUT/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k in ['primary','reference_errors']},indent=2))


if __name__=='__main__':report()
