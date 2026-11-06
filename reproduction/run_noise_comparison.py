"""Tune solvers on a separate scan, freeze settings, and evaluate noisy scans."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import numpy as np

from reproduction.noise_selection import AMPLITUDE,HALFWIDTH,TUNING_SEED,TEST_SEEDS,mean_values
from reproduction.benchmark_hierarchical_variance import read_field
from reproduction.run_matched_crossover import invoke
from reproduction.run_matched_lengthscales import flags,digest

ROOT=Path('reproduction/results/noise_selected')
BASE=Path('reproduction/results/density_sweep')
CUTOFFS=[1,2,3,4,6,8,10,12,14,16,20]


def mean_geometry_check(record,cell):
    """Keep numerical anchored-mean error small relative to model surface error.

    This is a linearized diagnostic on the independent reference probes, not
    a certified root bound. It prevents broad posterior variance from hiding
    substantial mean errors in the CDF gate alone.
    """
    ref=json.loads(Path(record['independent_audit']['reference']).read_text())
    q=np.asarray(ref['queries']);field=read_field(record['field'])
    mean,_=field.predict(q);truth=np.asarray(ref['mean']);bound=np.asarray(ref['mean_error_bound'])
    error=np.max(abs((mean-mean[-1])-(truth-truth[-1]))+bound+bound[-1])
    gradient=np.column_stack([mean_values(np.polynomial.chebyshev.chebder(field.mean,axis=d),q)/HALFWIDTH for d in range(3)])
    minimum=float(np.linalg.norm(gradient,axis=1).min())
    spatial=float(error/minimum) if minimum>0 else float('inf')
    budget=.1*cell['validation_rms']
    return dict(max_anchored_mean_error=float(error),minimum_probe_gradient=minimum,
                linearized_spatial_error=spatial,budget=budget,passes=bool(spatial<=budget),
                scope='Independent probes; linearized mean-error diagnostic, not a uniform root bound')


def benchmark(root,cell,method,seed,phase,settings):
    n,sigma,ell=cell['n'],cell['normal_noise'],cell['lengthscale']
    tag='-'.join(f'{k}{v:g}' for k,v in settings.items())
    out=root/f'{phase}/s{sigma:g}-n{n}-seed{seed}-{method}-{tag}.json';out.parent.mkdir(parents=True,exist_ok=True)
    status_path=out.with_suffix('.status.json')
    if not status_path.exists():
        template='sparse-streamed' if method.startswith('sparse') else method
        a=json.loads((BASE/f'{template}-8192.json').read_text())['arguments'].copy()
        a.update(n=n,seed=seed,geometry='ellipsoid',point_cloud=None,normal_noise=sigma,lengthscale=ell,
                 amplitude=AMPLITUDE,relative_noise=sigma*sigma/AMPLITUDE,halfwidth=HALFWIDTH,queries=200000,batches=5,
                 rhs_block_size=72,output=str(out),method=method,**settings)
        status=invoke([sys.executable,'-m','reproduction.benchmark_hierarchical_variance',*flags(a)],out.with_suffix('.log'))
        status['source_sha256']={p:digest(p) for p in ['reproduction/benchmark_hierarchical_variance.py',
            'geospsr/nonperiodic.py','geospsr/nonperiodic_local_gpu.py','geospsr/hierarchical_gpu.py']}
        status_path.write_text(json.dumps(status,indent=2)+'\n')
        print(phase,method,n,sigma,seed,settings,status.get('returncode','timeout'),round(status['wall_seconds'],2),flush=True)
    status=json.loads(status_path.read_text())
    for source,checksum in status['source_sha256'].items():
        if digest(source)!=checksum:raise ValueError('Stale benchmark source '+source)
    if not out.exists():return out,False
    d=json.loads(out.read_text())
    if 'total_seconds' not in d:return out,False
    expected=dict(n=n,seed=seed,normal_noise=sigma,lengthscale=ell,amplitude=AMPLITUDE,
                  relative_noise=sigma*sigma/AMPLITUDE,method=method,**settings)
    if any(d['arguments'][key]!=value for key,value in expected.items()):
        raise ValueError('Stale benchmark parameters '+str(out))
    if 'independent_audit' not in d:
        folder='references_strict' if n>16384 else 'references'
        reference=root/f'{folder}/s{sigma:g}-n{n}-seed{seed}.json';reference.parent.mkdir(exist_ok=True)
        bounds=['--reference-mean-bound','1e-9'] if n>16384 else []
        result=invoke([sys.executable,'-m','reproduction.audit_hierarchical_variance',str(out),
                      '--reference',str(reference),'--random-queries','8','--reference-preconditioner','prefix-kd',*bounds],out.with_suffix('.audit.log'))
        if result.get('returncode')!=0:raise RuntimeError('Independent audit failed '+str(out))
    d=json.loads(out.read_text())
    if 'mean_geometry_check' not in d:
        d['mean_geometry_check']=mean_geometry_check(d,cell)
        out.write_text(json.dumps(d,indent=2)+'\n')
    return out,bool(d['independent_audit']['passes'] and d['mean_geometry_check']['passes'])


def compare(root,candidate,baseline,phase):
    out=root/'checks'/phase/(candidate.stem+'.json');out.parent.mkdir(parents=True,exist_ok=True)
    if not out.exists():
        result=invoke([sys.executable,'-m','reproduction.compare_hierarchical_application',str(candidate),str(baseline),
                       '--output',str(out)],out.with_suffix('.log'))
        if result.get('returncode')!=0:raise RuntimeError('Full-query comparison failed to execute')
    d=json.loads(out.read_text())
    if d['candidate_sha256']!=digest(candidate) or d['baseline_sha256']!=digest(baseline):raise ValueError('Stale full-query comparison')
    return out,d


def tune(root,cell):
    n,sigma=cell['n'],cell['normal_noise'];out=root/f'settings/s{sigma:g}-n{n}.json';out.parent.mkdir(exist_ok=True)
    if out.exists():
        settings=json.loads(out.read_text())
        if settings['lengthscale']!=cell['lengthscale']:
            raise ValueError('Tuning settings use a different selected lengthscale')
        return settings
    attempts=[];chosen=None;geometry_failures=set()
    # Largest-N tuning exposed a mean/cross precision floor. Tighten both
    # sources before its held-out runs, retaining the original rejected trials.
    common=dict(cross_tolerance=1e-10,solve_tolerance=1e-10) if n>=131072 else {}
    tolerances=[1e-7,1e-8,1e-9] if common else [1e-5,1e-6,1e-7,1e-8,1e-9]
    for tolerance,iterations in itertools.product(tolerances,[8,16,32]):
        if tolerance in geometry_failures:continue
        setting=dict(h2_tolerance=tolerance,variance_iterations=iterations,**common)
        path,passed=benchmark(root,cell,'h2-streamed',TUNING_SEED,'tuning',setting)
        attempts.append(dict(path=str(path),passes=passed))
        if path.exists():
            record=json.loads(path.read_text())
            if not record.get('mean_geometry_check',{}).get('passes',True):
                # More variance iterations do not refine the hierarchy's mean bias.
                geometry_failures.add(tolerance)
        if passed:
            d=json.loads(path.read_text());q=d['independent_audit']
            if q['max_probability_error']<=3e-4 and q['max_variance_error_over_prior']<=3e-5:
                chosen=setting;baseline=path;break
    if chosen is None:raise RuntimeError('No H2 tuning setting passes the calibration margin')
    if n>=131072:
        # The user permits stopping competitors once they are measurably slower.
        # Verify that evidence for this noise level before omitting larger CSR runs.
        earlier=json.loads((root/'manifest.json').read_text())['rows']
        times={key:[] for key in ['h2','locality']}
        for row in earlier:
            if row['n']==32768 and row['normal_noise']==sigma and row['passes'] and row.get('full_query_passes'):
                times[row['family']].append(json.loads(Path(row['path']).read_text())['total_seconds'])
        if not (len(times['h2'])==len(times['locality'])==3 and np.median(times['locality'])>np.median(times['h2'])):
            raise RuntimeError('Missing held-out locality crossover evidence for the stopping rule')
        result=dict(n=n,normal_noise=sigma,lengthscale=cell['lengthscale'],h2=chosen,locality=None,
                    locality_method=None,tuning_seed=TUNING_SEED,attempts=attempts,test_settings_frozen=True,
                    competitor_stop='Locality slower on three accepted 32768-point scans; largest comparison uses a stronger full-kernel dense-action control',
                    test_seeds=TEST_SEEDS[:1])
        out.write_text(json.dumps(result,indent=2)+'\n');return result
    local=None
    for cutoff in CUTOFFS:
        for iterations in [8,16,32]:
            setting=dict(sparse_cutoff=cutoff,variance_iterations=iterations,**common)
            path,passed=benchmark(root,cell,'sparse-streamed',TUNING_SEED,'tuning',setting)
            attempt=dict(path=str(path),passes=passed);attempts.append(attempt)
            if not passed:
                # A hard-truncation curvature failure is not repaired by more iterations.
                if not path.exists():break
                continue
            check,quality=compare(root,path,baseline,'tuning');attempt['comparison']=str(check)
            q=json.loads(path.read_text())['independent_audit']
            if (quality['passes'] and quality['max_probability_error']<=3e-4 and
                quality['max_variance_error_over_prior']<=3e-5 and
                q['max_probability_error']<=3e-4 and q['max_variance_error_over_prior']<=3e-5):
                local=setting;local_path=path;break
        if local is not None:break
        if cutoff*cell['lengthscale']/np.sqrt(3)>=2:
            # Every pair on this ellipsoid is already retained. Larger radii
            # cannot repair a numerical failure or a resource endpoint.
            break
    if local is None:raise RuntimeError('No locality tuning setting passes the calibration margin')
    # Give locality the option to retain its graph when it fits. This changes
    # storage, not the kernel, radius or output-accuracy requirement.
    local_method='sparse-streamed';streamed=json.loads(local_path.read_text())
    cached_bound=streamed['sparse_diagnostics']['cached_graph_bytes']
    cached_control=dict(required_graph_bytes=cached_bound,attempted=False)
    if cached_bound<.6*streamed['device_memory_bytes']:
        cached,passed=benchmark(root,cell,'sparse-gpu',TUNING_SEED,'tuning',local)
        cached_control.update(attempted=True,path=str(cached),passes=passed)
        if passed:
            check,quality=compare(root,cached,baseline,'tuning')
            d=json.loads(cached.read_text());q=d['independent_audit']
            if (quality['passes'] and quality['max_probability_error']<=3e-4 and quality['max_variance_error_over_prior']<=3e-5
                and q['max_probability_error']<=3e-4 and q['max_variance_error_over_prior']<=3e-5
                and d['total_seconds']<streamed['total_seconds']):local_method='sparse-gpu'
    else:cached_control['reason']='graph alone exceeds 60 percent of device capacity; keep streamed route'
    result=dict(n=n,normal_noise=sigma,lengthscale=cell['lengthscale'],h2=chosen,locality=local,
                locality_method=local_method,cached_locality_control=cached_control,
                tuning_seed=TUNING_SEED,attempts=attempts,
                selection_rule='First passing hierarchy setting and shortest tested passing locality radius; 0.3x gate tuning margin',
                test_settings_frozen=True)
    out.write_text(json.dumps(result,indent=2)+'\n');return result


def run(root,sizes=None):
    root=Path(root);selection=json.loads((root/'selection.json').read_text())
    precision=json.loads((root/'precision.json').read_text())
    largest=max(c['n'] for c in selection)
    for cell in [c for c in selection if c['n']==largest]:
        if not any(p['n']==largest and p['normal_noise']==cell['normal_noise'] and
                   p['selected']==cell['lengthscale'] and p['passes'] for p in precision):
            raise ValueError('Complete the selection precision checks before tuning')
    manifest_path=root/'manifest.json';manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(rows=[])
    for cell in selection:
        if sizes is not None and cell['n'] not in sizes:continue
        settings=tune(root,cell)
        local_method=settings['locality_method']
        for seed in settings.get('test_seeds',TEST_SEEDS):
            paths={};passes={}
            methods=[('h2-streamed','h2')]+([(local_method,'locality')] if local_method else [])
            for method,key in methods:
                path,passed=benchmark(root,cell,method,seed,'test',settings[key]);paths[method]=path;passes[method]=passed
                row=dict(n=cell['n'],normal_noise=cell['normal_noise'],lengthscale=cell['lengthscale'],seed=seed,method=method,
                         family=key,path=str(path),passes=passed,settings=settings[key],sha256=digest(path) if path.exists() else None)
                manifest['rows']=[r for r in manifest['rows'] if (r['n'],r['normal_noise'],r['seed'],r['method'])!=
                                 (row['n'],row['normal_noise'],seed,method)]+[row]
            if local_method and all(passes.values()):
                check,quality=compare(root,paths[local_method],paths['h2-streamed'],'test')
                for row in manifest['rows']:
                    if (row['n'],row['normal_noise'],row['seed'])==(cell['n'],cell['normal_noise'],seed):
                        row.update(full_query_check=str(check),full_query_passes=quality['passes'])
            manifest['selection_sha256']=digest(root/'selection.json')
            manifest['precision_sha256']=digest(root/'precision.json')
            manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(ROOT));p.add_argument('--sizes',type=int,nargs='+')
    a=p.parse_args();run(a.root,a.sizes)
