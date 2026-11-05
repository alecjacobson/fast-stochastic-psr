"""Offline accuracy tuning and successive-band diagnostics for density sweep.

Run after run_density_sweep; rejected attempts are retained, never relabeled.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr

from reproduction.run_matched_crossover import invoke
from reproduction.run_matched_lengthscales import flags, digest
from reproduction.refine_matched_crossover import main as refine_hierarchy
from reproduction.density_protocol import lengthscale


def extend_dense(root):
    """Follow the dense curve to its observed device limit, with matched H²."""
    manifest_path=root/'manifest.json';manifest=json.loads(manifest_path.read_text())
    for n in [77824,81920]:
        for method in ['h2-streamed','dense']:
            out=root/f'{method}-{n}.json';status_path=out.with_suffix('.status.json')
            ell=float(lengthscale(n));cutoff=int(np.ceil(6/ell))
            if not status_path.exists():
                a=json.loads((root/f'{method}-65536.json').read_text())['arguments'].copy()
                a.update(n=n,output=str(out))
                if method=='dense':
                    a.update(ell=ell/(2*np.pi),cutoff=cutoff,fit_cutoff=cutoff);module='reproduction.benchmark_eagle_comparison'
                else:
                    a.update(lengthscale=ell);module='reproduction.benchmark_hierarchical_variance'
                status=invoke([sys.executable,'-m',module,*flags(a)],out.with_suffix('.log'))
                status['source_sha256']=manifest['source_sha256']
                status_path.write_text(json.dumps(status,indent=2)+'\n')
            status=json.loads(status_path.read_text())
            for source,checksum in status['source_sha256'].items():
                if digest(source)!=checksum:raise ValueError('Stale source '+source)
            d=json.loads(out.read_text())
            if method=='h2-streamed' and 'independent_audit' not in d:
                result=invoke([sys.executable,'-m','reproduction.audit_hierarchical_variance',str(out),
                    '--reference',str(root/f'reference-{n}.json'),'--random-queries','8',
                    '--reference-preconditioner','prefix-kd'],out.with_suffix('.audit.log'))
                if result.get('returncode')!=0:raise RuntimeError('Hierarchy audit failed')
            elif method=='dense' and 'total_seconds' in d and not d.get('quality',{}).get('passes'):
                if not out.with_name(out.stem+'.audit.json').exists():
                    invoke([sys.executable,'-m','reproduction.audit_periodic_scaling',str(out)],out.with_suffix('.audit.log'))
            row=dict(n=n,lengthscale=ell,cutoff=cutoff,method=method,path=str(out),sha256=digest(out),execution=status)
            manifest['rows']=[r for r in manifest['rows'] if (r['n'],r['method'])!=(n,method)]+[row]
            manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
            print('dense extension',method,n,d.get('total_seconds'),d.get('failure'),flush=True)
            if method=='dense' and 'Error creating plan' in d.get('failure','') and 'total_seconds' not in d:
                control=root/f'refined/dense-{n}-single-transform.json'
                if not control.exists():
                    a=d['arguments'].copy();a['output']=str(control)
                    status=invoke([sys.executable,'-m','reproduction.single_transform_control',*flags(a)],control.with_suffix('.log'))
                    status['control_source_sha256']=digest('reproduction/single_transform_control.py')
                    control.with_suffix('.status.json').write_text(json.dumps(status,indent=2)+'\n')
                d=json.loads(control.read_text())
                if 'total_seconds' in d and not d.get('quality',{}).get('passes'):
                    if not control.with_name(control.stem+'.audit.json').exists():
                        invoke([sys.executable,'-m','reproduction.single_transform_control','--module',
                            'reproduction.audit_periodic_scaling',str(control)],control.with_suffix('.audit.log'))
                print('single-transform control',n,d.get('total_seconds'),d.get('failure'),flush=True)
        if 'total_seconds' not in d:break


def main(root):
    root = Path(root); refined = root/'refined'; refined.mkdir(exist_ok=True)
    extend_dense(root)
    for path in sorted(root.glob('h2-streamed-*.json')):
        if '.status.' in path.name: continue
        d = json.loads(path.read_text())
        if 'total_seconds' in d and not d.get('independent_audit', {}).get('passes'):
            refine_hierarchy(root, d['arguments']['n'])
    for path in sorted(root.glob('sdd-*.json')):
        if '.status.' in path.name: continue
        d = json.loads(path.read_text())
        if d.get('quality', {}).get('passes') or not d.get('diverged'): continue
        for eta in [.03, .01]:
            out = refined/f"sdd-{d['arguments']['n']}-eta{eta:g}.json"
            if not out.exists():
                a=d['arguments'].copy(); a.update(eta=eta, output=str(out))
                status=invoke([sys.executable,'-m','reproduction.benchmark_eagle_comparison',*flags(a)],out.with_suffix('.log'))
                status['benchmark_sha256']=digest('reproduction/benchmark_eagle_comparison.py')
                out.with_suffix('.status.json').write_text(json.dumps(status,indent=2)+'\n')
            r=json.loads(out.read_text())
            print('refine SDD', d['arguments']['n'], eta, r.get('total_seconds'), r.get('quality'), flush=True)
            if r.get('quality',{}).get('passes'): break
    results=[]; bandwidth=root/'bandwidth'; bandwidth.mkdir(exist_ok=True)
    for n, cutoffs in [(2048,[50,100]), (8192,[75,100]), (16384,[100]), (32768,[150])]:
        previous=root/f'dense-{n}.json'
        if not previous.exists(): continue
        for cutoff in cutoffs:
            reused=n==8192
            out=(Path(f'reproduction/results/matched_lengthscales/bandwidth/ell0.12-cutoff{cutoff}.json')
                 if reused else bandwidth/f'dense-{n}-cutoff{cutoff}.json')
            if not out.exists():
                a=json.loads(previous.read_text())['arguments'].copy()
                a.update(cutoff=cutoff,fit_cutoff=cutoff,output=str(out))
                status=invoke([sys.executable,'-m','reproduction.benchmark_eagle_comparison',*flags(a)],out.with_suffix('.log'))
                status['benchmark_sha256']=digest('reproduction/benchmark_eagle_comparison.py')
                out.with_suffix('.status.json').write_text(json.dumps(status,indent=2)+'\n')
            old,new=[json.loads(p.read_text()) for p in [previous,out]]
            if 'probes' not in new and 'Error creating plan' in new.get('failure',''):
                control=out.with_name(out.stem+'-single-transform.json')
                if not control.exists():
                    a=new['arguments'].copy();a['output']=str(control)
                    status=invoke([sys.executable,'-m','reproduction.single_transform_control',*flags(a)],control.with_suffix('.log'))
                    status['control_source_sha256']=digest('reproduction/single_transform_control.py')
                    control.with_suffix('.status.json').write_text(json.dumps(status,indent=2)+'\n')
                out=control;new=json.loads(out.read_text())
            row=dict(n=n,source=str(previous),reference=str(out),source_sha256=digest(previous),
                     reference_sha256=digest(out),cutoff=old['arguments']['cutoff'],reference_cutoff=cutoff,
                     reused_reference=reused,passes=False,
                     scope='Uncompressed moments on identical 73 probes; successive finite bands, not infinite-band certification')
            if 'probes' not in old or 'probes' not in new:
                row.update(status='no converged bandwidth result',failure=new.get('failure',old.get('failure')))
                results.append(row); break
            for key in ['point_sha256','normal_sha256']:
                if old[key]!=new[key]: raise ValueError('Mismatched '+key)
            for key in ['ell','amplitude','relative_noise','halfwidth','seed']:
                if old['arguments'][key]!=new['arguments'][key]: raise ValueError('Mismatched '+key)
            if old['probes']['query']!=new['probes']['query']: raise ValueError('Different probes')
            values=[]
            for d in [old,new]:
                m,v=[np.array(d['probes'][key]) for key in ['reference_mean','reference_variance']]
                if not np.isfinite(m).all() or not np.isfinite(v).all() or np.any(v<=0): raise ValueError('Invalid reference')
                values.append((ndtr((m[-1]-m)/np.sqrt(v)),v))
            pe=float(np.max(abs(values[0][0]-values[1][0])))
            ve=float(np.max(abs(values[0][1]-values[1][1]))/new['probes']['prior'])
            row.update(probability_max_difference=pe,variance_max_difference_over_prior=ve,
                       passes=bool(pe<=1e-3 and ve<=1e-4))
            results.append(row);print('bandwidth',row,flush=True);previous=out
        (root/'bandwidth_checks.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='reproduction/results/density_sweep')
    main(p.parse_args().root)
