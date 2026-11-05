"""Plot density-law costs and local reconstruction checks, without mixing gates."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from reproduction.density_protocol import lengthscale
from reproduction.report_matched_lengthscales import LABELS, COLORS, save_figure, read_predict, numerical_checks
from reproduction.run_matched_lengthscales import digest

LABELS=LABELS.copy(); LABELS['original']='S–J controlled variant (matched grid)'
PERIODIC={'dense','sdd','nufft-cg','nufft-hybrid'}


def collect(root):
    manifest=json.loads((root/'manifest.json').read_text());rows=[];limits=[]
    sources=manifest['rows']+[dict(path=str(p),sha256=digest(p)) for p in (root/'refined').glob('*.json')
                              if '.status.' not in p.name and '.audit.' not in p.name]
    for source in sources:
        path=Path(source['path'])
        if not path.exists(): limits.append(source);continue
        if source['sha256']!=digest(path): raise ValueError('Changed result '+str(path))
        d=json.loads(path.read_text());a=d['arguments'];method=a['method'];n=a['n']
        ell=a.get('lengthscale',a.get('original_lengthscale') or a.get('ell',0)*2*np.pi)
        if not np.isclose(ell,lengthscale(n),rtol=1e-14): raise ValueError('Lengthscale mismatch')
        ell=float(lengthscale(n))
        if a['queries']*a['batches']!=1000000: raise ValueError('Query count mismatch')
        if a['amplitude']!=.0025 or a['relative_noise']!=.01: raise ValueError('Parameter mismatch')
        quality=d.get('quality',d.get('independent_audit',{}));audit_path=path.with_name(path.stem+'.audit.json')
        if audit_path.exists():
            audit=json.loads(audit_path.read_text())
            if audit['source_sha256']!=digest(path): raise ValueError('Stale audit')
            quality=audit['quality']
        accepted='total_seconds' in d and (d.get('complete') if method=='original' else quality.get('passes'))
        if not accepted:
            limits.append(dict(path=str(path),sha256=digest(path),method=method,n=n,
                               status=d.get('failure',d.get('status','failed numerical gates')),quality=quality,
                               completed_production_seconds=d.get('total_seconds')));continue
        audits=[d.get(k,{}) for k in ['production_gpu_audit','gpu_process_audit','sdd_pre_reference_gpu_audit']]
        observations=[o for audit in audits for o in audit.get('observations',[])]
        if any(audit.get('foreign_process_samples') for audit in audits): raise ValueError('Concurrent GPU process '+str(path))
        row=dict(path=str(path),sha256=digest(path),method=method,n=n,lengthscale=ell,
                 complete=True,numerical_passes=True,quality=quality,cutoff=a.get('cutoff'),
                 total_seconds=d['total_seconds'],preparation_seconds=d['preparation_seconds'],
                 query_seconds=sum(d['query_seconds']),factor_seconds=d.get('stages',{}).get('cholesky_factorization'),
                 host_peak_bytes=(d['production_host_peak_bytes'] if method=='original' else
                                  max([o.get('own_host_rss_bytes',0) for o in observations],default=0)),
                 gpu_peak_bytes=max([o.get('own_memory_bytes') or 0 for o in observations],default=0),
                 memory_scope=('CPU process high-water, production only' if method=='original' else
                               'Sampled process RSS/device bytes; excludes reference interval; SDD host allocator residue may persist'),
                 mean_iterations=d.get('solve_info',{}).get('mean',{}).get('max_iterations'),
                 variance_iterations=a.get('variance_iterations'),h2_tolerance=a.get('h2_tolerance'),
                 sdd_iterations=d.get('checkpoints',[{}])[-1].get('iterations') if d.get('checkpoints') else None,
                 execution_control=d.get('execution_control'),
                 cross_model_equivalence=False,bandwidth_converged=None if method not in PERIODIC else False)
        if method=='original':
            if a.get('original_normalization')!='controlled': raise ValueError('Uncontrolled original')
            row.update(original_grid_sha256=digest(d['original_grid_file']),grid=d['grid'],
                       parameter_control=d['original_parameter_control'])
        if audit_path.exists(): row.update(audit=str(audit_path),audit_sha256=digest(audit_path))
        rows.append(row)
    chosen={}
    for r in rows:
        key=(r['method'],r['n'])
        if key in chosen: raise ValueError('Multiple accepted settings '+str(key))
        chosen[key]=r
    return rows,limits


def geometry(root,rows):
    sample=json.loads(Path(rows[0]['path']).read_text());cloud_path=sample['arguments']['point_cloud']
    cloud=np.load(cloud_path,mmap_mode='r');start=262144
    mask=np.max(abs(cloud[start:,:3]-[1,0,0]),axis=1)<.005-1e-5
    mask &= np.linalg.norm(cloud[start:,:3]-[1,0,0],axis=1)>1e-12
    heldout=np.array(cloud[start:][mask]);q=heldout[:,:3]
    if len(q)==0: raise ValueError('No held-out observations')
    result=dict(point_cloud=cloud_path,point_cloud_sha256=digest(cloud_path),training_max_n=start,
                heldout_indices=(np.flatnonzero(mask)+start).tolist(),heldout_points=heldout.tolist(),anchor_excluded=True,
                scope='Local held-out geometry, 11 observations; no global reconstruction or probability-calibration claim',rows=[])
    profile_q=np.column_stack((1+np.linspace(-.005,.005,201),np.zeros((201,2))))
    for row in rows:
        d=json.loads(Path(row['path']).read_text());predict=read_predict(d);level=d['level']
        mean,var=predict(q)
        gradient=np.column_stack([(predict(q+1e-6*e)[0]-predict(q-1e-6*e)[0])/2e-6 for e in np.eye(3)])
        norm=np.linalg.norm(gradient,axis=1);roots=[];missing=0
        for point in q:
            def residual(t):return float(predict([[t,point[1],point[2]]])[0][0]-level)
            if residual(.995)*residual(1.005)<=0: roots.append(brentq(residual,.995,1.005)-point[0])
            else: missing+=1
        pm,pv=predict(profile_q)
        r=dict(method=row['method'],n=row['n'],lengthscale=row['lengthscale'],path=row['path'],sha256=row['sha256'],
               heldout_count=len(q),missing_x_roots=missing,
               surface_rms_x_error=float(np.sqrt(np.mean(np.array(roots)**2))) if roots else None,
               surface_rms_linearized_distance=float(np.sqrt(np.mean(((mean-level)/norm)**2))),
               mean_normal_alignment=float(np.mean(np.sum(gradient*heldout[:,3:],axis=1)/norm)),
               median_normal_sigma=float(np.median(np.sqrt(var)/norm)),
               mean_absolute_probability_minus_half=float(np.mean(abs(ndtr((level-mean)/np.sqrt(var))-.5))),
               profile_x_offset=(profile_q[:,0]-1).tolist(),profile_probability=ndtr((level-pm)/np.sqrt(pv)).tolist())
        result['rows'].append(r)
    (root/'phenomenology.json').write_text(json.dumps(result,indent=2)+'\n');return result


def fit(rows,key,min_n):
    selected=[r for r in rows if r['n']>=min_n and r.get(key)]
    if len(selected)<2:return None
    return dict(slope=float(np.polyfit(np.log([r['n'] for r in selected]),np.log([r[key] for r in selected]),1)[0]),
                points=len(selected),minimum_n=selected[0]['n'],maximum_n=selected[-1]['n'])


def main(root):
    root=Path(root);rows,limits=collect(root)
    checks=numerical_checks(rows,root)
    if not all(c['passes'] for c in checks):raise ValueError('Full-query comparison failed; refine before plotting accepted timings')
    diagnostics=geometry(root,rows)
    groups={m:sorted([r for r in rows if r['method']==m],key=lambda r:r['n']) for m in LABELS}
    fits={m:{key:fit(g,key,16384 if m!='original' else 8192)
             for key in ['total_seconds','preparation_seconds','gpu_peak_bytes','host_peak_bytes','factor_seconds']} for m,g in groups.items()}
    for m,g in groups.items():
        fits[m]['total_all_sizes']=fit(g,'total_seconds',0)
        fits[m]['adjacent_total_slopes']=[dict(lower_n=a['n'],upper_n=b['n'],
            slope=float(np.log(b['total_seconds']/a['total_seconds'])/np.log(b['n']/a['n']))) for a,b in zip(g,g[1:])]
    h2={r['n']:r for r in groups['h2-streamed']}
    pairs={m:[dict(n=r['n'],competitor_over_h2=r['total_seconds']/h2[r['n']]['total_seconds'])
              for r in g if r['n'] in h2] for m,g in groups.items() if m!='h2-streamed'}
    summary=dict(rows=rows,limits=limits,fits=fits,observed_pairs=pairs,full_query_checks=checks,
                 analysis_source_sha256=digest(__file__),
                 scope='Fixed vector amplitude/noise and unchanged numerical gates; different models, finite bands not established converged')
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for method,g in groups.items():
        style=dict(color=COLORS[method],label=LABELS[method],marker='o',
                   markerfacecolor='white' if method in PERIODIC else COLORS[method],linestyle='--' if method in PERIODIC else '-')
        for ax,key,scale in [(axes[0,0],'total_seconds',1),(axes[0,1],'gpu_peak_bytes',2**30),(axes[1,0],'host_peak_bytes',2**30)]:
            good=[r for r in g if r[key]>0]
            ax.loglog([r['n'] for r in good],[r[key]/scale for r in good],**style)
        dg=sorted([r for r in diagnostics['rows'] if r['method']==method and r['surface_rms_x_error'] is not None
                   and r['missing_x_roots']==0],key=lambda r:r['n'])
        axes[1,1].loglog([r['n'] for r in dg],[r['surface_rms_x_error'] for r in dg],**style)
    for r in limits:
        if r.get('completed_production_seconds'):
            axes[0,0].scatter(r['n'],r['completed_production_seconds'],marker='x',color=COLORS[r['method']],s=65)
    axes[0,0].set(ylabel='Preparation + 1,000,000 queries (s)',title='Measured mean-and-variance cost')
    axes[0,1].set(ylabel='GPU process memory (GiB)',title='Sampled production peak (0.1 s requested)')
    axes[1,0].set(ylabel='Host process RSS (GiB)',title='Sampled peak; original CPU high-water')
    axes[1,1].set(ylabel='Held-out surface x-root RMS error',title='Local geometry: 11 held-out scan samples')
    axes[1,1].text(.98,.98,'Incomplete root sets omitted;\ncounts recorded in phenomenology.json.',
                   transform=axes[1,1].transAxes,ha='right',va='top',fontsize=7)
    exponents={m:fits[m]['total_seconds']['slope'] for m in ['h2-streamed','sparse-streamed','original']}
    axes[0,0].text(.98,.04,f"Tail fits: H² {exponents['h2-streamed']:.2f}; locality {exponents['sparse-streamed']:.2f}\nOriginal {exponents['original']:.2f} (two largest completed sizes)",
                   transform=axes[0,0].transAxes,ha='right',va='bottom',fontsize=8)
    device_record=json.loads(Path(next(r['path'] for r in rows if r['method']=='h2-streamed')).read_text())
    capacity=device_record['device_memory_bytes']/2**30
    axes[0,1].axhline(capacity,color='black',linestyle=':',alpha=.6)
    axes[0,1].text(.02,.97,f'Device capacity: {capacity:.1f} GiB',transform=axes[0,1].transAxes,va='top',fontsize=8)
    for ax in axes.flat:ax.set_xlabel('Training points N');ax.grid(True,which='both',alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8,bbox_to_anchor=(.5,.035))
    fig.suptitle('Eagle: ℓ(N) = 0.12√(8192/N), fixed vector amplitude/noise',fontsize=12)
    fig.text(.5,.012,'Hollow/dashed: periodic finite-band solver accuracy only. ×: rejected numerical settings. Cross-model calibration unresolved.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.14,1,.96));save_figure(fig,root,'density_sweep');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.3))
    for method in LABELS:
        g=sorted([r for r in diagnostics['rows'] if r['method']==method],key=lambda r:r['n'])
        axes[0].semilogx([r['n'] for r in g],[r['mean_normal_alignment'] for r in g],'o-',color=COLORS[method],label=LABELS[method])
        axes[1].loglog([r['n'] for r in g],[r['median_normal_sigma'] for r in g],'o-',color=COLORS[method])
        selected=[r for r in g if r['n']==8192]
        if selected:
            r=selected[0];axes[2].plot(np.array(r['profile_x_offset'])*1000,r['profile_probability'],color=COLORS[method])
    axes[0].set(xlabel='Training points N',ylabel='Mean normal alignment',title='Held-out local normals')
    axes[1].set(xlabel='Training points N',ylabel='Median √variance / |gradient|',title='Uncertainty scale (not calibration)')
    axes[2].set(xlabel='Contact-normal offset (10⁻³ units)',ylabel='Marginal occupancy probability',title='N = 8,192')
    for ax in axes:ax.grid(True,which='both',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8)
    fig.tight_layout(rect=(0,.16,1,1));save_figure(fig,root,'quality');plt.close(fig)
    manifest=json.loads((root/'manifest.json').read_text())
    grids=sorted([r for r in manifest['rows'] if r['method']=='original'],key=lambda r:r['n'])
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].loglog([r['n'] for r in grids],[r['grid']['grid_points'] for r in grids],'o-',color=COLORS['original'])
    axes[1].loglog([r['n'] for r in grids],[r['grid']['one_dense_covariance_bytes']/2**30 for r in grids],'o-',color=COLORS['original'])
    axes[1].axhline(12,color='black',linestyle=':',label='12 GiB process address-space guard')
    axes[0].set(ylabel='Original grid nodes G',title='Matched spacing h = 0.8943 ℓ(N)')
    axes[1].set(ylabel='One float64 G × G array (GiB)',title='Required allocation, not measured peak');axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xlabel('Training points N');ax.grid(True,which='both',alpha=.2)
    fig.tight_layout();save_figure(fig,root,'original_grid');plt.close(fig)
    print(json.dumps(dict(fits=fits,pairs=pairs,accepted=len(rows),limits=len(limits)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='reproduction/results/density_sweep')
    main(p.parse_args().root)
