"""Audit saved noisy-scan results and plot timings, work, memory and quality."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from reproduction.benchmark_hierarchical_variance import read_field
from reproduction.noise_selection import surface_points, surface_metrics
from reproduction.run_matched_lengthscales import digest

COLORS = {'h2':'#1769aa', 'locality':'#d65f00'}
LABELS = {'h2':'H² streamed', 'locality':'Locality (storage tuned)'}


def pinned(path, checksum):
    if digest(path) != checksum:
        raise ValueError('Changed record '+str(path))
    return json.loads(Path(path).read_text())


def slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0]) if len(x)>1 else None


def report(root):
    root = Path(root)
    selection = json.loads((root/'selection.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    pinned(root/'selection.json', manifest['selection_sha256'])
    precision = json.loads((root/'precision.json').read_text())
    pinned(root/'precision.json', manifest['precision_sha256'])
    if not all(r['passes'] for r in precision):
        raise ValueError('Failed selection precision check')
    for cell in precision:
        for candidate in cell['candidates']:
            for record in candidate['records']:
                pinned(record['path'], record['sha256'])
    rows = []
    for r in manifest['rows']:
        if not r.get('sha256'):
            rows.append(dict(**r, accepted=False))
            continue
        d = pinned(r['path'], r['sha256'])
        if 'independent_audit' not in d:
            rows.append(dict(**r, accepted=False, reason='No independent audit'))
            continue
        for source, checksum in d['source_sha256'].items():
            if digest(source) != checksum:
                raise ValueError('Changed production source '+source)
        audit = d['independent_audit']
        reference = pinned(audit['reference'], audit['reference_sha256'])
        if reference['contract']['normal_noise'] != r['normal_noise']:
            raise ValueError('Reference noise mismatch')
        paired=bool(r.get('full_query_check'))
        if paired:
            pair = json.loads(Path(r['full_query_check']).read_text())
            pinned(pair['candidate'], pair['candidate_sha256'])
            pinned(pair['baseline'], pair['baseline_sha256'])
        else:
            pair=pinned(r['full_query_diagnostics'],r['full_query_diagnostics_sha256'])
            pinned(pair['source'],pair['source_sha256'])
        field = read_field(d['field'])
        log_path=Path(r['path']).with_suffix('.log')
        events={}
        for line in log_path.read_text().splitlines():
            if line.startswith('{'):
                event=json.loads(line)
                if 'phase' in event:
                    events[event['phase']]=event['elapsed_seconds']
        mean_seconds=events['mean complete']-events['conditioning start']
        variance_seconds=events.get('variance complete',events.get('variance block complete'))-events['mean complete']
        points = surface_points(6102)
        metrics = surface_metrics(field.mean, points)
        _, variance = field.predict(points)
        obs = d['production_gpu_audit']['observations']
        if d['production_gpu_audit']['foreign_process_samples']:
            raise ValueError('Overlapping GPU process in production '+r['path'])
        if r['family'] == 'locality':
            interactions = d['sparse_diagnostics']['nnz']
            degree = d['sparse_diagnostics']['mean_neighbors']
        else:
            interactions = d['interaction_nnz']
            degree = None
        row = dict(**r, accepted=bool(audit['passes'] and (pair['passes'] if paired else True) and d['mean_geometry_check']['passes']),
            numerical_evidence='Independent full-kernel probes'+(' and full-query opponent comparison' if paired else '; opponent stopped, no full-query numerical comparison'),
            total_seconds=d['total_seconds'], setup_seconds=d['setup_seconds'],
            conditioning_seconds=d['regional_cache_timings']['conditioning_seconds'],
            mean_solve_seconds=mean_seconds, variance_solve_seconds=variance_seconds, log_sha256=digest(log_path),
            compression_seconds=d['regional_cache_timings']['compression_seconds'],
            query_seconds=sum(d['query_seconds']),
            peak_gpu_gib=max(o['own_memory_bytes'] or 0 for o in obs)/2**30,
            peak_host_gib=max(o['own_host_rss_bytes'] or 0 for o in obs)/2**30,
            interaction_entries=interactions, mean_neighbors=degree, solve_rank=d['solve_rank'],
            mean_iterations=max(d['solve_info']['mean']['iterations']),
            variance_iterations=max(d['solve_info']['variance']['iterations']),
            max_cdf_error=audit['max_probability_error'],
            max_variance_error_over_prior=audit['max_variance_error_over_prior'],
            linearized_numerical_surface_error=d['mean_geometry_check']['linearized_spatial_error'],
            numerical_surface_error_budget=d['mean_geometry_check']['budget'],
            surface_rms=metrics['penalized_surface_rms'], missing_roots=metrics['missing_roots'],
            normal_alignment=metrics['normal_alignment'],
            median_spatial_uncertainty=float(np.median(np.sqrt(variance)/metrics['gradient_norm'])),
            surface_variance_relative_range=float(np.ptp(variance)/np.mean(variance)),
            shape_brier=pair['shape_brier']['baseline' if r['family']=='h2' else 'candidate'],
            full_query_cdf_error=pair['max_probability_error'] if paired else None,
            full_query_variance_error_over_prior=pair['max_variance_error_over_prior'] if paired else None,
            max_cdf_error_constant_variance=pair['max_error_if_constant_contact_variance_is_used'],
            max_cdf_error_prior_variance=pair['max_error_if_prior_variance_is_used'])
        rows.append(row)
    groups = []
    keys = ['total_seconds','setup_seconds','conditioning_seconds','mean_solve_seconds','variance_solve_seconds','compression_seconds','query_seconds',
            'peak_gpu_gib','peak_host_gib','interaction_entries','mean_iterations','variance_iterations','solve_rank',
            'surface_rms','normal_alignment','median_spatial_uncertainty','shape_brier','max_cdf_error_constant_variance',
            'max_cdf_error_prior_variance','surface_variance_relative_range']
    for sigma in sorted({r['normal_noise'] for r in rows}):
        for n in sorted({r['n'] for r in rows}):
            for family in COLORS:
                sample = [r for r in rows if r['normal_noise']==sigma and r['n']==n and r['family']==family and r['accepted']]
                if not sample:
                    continue
                g = dict(n=n, normal_noise=sigma, family=family, count=len(sample),
                         lengthscale=sample[0]['lengthscale'], methods=sorted({r['method'] for r in sample}),
                         max_cdf_error=max(r['max_cdf_error'] for r in sample),
                         max_variance_error_over_prior=max(r['max_variance_error_over_prior'] for r in sample),
                         max_full_query_cdf_error=max((r['full_query_cdf_error'] for r in sample if r['full_query_cdf_error'] is not None),default=None),
                         max_full_query_variance_error_over_prior=max((r['full_query_variance_error_over_prior'] for r in sample if r['full_query_variance_error_over_prior'] is not None),default=None),
                         missing_roots=sum(r['missing_roots'] for r in sample))
                for key in keys:
                    values = [r[key] for r in sample]
                    g[key] = dict(median=float(np.median(values)), minimum=float(min(values)), maximum=float(max(values)))
                if family=='locality':
                    g['mean_neighbors'] = float(np.median([r['mean_neighbors'] for r in sample]))
                groups.append(g)
    fits = []
    for sigma in sorted({r['normal_noise'] for r in rows}):
        for family in COLORS:
            series = sorted([r for r in groups if r['normal_noise']==sigma and r['family']==family], key=lambda r:r['n'])
            overlap_end=max(r['n'] for r in groups if r['normal_noise']==sigma and r['family']=='locality')
            for scope, sample in [('all',series),('last_three',series[-3:]),
                                  ('locality_overlap',[r for r in series if r['n']<=overlap_end])]:
                fits.append(dict(normal_noise=sigma, family=family, scope=scope, sizes=[r['n'] for r in sample],
                    **{key:slope([r['n'] for r in sample],[r[key]['median'] for r in sample]) for key in
                       ['total_seconds','conditioning_seconds','mean_solve_seconds','variance_solve_seconds','interaction_entries','peak_gpu_gib']}))
    validation = []
    for cell in selection:
        for curve in cell['curves']:
            for record in curve['records']:
                d = pinned(record['path'], record['sha256'])
                for source, checksum in d['source_sha256'].items():
                    if digest(source)!=checksum:
                        raise ValueError('Changed selection source '+source)
        density_ell=.12*np.sqrt(8192/cell['n'])
        density=next(c for c in cell['curves'] if np.isclose(c['lengthscale'],density_ell))
        validation.append(dict(n=cell['n'],normal_noise=cell['normal_noise'],selected_lengthscale=cell['lengthscale'],
            boundary_selected=cell['boundary_selected'],selected_rms=cell['validation_rms'],
            density_rule_lengthscale=density_ell,density_rule_rms=density['score'],
            density_rule_error_ratio=density['score']/cell['validation_rms'],
            clean_same_likelihood_best_lengthscale=cell['clean_same_likelihood_best_lengthscale']))
    offline={}
    for phase in ['validation','precision','precision_refined','tuning']:
        status=[json.loads(p.read_text()) for p in (root/phase).glob('*.status.json')]
        offline[phase+'_worker_wall_seconds']=sum(r['wall_seconds'] for r in status)
        offline[phase+'_worker_count']=len(status)
    offline['reference_seconds']=sum(json.loads(p.read_text())['seconds'] for folder in ['references','references_strict'] for p in (root/folder).glob('*.json'))
    offline['scope']='Recorded worker wall times and unique reference construction; excludes pilot, orchestration, pair comparisons and plotting.'
    controls=[]
    if (root/'dense_control.json').exists():
        for c in json.loads((root/'dense_control.json').read_text()):
            if not c.get('sha256'):
                controls.append(c);continue
            d=pinned(c['path'],c['sha256'])
            pair=json.loads(Path(c['full_query_check']).read_text())
            pinned(pair['candidate'],pair['candidate_sha256']);pinned(pair['baseline'],pair['baseline_sha256'])
            audit=d['independent_audit'];pinned(audit['reference'],audit['reference_sha256'])
            for source,checksum in d['source_sha256'].items():
                if digest(source)!=checksum:raise ValueError('Changed dense control source '+source)
            obs=d['production_gpu_audit']['observations']
            if d['production_gpu_audit']['foreign_process_samples']:raise ValueError('Dense control GPU overlap')
            controls.append(dict(**c,peak_gpu_gib=max(o['own_memory_bytes'] or 0 for o in obs)/2**30,
                interaction_entries=c['n']**2,max_cdf_error=audit['max_probability_error'],
                max_variance_error_over_prior=audit['max_variance_error_over_prior'],
                linearized_numerical_surface_error=d['mean_geometry_check']['linearized_spatial_error']))
    summary=dict(rows=rows,groups=groups,fits=fits,validation=validation,offline_costs=offline,dense_controls=controls,
        scope='Three held-out scans through 32768, one at 131072; not repeated timing trials. Locality and dense controls stop after measured crossovers. Largest H2 cases have independent probes but no completed full-query opponent comparison.',
        numerical_gates=dict(max_cdf_error=.001,max_variance_error_over_prior=.0001),
        manifest_sha256=digest(root/'manifest.json'),selection_sha256=digest(root/'selection.json'),
        report_source_sha256=digest(__file__))
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    noises=sorted({r['normal_noise'] for r in rows})

    def series_plot(ax,sigma,key,ylabel):
        for family,color in COLORS.items():
            sample=sorted([r for r in groups if r['normal_noise']==sigma and r['family']==family],key=lambda r:r['n'])
            x=np.array([r['n'] for r in sample]);y=np.array([r[key]['median'] for r in sample])
            lower=np.array([r[key]['minimum'] for r in sample]);upper=np.array([r[key]['maximum'] for r in sample])
            ax.loglog(x,y,'o-',color=color,label=LABELS[family]);ax.fill_between(x,lower,upper,color=color,alpha=.15)
        extra=sorted([c for c in controls if c['normal_noise']==sigma and c['passes'] and key in c],key=lambda c:c['n'])
        if extra:
            ax.loglog([c['n'] for c in extra],[c[key] for c in extra],'s--',color='#444444',
                      label='Dense-action PCG (one scan)')
        rejected=[r for r in rows if r['normal_noise']==sigma and not r['accepted'] and key in r]
        if rejected:
            ax.scatter([r['n'] for r in rejected],[r[key] for r in rejected],marker='x',s=65,color='#b22222',
                       label='Rejected held-out setting',zorder=5)
        ax.set(xlabel='Training points N',ylabel=ylabel);ax.grid(alpha=.2,which='both')

    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for i,sigma in enumerate(noises):
        for j,(key,label) in enumerate([('total_seconds','End-to-end seconds'),('peak_gpu_gib','Sampled peak GPU GiB'),('interaction_entries','Stored/generated interaction entries')]):
            series_plot(axes[i,j],sigma,key,label);axes[i,j].set_title(f'Normal component noise σ={sigma:g}')
        axes[i,0].legend()
    fig.suptitle('Noisy ellipsoid: independently selected ℓ; 1,000,000 mean + variance queries\n3 held-out scans through 32k; one scan at 131k and for dense-action controls')
    fig.savefig(root/'performance.png',dpi=180);fig.savefig(root/'performance.pdf');plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for i,sigma in enumerate(noises):
        for j,(key,label) in enumerate([('setup_seconds','System setup seconds'),('mean_solve_seconds','Mean solve seconds'),('variance_solve_seconds','Variance solve seconds')]):
            series_plot(axes[i,j],sigma,key,label);axes[i,j].set_title(f'σ={sigma:g}')
        axes[i,0].legend()
    fig.suptitle('Production stages; cross compression, field construction and queries remain in the total\nMean and variance stages read from synchronized solver progress logs')
    fig.savefig(root/'stages.png',dpi=180);fig.savefig(root/'stages.pdf');plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for i,sigma in enumerate(noises):
        for j,(key,label) in enumerate([('surface_rms','Held-out surface root RMS'),('median_spatial_uncertainty','Median √variance / |∇mean|'),('max_cdf_error_constant_variance','Max CDF error with constant contact variance')]):
            series_plot(axes[i,j],sigma,key,label);axes[i,j].set_title(f'σ={sigma:g}')
        axes[i,2].axhline(.001,color='gray',linestyle='--',label='Numerical gate');axes[i,0].legend()
    fig.suptitle('Geometry and uncertainty diagnostics at a known local contact\nThese checks do not establish probability calibration')
    fig.savefig(root/'quality.png',dpi=180);fig.savefig(root/'quality.pdf');plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    sizes=sorted({r['n'] for r in selection});colors=plt.cm.viridis(np.linspace(.1,.85,len(sizes)))
    for i,sigma in enumerate(noises):
        for n,color in zip(sizes,colors):
            cell=next(r for r in selection if r['n']==n and r['normal_noise']==sigma)
            c=sorted(cell['curves'],key=lambda r:r['lengthscale'])
            x=[r['lengthscale'] for r in c]
            axes[i,0].loglog(x,[r['score'] for r in c],'o-',color=color,label=f'N={n:,}')
            axes[i,1].loglog(x,[r['clean_score'] for r in c],'o--',color=color)
            axes[i,0].plot(cell['lengthscale'],cell['validation_rms'],'s' if cell['boundary_selected'] else '*',color=color,markersize=10)
        for j in range(2):
            axes[i,j].set(xlabel='Lengthscale ℓ',ylabel='Validation surface root RMS',title=f'σ={sigma:g}: '+('noisy observations' if j==0 else 'clean values, same likelihood variance'))
            axes[i,j].grid(alpha=.2,which='both')
        axes[i,0].legend()
    fig.suptitle('Lengthscale selection: 2 geometries × 8 noise draws\nStars mark interior selections; squares mark search-boundary selections')
    fig.savefig(root/'selection.png',dpi=180);fig.savefig(root/'selection.pdf');plt.close(fig)
    print(json.dumps(dict(accepted=sum(r['accepted'] for r in rows),total=len(rows),fits=fits),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='reproduction/results/noise_selected')
    report(p.parse_args().root)
