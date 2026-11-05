"""Measured scaling with stage separation, adjacent slopes and explicit limits."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator,FuncFormatter,NullFormatter
import numpy as np

matplotlib.rcParams['svg.fonttype']='none'
NAMES={'dense':'GeoSPR Cholesky','sdd':'GeoSPR SDD','nufft-cg':'NUFFT + analytic PCG',
       'nufft-hybrid':'Cholesky–NUFFT variance','original':'Sellán–Jacobson (16³, CPU)',
       'h2-streamed':'H²','sparse-streamed':'Streamed locality','dense-float32':'Free-space Cholesky'}
COLORS=dict(zip(NAMES.values(),['#009E73','#CC79A7','#7A3DB8','#E69F00','#303030','#0072B2','#D55E00','#888888']))
OLD=Path('reproduction/results/poissonrecon_large/eagle')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def slope(x,y):return float(np.polyfit(np.log(x),np.log(y),1)[0])


def slopes(rows,stage='total'):
    rows=sorted(rows,key=lambda r:r['n']);x=np.array([r['n'] for r in rows]);y=np.array([r[stage] for r in rows])
    valid=y>0;x,y=x[valid],y[valid]
    if len(x)<2:return None
    return dict(fit=slope(x,y),range=[int(x[0]),int(x[-1])],
                adjacent=[dict(first=int(a),last=int(b),slope=float(np.log(v/u)/np.log(b/a))) for a,b,u,v in zip(x[:-1],x[1:],y[:-1],y[1:])])


def collect(root):
    paths=list((root/'periodic').glob('*.json'))+list((root/'free-space').glob('*.json'))
    paths+=[OLD/'eight-large/h2-streamed-796825-0.json',OLD/'large-control/sparse-streamed-796825-0.json']
    paths+=[OLD/f'tuned/dense-float32-{n}-0.json' for n in [32768,77824]]
    paths+=list((root/'original').glob('full-grid16-*.json'))
    rows=[];limits=[];hardware=None;cloud_hash=None;covered=set()
    checks={v['candidate']:v for v in json.loads((root/'full_query_checks.json').read_text())}
    for path in json.loads((root/'full_query_checks_free.json').read_text()):
        checked=json.loads(Path(path).read_text())
        if not checked['passes']:raise ValueError('Failed full query check '+path)
        for role in ['candidate','baseline']:
            covered.add(checked[role])
            if checked[role+'_sha256']!=digest(checked[role]):raise ValueError('Stale full query check '+path)
    for i in [0,1]:
        audit=json.loads((root/f'targeted/audit-{i}.json').read_text())
        if not audit['passes'] or audit['source_sha256']!=digest(audit['source']) or audit['reference_sha256']!=digest(audit['reference']):
            raise ValueError('Missing/passing targeted audit required')
    for p in sorted(paths):
        if any(s in p.name for s in ['.status.','.audit.']):continue
        d=json.loads(p.read_text());a=d['arguments'];method=a['method']
        free=method in ['h2-streamed','sparse-streamed','dense-float32'];original=method=='original'
        if a['queries']*a['batches']!=1048576:raise ValueError('Query workload mismatch')
        current_cloud=d['point_cloud_metadata']['normalized_sha256']
        if cloud_hash is None:cloud_hash=current_cloud
        if current_cloud!=cloud_hash:raise ValueError('Cloud mismatch')
        if free and str(p) not in covered:raise ValueError('Missing full-query check '+str(p))
        current=(d.get('device'),d.get('device_memory_bytes',d.get('device_memory')),d.get('cpu_quota'),d.get('host_memory_limit',d.get('host_limit')))
        if not original:
            if hardware is None:hardware=current
            if hardware!=current:raise ValueError('Hardware mismatch '+str(p))
        elif hardware is not None and current[2:]!=hardware[2:]:raise ValueError('CPU hardware mismatch')
        quality=d.get('independent_audit',d.get('quality',{}));audit=p.with_name(p.stem+'.audit.json')
        if audit.exists():
            audit_data=json.loads(audit.read_text())
            if audit_data['source_sha256']!=digest(p):raise ValueError('Stale offline audit')
            quality=audit_data['quality']
        accepted=(d.get('complete') if original else quality.get('passes')) and 'total_seconds' in d
        if accepted and not free and not original and not d.get('gpu_process_audit',{}).get('sampled_isolation_pass'):
            raise ValueError('Missing GPU isolation evidence '+str(p))
        if accepted and not free and not original and method!='dense':
            checked=checks.get(str(p))
            if checked is None or checked['candidate_sha256']!=digest(p) or checked['reference_sha256']!=digest(checked['reference']):
                raise ValueError('Missing/stale periodic full-query check '+str(p))
            accepted=checked['passes']
        if not accepted:
            limits.append(dict(source=str(p),sha256=digest(p),method=NAMES[method],n=a['n'],
                status=d.get('failure',d.get('status','failed accuracy')),stages=d.get('stages'),
                censored_seconds=d.get('censored_preparation_seconds')))
            continue
        if free:
            stages=d['regional_cache_timings'];construction=d['setup_seconds']
            conditioning=stages['conditioning_seconds'];regional=stages['compression_seconds']+stages['moments_seconds']+d['field_seconds']
            matrix=factor=0.;iterations=d.get('solve_info',{})
        elif original:
            profile=d.get('original_profile',[])
            factor=sum(v['self_seconds'] for v in profile if 'gstrf' in v['function'])
            conditioning=sum(v['self_seconds'] for v in profile if 'SuperLU' in v['function'] and 'solve' in v['function'])
            construction=max(0,d['preparation_seconds']-factor-conditioning);conditioning+=factor
            regional=0.;matrix=0.;iterations={}
        else:
            s=d['stages'];matrix=s.get('matrix_construction',0);factor=s.get('cholesky_factorization',0)
            construction=sum(s.get(k,0) for k in ['matrix_construction','cross_spectrum','training_spectrum','analytic_action_setup','analytic_preconditioner','sdd_compile'])
            conditioning=factor+sum(s.get(k,0) for k in ['solve','mean_solve','global_variance_series','mean_aggregate'])
            regional=s.get('regional_cross_compression',0)+s.get('field',0);iterations=d.get('solve_blocks',d.get('checkpoints',[]))
        rows.append(dict(method=NAMES[method],key=method,model='free-space' if free else 'original-grid' if original else 'periodic',
            n=a['n'],total=d['total_seconds'],preparation=d['preparation_seconds'],query=sum(d['query_seconds']),
            construction=construction,conditioning=conditioning,regional=regional,matrix=matrix,factor=factor,
            stages_overlap_note='factor is also part of conditioning; matrix is also part of construction',
            source=str(p),sha256=digest(p),quality=quality,parameters=a,iterations=iterations,
            rank=d.get('solve_rank'),h2_max_rank=d.get('max_h2_rank'),h2_sum_ranks=d.get('sum_h2_ranks'),
            interactions=d.get('interaction_nnz',d.get('sparse_diagnostics',{}).get('nnz')),
            mean_iterations=d.get('solve_info',{}).get('mean',{}).get('max_iterations'),
            variance_iterations=d.get('solve_info',{}).get('variance',{}).get('max_iterations'),
            matrix_bytes=d.get('matrix_bytes',d.get('factor_bytes')),reused=str(p).startswith(str(OLD)),
            source_hashes=d.get('source_sha256'),gpu_audit=d.get('gpu_process_audit')))
    return rows,limits,hardware


def save(fig,root,name):
    for ext in ['png','pdf','svg']:fig.savefig(root/f'{name}.{ext}',dpi=180)
    # Stable text-friendly SVG rather than trailing-space noise.
    p=root/f'{name}.svg';p.write_text('\n'.join(l.rstrip() for l in p.read_text().splitlines())+'\n')
    plt.close(fig)


def main(root):
    root=Path(root);rows,limits,hardware=collect(root)
    groups={name:sorted([r for r in rows if r['method']==name],key=lambda r:r['n']) for name in COLORS}
    for name,g in groups.items():
        settings=[{k:v for k,v in r['parameters'].items() if k not in ['n','output']} for r in g]
        if settings and any(a!=settings[0] for a in settings[1:]):raise ValueError('Mixed settings in '+name)
    fits={name:{stage:slopes(g,stage) for stage in ['total','conditioning','construction','regional','query','factor']} for name,g in groups.items()}
    for name in ['H²','Streamed locality']:
        fits[name]['large_total']=slopes([r for r in groups[name] if r['n']>=32768])
    auxiliary=[]
    for folder in ['original','bandwidth','hybrid-float32']:
        for path in sorted((root/folder).glob('*.json')):
            if path.name.endswith('.status.json'):continue
            d=json.loads(path.read_text())
            auxiliary.append(dict(source=str(path),sha256=digest(path),arguments=d['arguments'],
                complete=d.get('complete'),failure=d.get('failure'),quality=d.get('quality'),
                total_seconds=d.get('total_seconds'),stages=d['stages']))
    (root/'manifest.json').write_text(json.dumps(dict(rows=rows,limits=limits,auxiliary=auxiliary,hardware=hardware,slopes=fits,
        fit_scope='Descriptive finite-range fits; not asymptotic proofs. Same hardware per fitted series.',
        workload='1,048,576 marginal mean/variance/CDF queries, one fixed contact region; includes preparation',
        model_scope='Cross-model accuracy/calibration unmatched; original resolution unvalidated'),indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(15,10));ax,dx,ix,gx=axes.ravel()
    for name,g in groups.items():
        if not g:continue
        x=[r['n'] for r in g];y=[r['total'] for r in g];free=g[0]['model']=='free-space'
        label=name
        if name in ['H²','Streamed locality']:label+=f" (slope {fits[name]['large_total']['fit']:.2f})"
        ax.loglog(x,y,'o-' if free else 's--',color=COLORS[name],label=label,ms=4,lw=2)
    controls=[]
    for rel in ['tuned/sparse-gpu-32768-0.json','tuned/sparse-gpu-77824-0.json']:
        d=json.loads((OLD/rel).read_text())
        if not d['independent_audit']['passes']:raise ValueError('Invalid cached locality control')
        controls.append((d['arguments']['n'],d['total_seconds']))
    ax.loglog(*zip(*controls),linestyle='none',marker='v',mfc='none',color=COLORS['Streamed locality'],label='Cached locality controls (not fitted)')
    ax.set(title='Complete mean + variance workload',xlabel='Observations N',ylabel='Seconds including preparation')
    ax.legend(fontsize=7.5,loc='upper left',ncol=2);ax.set_xlim(1500,1.2e6)
    ax.axvline(98304,color='#009E73',ls=':',alpha=.6)
    ax.text(105000,.65,'Dense allocation\nfails at 98k',fontsize=8,color='#009E73')
    dense=groups['GeoSPR Cholesky'];x=np.array([r['n'] for r in dense])
    for key,label in [('factor','Factorization'),('matrix','Matrix construction'),('regional','Regional cache'),('total','Complete workload')]:
        y=[r[key] for r in dense];dx.loglog(x,y,'o-',label=label,ms=4)
    high=[r for r in dense if r['n']>=32768]
    if len(high)>1:
        power=slope([r['n'] for r in high],[r['factor'] for r in high])
        dx.text(.04,.96,f'Factor fit, N ≥ 32k: {power:.2f}\nReference slope: 3',va='top',transform=dx.transAxes,fontsize=10)
        guide=np.array([32768,77824]);dx.loglog(guide,high[0]['factor']*(guide/guide[0])**3,':',color='black',label='slope 3 guide')
    dx.xaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
    dx.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1000:g}k'))
    dx.xaxis.set_minor_formatter(NullFormatter())
    dx.set(title='Dense costs: cubic factorization emerges',xlabel='Observations N',ylabel='Seconds');dx.legend(fontsize=8,loc='lower right')
    for name in ['H²','Streamed locality']:
        g=groups[name];x=np.array([r['n'] for r in g]);y=np.array([r['interactions'] for r in g])
        ix.loglog(x,y,'o-',color=COLORS[name],label=f'{name}: measured slope {slope(x,y):.2f}')
    guide=np.array([32768,796825]);ix.loglog(guide,1e8*(guide/guide[0]),'k:',label='slope 1 guide')
    ix.loglog(guide,1e8*(guide/guide[0])**2,'k--',alpha=.5,label='slope 2 guide')
    ix.set(title='Represented interaction entries',xlabel='Observations N',ylabel='Entries (different representations)');ix.legend(fontsize=8)
    grid=[]
    for p in sorted((root/'original').glob('full-grid*-8192.json')):
        if '.status.' in p.name:continue
        d=json.loads(p.read_text())
        if d.get('complete'):
            covariance=sum(v['self_seconds'] for v in d.get('original_profile',[]) if 'gstrf' in v['function'] or ('SuperLU' in v['function'] and 'solve' in v['function']))
            grid.append((d['arguments']['grid']**3,d['total_seconds'],d['arguments']['grid'],covariance))
    grid.sort()
    if grid:
        gx.loglog([r[0] for r in grid],[r[1] for r in grid],'x-',color='#303030',label='Complete workload')
        gx.loglog([r[0] for r in grid],[r[3] for r in grid],'o--',color='#777777',label='Covariance PDE stage')
        gx.legend(fontsize=8,loc='lower right')
        for g,t,k,_ in grid:gx.annotate(f'{k}³',(g,t),xytext=(5,6),textcoords='offset points',fontsize=9)
        gx.text(.04,.95,f"Measured fit: {slope([r[0] for r in grid],[r[1] for r in grid]):.2f}\nN = 8,192; CPU full covariance\nGrid refinement changes kernel width",transform=gx.transAxes,va='top',fontsize=10)
    gx.set(title='Original SPSR: grid size is a separate cost',xlabel='Scalar grid unknowns G',ylabel='Seconds');gx.margins(y=.12)
    for a in axes.ravel():a.grid(True,which='both',alpha=.18)
    fig.suptitle('Eagle scaling: observed slopes and practical limits',fontsize=17)
    fig.text(.5,.025,'Fixed large-case H²/locality settings; one million queries. Blackwell 1g.24gb; original SPSR on CPU. Single-run preparation timings.',ha='center',fontsize=10)
    fig.text(.5,.008,'Dashed curves use different models. No cross-model quality-matched speedup claim. Dotted guides are not measured data.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.045,1,.96));save(fig,root,'scaling')
    fig,axes=plt.subplots(2,4,figsize=(17,8))
    for ax,(name,g) in zip(axes.ravel(),groups.items()):
        if not g:continue
        for key,label in [('total','Total'),('construction','Construction / setup'),('conditioning','Conditioning'),('regional','Regional cache'),('query','All queries')]:
            pairs=[(r['n'],r[key]) for r in g if r[key]>0]
            if pairs:ax.loglog(*zip(*pairs),'o-',ms=3,label=label)
        ax.set(title=name,xlabel='N',ylabel='Seconds');ax.grid(True,which='both',alpha=.18)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Stage timings — end-to-end slopes need not equal solve slopes',fontsize=15)
    fig.text(.5,.015,'Original: setup includes mean PDE + assembly; conditioning is covariance LU/solves. Free-space Cholesky setup includes factorization.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.035,1,.96));save(fig,root,'stages')
    if (root/'query_sweep.json').exists():
        qr=json.loads((root/'query_sweep.json').read_text());fig,axes=plt.subplots(1,3,figsize=(15,4.8))
        for method in ['dense','h2-streamed','sparse-streamed']:
            group=[r for r in qr if r['method']==method and r['batches']==1];label=NAMES[method]
            axes[0].loglog([r['m'] for r in group],[r['median_query_seconds'] for r in group],'o-',label=label)
            group=[r for r in qr if r['method']==method and r['queries']==65536]
            axes[1].loglog([r['batches'] for r in group],[r['composed_total_seconds'] for r in group],'o-',label=label)
        for name in ['H²','Streamed locality']:
            g=groups[name];axes[2].plot([r['n'] for r in g],[r['mean_iterations'] for r in g],'o-',label=name+' mean')
            axes[2].plot([r['n'] for r in g],[r['rank'] for r in g],'s--',label=name+' cross rank')
        axes[0].set(title='Cached mean/variance/CDF queries',xlabel='Queries M in one batch',ylabel='Measured median seconds (3 runs)')
        axes[1].set(title='Reuse at N = 32,768',xlabel='65,536-query batches',ylabel='Measured prep + new query times (s)')
        axes[2].set(title='Ranks and mean iterations',xlabel='N',ylabel='Count',xscale='log')
        for ax in axes:ax.legend(fontsize=8);ax.grid(True,which='both',alpha=.18)
        fig.tight_layout();save(fig,root,'diagnostics')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='reproduction/results/asymptotic_regimes')
    main(p.parse_args().root)
