"""Plot provenance-pinned measured Eagle mean/variance timings, never extrapolate."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

matplotlib.rcParams['svg.fonttype']='none'


def load_rows(root):
    old=Path('reproduction/results/poissonrecon_large')
    selection=json.loads((old/'selection.json').read_text())
    for pair in selection['comparisons']:
        checked=json.loads((old/pair['output']).read_text())
        if not checked['passes']:raise ValueError('Failed reused full-query comparison')
        for role in ['candidate','baseline']:
            if checked[role+'_sha256']!=hashlib.sha256((old/pair[role]).read_bytes()).hexdigest():
                raise ValueError('Stale reused full-query comparison')
    rows=[];hardware=None;cloud_hash=None
    labels={'H2':'H²','Sparse':'Locality','Cholesky':'Free-space Cholesky'}
    for series in selection['series']:
        if series['dataset']!='eagle':continue
        for rel in series['records']:
            path=old/rel;d=json.loads(path.read_text())
            if not d.get('independent_audit',{}).get('passes'):raise ValueError(f'Failed reused audit: {path}')
            a=d['arguments']
            if a['queries']*a['batches']!=1048576:raise ValueError('Query workload mismatch')
            current=(d['device'],d['device_memory_bytes'],d['cpu_quota'],d['host_memory_limit'])
            if hardware is None:hardware=current;cloud_hash=a['point_cloud_sha256']
            if hardware!=current or cloud_hash!=a['point_cloud_sha256']:raise ValueError('Reused hardware/input mismatch')
            rows.append(dict(method=labels[series['method']],model='free-space',n=a['n'],seconds=d['total_seconds'],
                             preparation=d['preparation_seconds'],query_seconds=sum(d['query_seconds']),
                             quality=d['independent_audit'],source=str(path),reused=True,status='accepted',
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    names={'dense':'GeoSPR Cholesky + regional cache','sdd':'GeoSPR SDD + regional cache',
           'nufft-hybrid':'Cholesky–NUFFT variance field','nufft-cg':'NUFFT + analytic PCG + regional cache'}
    checks_path=root/'full_query_checks.json'
    if not checks_path.exists():raise ValueError('Run full-query verification before plotting')
    checks={c['candidate']:c for c in json.loads(checks_path.read_text())}
    selected={}
    for path in sorted((root/'periodic').glob('*.json')):
        if path.name.endswith('.status.json'):continue
        d=json.loads(path.read_text());a=d['arguments'];name=names[a['method']]
        if d.get('queries_total')!=1048576:raise ValueError('Query workload mismatch')
        if (d.get('device'),d.get('device_memory'),d['cpu_quota'],d['host_limit'])!=hardware:
            raise ValueError('Periodic hardware mismatch')
        if d['point_cloud_metadata']['normalized_sha256']!=cloud_hash:raise ValueError('Periodic input mismatch')
        if d.get('complete') and d.get('quality',{}).get('passes'):
            if a['method']!='dense':
                check=checks.get(str(path))
                if check is None:raise ValueError(f'Missing full-query check: {path}')
                if (check['candidate_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest() or
                    check['reference_sha256']!=hashlib.sha256(Path(check['reference']).read_bytes()).hexdigest()):
                    raise ValueError('Stale full-query check')
                if not check['passes']:continue
            if not d.get('gpu_process_audit',{}).get('sampled_isolation_pass'):
                raise ValueError(f'Missing GPU isolation audit: {path}')
            row=dict(method=name,model='periodic',n=a['n'],seconds=d['total_seconds'],
                     preparation=d['preparation_seconds'],query_seconds=sum(d['query_seconds']),
                     quality=d['quality'],source=str(path),reused=False,status='accepted',
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            key=(name,a['n'])
            if key not in selected or row['seconds']<selected[key]['seconds']:selected[key]=row
    rows+=list(selected.values())
    for path in sorted((root/'original').glob('full-grid16-*.json')):
        if path.name.endswith('.status.json'):continue
        d=json.loads(path.read_text())
        if not d.get('complete'):continue
        if (d['cpu_quota'],d['host_limit'])!=hardware[2:]:raise ValueError('Original CPU hardware mismatch')
        if d['point_cloud_metadata']['normalized_sha256']!=cloud_hash:raise ValueError('Original input mismatch')
        rows.append(dict(method='Sellán–Jacobson, 16³ grid (CPU)',model='original-grid',n=d['arguments']['n'],
                         seconds=d['total_seconds'],preparation=d['preparation_seconds'],query_seconds=sum(d['query_seconds']),
                         quality=None,source=str(path),reused=False,status='native, resolution unvalidated',
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return sorted(rows,key=lambda r:(r['method'],r['n']))


def main(root):
    root=Path(root);rows=load_rows(root)
    colors={'H²':'#0072B2','Locality':'#D55E00','Free-space Cholesky':'#777777',
            'GeoSPR Cholesky + regional cache':'#009E73','GeoSPR SDD + regional cache':'#CC79A7',
            'Cholesky–NUFFT variance field':'#E69F00','NUFFT + analytic PCG + regional cache':'#7A3DB8',
            'Sellán–Jacobson, 16³ grid (CPU)':'#303030'}
    fig,(ax,gridax)=plt.subplots(1,2,figsize=(15,7),gridspec_kw={'width_ratios':[2.4,1]})
    slopes={}
    for name,color in colors.items():
        group=sorted([r for r in rows if r['method']==name],key=lambda r:r['n'])
        if not group:continue
        ns=[r['n'] for r in group];ts=[r['seconds'] for r in group]
        model=group[0]['model'];style='-' if model=='free-space' else '--'
        marker='x' if model=='original-grid' else ('s' if model=='periodic' else 'o')
        label=name
        if name in ['H²','Locality']:
            exponent=np.polyfit(np.log(ns),np.log(ts),1)[0]
            label+=f' (observed slope {exponent:.2f})'
        ax.loglog(ns,ts,style,marker=marker,color=color,label=label,lw=2,ms=6)
        if len(ns)>1:
            slopes[name]=dict(local_loglog_fit=float(np.polyfit(np.log(ns),np.log(ts),1)[0]),
                              first_n=ns[0],last_n=ns[-1],scope='Descriptive finite-range fit; settings may vary; no asymptotic proof')
    ax.annotate('8.08× vs locality\nat full Eagle',xy=(796825,305.044),xytext=(120000,140),
                arrowprops=dict(arrowstyle='->',color=colors['H²']),color=colors['H²'],fontsize=10)
    ax.set(xlabel='Oriented observations N (nested Eagle subsets)',ylabel='Seconds: preparation + mean + variance + 1,048,576 queries',
           title='Measured end-to-end workload • fixed contact region')
    ax.set_xlim(1500,1.15e6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1000:g}k' if x>=1000 else f'{x:g}'))
    ax.legend(loc='upper left',fontsize=8,framealpha=.94)
    gridrows=[]
    for path in sorted((root/'original').glob('full-grid*-8192.json')):
        d=json.loads(path.read_text())
        if not d.get('complete'):continue
        g=d['arguments']['grid'];gridrows.append((g,d['total_seconds'],d['grid']['kernel_sigma']))
    gridrows.sort()
    if gridrows:
        gridax.loglog([g**3 for g,_,_ in gridrows],[t for _,t,_ in gridrows],'x-',color='#303030',lw=2)
        for g,t,k in gridrows:gridax.annotate(f'{g}³',xy=(g**3,t),xytext=(4,7),textcoords='offset points',fontsize=9)
    gridax.set(title='Original SPSR: grid cost\nN = 8,192, same million queries',xlabel='Scalar grid unknowns G',ylabel='Total seconds (CPU)')
    gridax.margins(x=.15,y=.12)
    gridax.text(.04,.96,'Refining G also changes the kernel.\nNative-grid timings; quality unmatched.',
                transform=gridax.transAxes,fontsize=9,va='top',
                bbox=dict(facecolor='white',edgecolor='#bbbbbb',alpha=.94))
    for axis in [ax,gridax]:axis.grid(True,which='both',alpha=.18)
    fig.suptitle('Eagle: marginal mean-and-variance performance across method families',fontsize=15)
    fig.text(.5,.045,'Solid: free-space Matérn model. Dashed: periodic GeoSPR or original grid SPSR. No cross-model speedup ratio is established.',ha='center',fontsize=10)
    fig.text(.5,.018,'GPU: Blackwell 1g.24gb; original SPSR: CPU. Single runs. No timing extrapolation. Regional caching is granted to Cholesky and SDD.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.96))
    for suffix in ['png','pdf']:fig.savefig(root/f'comparison.{suffix}',dpi=180)
    svg=root/'comparison.svg'
    fig.savefig(svg,metadata={'Date':None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    (root/'summary.json').write_text(json.dumps(dict(rows=rows,slopes=slopes,original_grid_sweep=gridrows),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='reproduction/results/eagle_comparison')
    main(p.parse_args().root)
