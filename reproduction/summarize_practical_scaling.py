"""Render measured scaling records; omit failed/incomplete runs from curves."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def median(runs, key):
    return float(np.median([r[key] for r in runs]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,required=True)
    a=p.parse_args()
    records=[json.loads(p.read_text()) for p in sorted((a.results/'solver').glob('*/results.json'))]
    valid=[r for r in records if r['complete'] and r['gpu_audit']['sampled_isolation_pass']]
    valid.sort(key=lambda r:r['n'])
    summary={'solver':[],'variance':[]}
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    names={'cholesky':'Cholesky (best measured precision)','sdd':'SDD (updates only)','nufft_refined':'NUFFT + analytic refinement','analytic_cg':'Analytic CG control'}
    for name,label in names.items():
        ns=[];solve=[];total=[];low=[];high=[];total_low=[];total_high=[]
        for r in valid:
            method=r['methods'].get(name,{})
            if name=='cholesky':
                choices=[r['methods'].get(k,{}) for k in ('cholesky','cholesky_float64')]
                choices=[m for m in choices if 'solve_seconds' in m and m['quality']['pass']]
                if choices:method=min(choices,key=lambda m:m['prepare_seconds']+m['solve_seconds'])
            runs=method.get('runs',[method] if 'solve_seconds' in method else [])
            if not runs or not all(v['quality']['pass'] for v in runs):continue
            ns.append(r['n']);solve.append(median(runs,'solve_seconds'))
            # Give SDD reusable compilation and verification for free in the
            # comparison curve. Its full costs remain explicit in the records.
            total.append((0. if name=='sdd' else method['prepare_seconds'])+solve[-1])
            low.append(min(v['solve_seconds'] for v in runs));high.append(max(v['solve_seconds'] for v in runs))
            total_low.append(total[-1]-solve[-1]+low[-1]);total_high.append(total[-1]-solve[-1]+high[-1])
            summary['solver'].append({'n':r['n'],'method':name,'update_seconds':solve[-1],
                'prepare_plus_update_seconds':total[-1],
                'verified_solve_seconds':median(runs,'verified_solve_seconds') if name=='sdd' else solve[-1],
                'verification_setup_seconds':method.get('verification_setup_seconds',0.),
                'mean_error_max':max(v['quality']['mean_relative_l2'] for v in runs)})
        if ns:
            line,=axes[0].loglog(ns,solve,'o-',label=label)
            axes[0].fill_between(ns,low,high,color=line.get_color(),alpha=.15)
            line,=axes[1].loglog(ns,total,'o-',label=label)
            axes[1].fill_between(ns,total_low,total_high,color=line.get_color(),alpha=.15)
    for name,label in [('sdd','SDD updates'),('nufft_refined','NUFFT CG iterations (all corrections)')]:
        ns=[];its=[]
        for r in valid:
            runs=r['methods'][name].get('runs',[])
            if not runs or not all(v['quality']['pass'] for v in runs):continue
            ns.append(r['n'])
            vals=[v['iterations'] if name=='sdd' else v['solver']['initial']['iterations']+
                  sum(w['iterations'] for w in v['solver']['refinement']['inner_solves']) for v in runs]
            its.append(float(np.median(vals)))
        axes[2].loglog(ns,its,'o-',label=label,color='tab:orange' if name=='sdd' else 'tab:green')
    for ax,title,ylabel in zip(axes,['Solve only (Cholesky factor already available)','Prepare + solve; SDD updates only','Observed iteration counts'],['seconds','seconds','iterations (different work per iteration)']):
        ax.set(xlabel='observations N',ylabel=ylabel,title=title);ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Fixed analytic model; full residual ≤ 1e−4 and held-out mean error ≤ 0.2%')
    fig.savefig(a.results/'solver_scaling.png',dpi=180);fig.savefig(a.results/'solver_scaling.pdf');plt.close(fig)
    variance_files=sorted((a.results/'variance').glob('*/results.json'))
    for path in variance_files:
        r=json.loads(path.read_text())
        if not r['complete'] or not r.get('gpu_audit',{}).get('sampled_isolation_pass'):continue
        fig,ax=plt.subplots(figsize=(7,4.8),layout='constrained')
        methods=list(r['measurements'][0]['seconds'])
        for name in methods:
            ms=[];ts=[]
            for m in sorted({v['queries'] for v in r['measurements']}):
                rows=[v for v in r['measurements'] if v['queries']==m]
                if not all(v['quality'][name]['pass'] and v['heldout_exact_quality'][name]['pass'] for v in rows):continue
                ms.append(m);ts.append(float(np.median([v['seconds'][name] for v in rows])))
                summary['variance'].append({'run':path.parent.name,'method':name,'queries':m,'seconds':ts[-1]})
            if ms:ax.loglog(ms,ts,'o-',label=name.replace('cholesky_','').replace('_',' '))
        ax.set(xlabel='fresh query positions M',ylabel='seconds per batch (preparation excluded)',title=f'Marginal mean and variance, N={r["n"]}')
        ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=8)
        fig.savefig(a.results/f'variance_{path.parent.name}.png',dpi=180)
        fig.savefig(a.results/f'variance_{path.parent.name}.pdf');plt.close(fig)
    (a.results/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
