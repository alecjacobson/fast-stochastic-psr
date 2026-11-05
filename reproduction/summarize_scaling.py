"""Compact measured statistics for the issue-104 report."""
import json
from pathlib import Path
import numpy as np


def main(root=Path('reproduction/results/asymptotic_regimes')):
    d=json.loads((root/'manifest.json').read_text());rows=d['rows'];result={}
    for name in sorted({r['method'] for r in rows}):
        group=sorted([r for r in rows if r['method']==name],key=lambda r:r['n'])
        result[name]=dict(points=[dict(n=r['n'],seconds=r['total'],conditioning=r['conditioning'],
            regional=r['regional'],query=r['query'],interactions=r['interactions'],cross_rank=r['rank'],
            h2_max_rank=r['h2_max_rank'],mean_iterations=r['mean_iterations']) for r in group],
            slopes=d['slopes'][name])
    periodic=[]
    for method in ['sdd','nufft-cg','nufft-hybrid']:
        for n in [2048,8192,32768]:
            path=root/f'periodic/{method}-{n}.json'
            if not path.exists():continue
            v=json.loads(path.read_text())
            item=dict(method=method,n=n,complete=v.get('complete'),quality=v.get('quality'),total=v.get('total_seconds'))
            if method=='sdd':
                item.update(iterations=v['checkpoints'][-1]['iterations'],seconds_per_iteration=v.get('seconds_per_iteration'))
            if method=='nufft-cg':
                item.update(breakdown=v.get('solve_breakdown'),analytic_variance_iterations=[b['analytic']['iterations'] for b in v['solve_blocks']],mean_iterations=v.get('mean_solve'))
            if method=='nufft-hybrid':item.update(variance_profile=v.get('variance_series'))
            periodic.append(item)
    dense=[r for r in rows if r['key']=='dense' and r['n']>=32768]
    result['dense_large_factor_fit']=float(np.polyfit(np.log([r['n'] for r in dense]),np.log([r['factor'] for r in dense]),1)[0])
    result['periodic_diagnostics']=periodic
    result['bandwidth']=json.loads((root/'bandwidth_checks.json').read_text());result['subspace']=json.loads((root/'subspace_checks.json').read_text())
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for k,v in result.items():
        if isinstance(v,dict) and 'slopes' in v:print(k,v['slopes'].get('large_total',v['slopes']['total']))
    print('dense factor large',result['dense_large_factor_fit'])


if __name__=='__main__':main()
