"""Audit and plot the fixed-query mixed-H2 crossover, including rejected pilots."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr
from geospsr.nonperiodic import FreeSpacePoissonMatern
from reproduction.partial_competitor_common import gates
from reproduction.partial_occupancy import ROOT, CACHE

RESULTS=Path('reproduction/results/mixed_h2')
PREVIOUS=Path('reproduction/results/partial_competitors')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def summarize(path,source_root):
 d=json.loads(path.read_text());a=d['arguments'];c=d['contract'];field=path.with_suffix('.npz')
 assert c['n']==a['n'] and c['group']=='test_side' and c['contacts']==32 and c['queries']==128
 for key,value in [('lengthscale',.48),('amplitude',1/3),('normal_noise_variance',.04),('scalar_noise_variance',.0001)]:
  assert np.isclose(c[key],value,rtol=0,atol=1e-15),(path,key)
 assert c['query_indices']==np.linspace(0,6073,128,dtype=int).tolist(),path
 assert d['field_sha256']==sha(field),path
 for key,name in [('dataset_sha256','dataset.json'),('selected_sha256','selected.json'),('controls_sha256','controls.json'),('query_sha256','test_side-queries.npz')]:
  assert c[key]==sha(ROOT/name),(path,key)
 for name,h in d['source_sha256'].items():assert sha(source_root/'sources'/(h+'.py'))==h,(path,name)
 f=dict(np.load(field));model=FreeSpacePoissonMatern(.48,1/3)
 fit=json.loads((ROOT/'controls.json').read_text())['fits']['spatial']
 if a.get('method')=='reference':
  assert d['reference_converged']
  last=d['refinement_history'][-1]
  assert last['max_relative_residual']<=1e-9
  assert last['change_from_previous']['cdf']<=1e-6 and last['change_from_previous']['calibrated_cdf']<=1e-6
 refpath=Path(c.get('reference',ROOT/'sparse_queries'/f"dense-{a['n']}.npz"))
 if refpath.exists():
  assert c['reference_sha256']==sha(refpath),path
  assert c['reference_record_sha256']==sha(refpath.with_suffix('.json')),path
  error=gates(f['mean'],f['variance'],dict(np.load(refpath)),model,fit)
 else:
  assert a['method']=='reference' and d['reference_converged']
  last=d['refinement_history'][-1]
  assert last['max_relative_residual']<=1e-9
  assert last['change_from_previous']['cdf']<=1e-6 and last['change_from_previous']['calibrated_cdf']<=1e-6
  valid=bool(np.isfinite(f['mean']).all() and np.isfinite(f['variance']).all() and np.all(f['variance']>0))
  error=dict(valid_moments=valid,passes=valid)
 audit=d['gpu_audit']
 assert audit['samples']>=2 and not audit['errors'] and not audit['foreign_process_samples'],path
 assert audit['device_uuid']=='MIG-c7b1828a-c900-51c9-b6b8-db042ea0fb2b',path
 if path.name.startswith('repeat-') and a.get('method')=='h2':
  assert a['dtype']=='float32' and a['storage']=='cached' and a['iterations']==10 and a['h2_tolerance']==1e-6,path
 accepted=error['passes'] and audit['sampled_isolation_pass'] and d['status']=='completed'
 assert bool(accepted)==d['accepted'],path
 ids=np.asarray(c['query_indices']);labels=np.load(ROOT/'test_side-queries.npz')['inside'][ids]
 z=-f['mean']/np.sqrt(f['variance']);p=ndtr(z);cal=ndtr(np.exp(fit['log_scale'])*z+fit['bias'])
 return dict(file=str(path),n=a['n'],method=a.get('method','dense'),dtype=a['dtype'],storage=a.get('storage'),iterations=a.get('iterations'),
  seconds=d['total_seconds'],accepted=accepted,isolated=audit['sampled_isolation_pass'],
  gpu_gib=max(o['own_memory_bytes'] or 0 for o in audit['observations'])/2**30,
  host_gib=max(o['own_host_rss_bytes'] for o in audit['observations'])/2**30,
  brier=float(np.mean((p-labels)**2)),calibrated_brier=float(np.mean((cal-labels)**2)),**error)

def main():
 metadata=json.loads((ROOT/'dataset.json').read_text())
 assert sha(CACHE/'test_side.npy')==metadata['groups']['test_side']['sha256']
 paths=[p for p in sorted(RESULTS.glob('*.json')) if 'arguments' in json.loads(p.read_text())]
 rows=[summarize(p,RESULTS) for p in paths]
 reused=[summarize(PREVIOUS/f'repeat-mixed-32768-{i}.json',PREVIOUS) for i in [1,2,3]]
 rows+=reused
 (RESULTS/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
 columns=['file','n','method','dtype','storage','iterations','seconds','accepted','isolated','gpu_gib','host_gib','brier','calibrated_brier','valid_moments','passes','mean','variance_over_prior','cdf','calibrated_cdf']
 with (RESULTS/'summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=columns,lineterminator='\n');w.writeheader();w.writerows(rows)
 primary={};fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 for method,label,color in [('h2','Mixed H², cached','#176c91'),('dense','Mixed Cholesky','#cc6e1f')]:
  rr=[r for r in rows if r['method']==method and (Path(r['file']).name.startswith('repeat-'))]
  assert all(r['accepted'] for r in rr)
  points=[]
  for n in sorted({r['n'] for r in rr}):
   v=[r for r in rr if r['n']==n];assert len(v)==3,(method,n)
   points.append(dict(n=n,seconds=float(np.median([x['seconds'] for x in v])),minimum=min(x['seconds'] for x in v),maximum=max(x['seconds'] for x in v),
    gpu_gib=float(np.median([x['gpu_gib'] for x in v])),host_gib=float(np.median([x['host_gib'] for x in v])),
    max_cdf=max(x['cdf'] for x in v),max_calibrated_cdf=max(x['calibrated_cdf'] for x in v),max_variance_over_prior=max(x['variance_over_prior'] for x in v),
    brier=float(np.median([x['brier'] for x in v])),calibrated_brier=float(np.median([x['calibrated_brier'] for x in v]))))
  ns=[p['n'] for p in points];tt=[p['seconds'] for p in points]
  slope=float(np.polyfit(np.log(ns),np.log(tt),1)[0]);primary[method]=dict(points=points,log_log_slope=slope)
  axes[0].loglog(ns,tt,'o-',color=color,label=f'{label} (slope {slope:.2f})')
  axes[0].fill_between(ns,[p['minimum'] for p in points],[p['maximum'] for p in points],color=color,alpha=.2)
  axes[1].loglog(ns,[p['gpu_gib'] for p in points],'o-',color=color,label=label)
 axes[1].axhline(23.625,color='gray',linestyle='--',label='Allocated GPU capacity')
 for ax in axes:ax.set_xlabel('Normal observations N');ax.grid(True,which='both',alpha=.2);ax.legend(fontsize=8)
 axes[0].set_ylabel('Fresh setup + 128 mean/variance/CDF queries (s)')
 axes[1].set_ylabel('Sampled peak process GPU memory (GiB)')
 fig.savefig(RESULTS/'crossover.png',dpi=180);fig.savefig(RESULTS/'crossover.pdf');plt.close(fig)
 (RESULTS/'primary.json').write_text(json.dumps(primary,indent=2)+'\n')
 print('Audited',len(paths),'new trials and',len(reused),'reused controls; accepted',sum(r['accepted'] for r in rows),'isolated',sum(r['isolated'] for r in rows))

if __name__=='__main__':main()
