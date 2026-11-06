import hashlib,json,pathlib
import numpy as np
from scipy.special import ndtr
root=pathlib.Path('reproduction/results/real_partial')
def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
summary=json.loads((root/'summary.json').read_text());selection=json.loads((root/'selected.json').read_text());dataset=json.loads((root/'dataset.json').read_text())
assert selection['ell']==.48 and selection['sigma']==.2
sourcepaths={};files=[]
for row in summary['sparse_queries']:
 p=pathlib.Path(row['path']);r=json.loads(p.read_text());a=r['arguments']
 assert row['sampled_gpu_isolation'] and r['production_gpu_audit']['sampled_isolation_pass']
 assert r['cdf_timed'] and r['queries']==128
 assert abs(r['total_seconds']-sum(r[k] for k in ['setup_seconds','contact_and_mean_seconds','query_seconds']))<1e-10
 field=np.load(p.with_suffix('.npz'))
 assert all(np.isfinite(field[k]).all() and field[k].shape==(128,) for k in ['mean','variance','shortcut_variance'])
 assert np.all(field['variance']>0) and np.all(field['shortcut_variance']>0)
 assert abs(ndtr(-field['mean']/np.sqrt(field['variance'])).sum()-r['probability_checksum'])<1e-9
 if a['method']!='dense':
  assert row['passes'] and row['cdf']<=.001 and row['calibrated_cdf']<=.001 and row['variance_over_prior']<=1e-4
  assert a['iterations']==8 and a['rank']==100 and a['neighbors']==80 and a['solve_tolerance']==1e-9
  if a['method']=='h2':assert a['h2_tolerance']==1e-6
 for src,h in r['source_sha256'].items():
  current=pathlib.Path(src);snapshot=root/'source_snapshots'/f'{h}.py'
  if current.exists() and digest(current)==h:sourcepaths[h]=str(current)
  else:
   assert snapshot.exists() and digest(snapshot)==h,(p,src,h)
   sourcepaths[h]=str(snapshot)
 files.append(dict(record=str(p),record_sha256=digest(p),field_sha256=digest(p.with_suffix('.npz')),numerical_gates=row['passes'],reference=a['method']=='dense',sampled_gpu_isolation=True))
assert len(files)==13
cache=pathlib.Path.home()/'.cache/fast-stochastic-psr/stanford-armadillo'
for group,g in dataset['groups'].items():
 assert digest(root/f'{group}-queries.npz')==g['query_sha256']
 assert digest(cache/f'{group}.npy')==g['sha256']
for path,h in summary['artifact_sha256'].items():assert digest(root/path)==h, path
manifest=dict(schema_version=1,primary_records=files,primary_source_versions=sourcepaths,
 shared_inputs=dict(dataset_sha256=digest(root/'dataset.json'),selected_sha256=digest(root/'selected.json'),
                   controls_sha256=digest(root/'controls.json'),query_sha256={g:v['query_sha256'] for g,v in dataset['groups'].items()}),
 input_contract='Selected model and prepared inputs remained unchanged during these runs. Workers verified raw prepared-group and query hashes on loading. This final manifest records the shared model contract; it does not retrofit unrecorded per-run model digests.',
 checks=['13 primary records: all query fields finite with positive variances; 9 nonreference candidates pass raw/calibrated CDF and variance gates',
         'All primary records pass sampled GPU isolation; CDF checksums were timed',
         'All recorded primary source versions resolve to current or archived content',
         'Raw prepared inputs, query hashes and report artifact hashes match',
         '24 targeted/regression tests passed; see tests.log'],
 limitations=['Single timing run per primary configuration; process sampling can miss short-lived GPU work',
              'Reconstructed reference from same acquisition; no population calibration claim',
              '128-query fresh-preparation workload; whole-object cache trials still fail accuracy'],
 artifacts={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest.json'})
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('AUDIT PASSED:',len(files),'primary runs,',len(sourcepaths),'source versions,',len(manifest['artifacts']),'artifact hashes')
