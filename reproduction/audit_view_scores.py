"""Recompute archived next-view diagnostics without changing source records."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reproduction/results/view_score_audit'


def scores(probability,distance):
    mask=(probability>=.05)&(probability<=.95)
    return (np.trapezoid(mask,distance,axis=1),
            np.trapezoid(mask.astype(float),distance,axis=1))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    index_path=OUT/'geometry_index.json'
    geometry_index=json.loads(index_path.read_text()) if index_path.exists() else {}
    output_index={}
    for p in (ROOT/'reproduction/results').rglob('results.json'):
        d=json.loads(p.read_text())
        output=d.get('args',{}).get('output')
        if output: output_index.setdefault(output,[]).append(p.parent)
    records=[];skipped=[]
    for p in sorted((ROOT/'reproduction/results').rglob('*.npz')):
        if p.name not in ('events.npz','fields.npz','applications.npz'): continue
        data=np.load(p)
        keys=[k for k in data.files if k=='rays' or k.endswith('_rays')]
        keys=[k for k in keys if data[k].ndim==3 and np.all((data[k]==0)|(data[k]==1))]
        application=p.name=='applications.npz' and 'transmittance' in data
        if not keys and not application: continue
        distance=None;geometry_source=None
        meta={};meta_path=p.parent/'results.json'
        if meta_path.exists(): meta=json.loads(meta_path.read_text())
        if 'distance' in data:
            distance=data['distance'];geometry_source='distance in archived NPZ'
        elif p.name=='events.npz' and (p.parent/'fields.npz').exists():
            fields=np.load(p.parent/'fields.npz')
            if 'distance' in fields:
                distance=fields['distance'];geometry_source='distance in sibling archived fields.npz'
            elif 'conditional_solves' in p.parts and all(data[k].shape[1:]==(24,256) for k in keys):
                distance=np.linspace(0,5,256,dtype='f')
                geometry_source='inferred from benchmark_conditional_solves.py default application_geometry float32 grid'
        elif meta.get('geometry_sha256') and meta.get('args',{}).get('scratch'):
            geometry=Path(meta['args']['scratch'])/'geometry.npz'
            if geometry.exists() and hashlib.sha256(geometry.read_bytes()).hexdigest()==meta['geometry_sha256']:
                distance=np.load(geometry)['distance'];geometry_source={'original_geometry_sha256':meta['geometry_sha256']}
            elif meta['geometry_sha256'] in geometry_index:
                saved=geometry_index[meta['geometry_sha256']]
                path=OUT/saved['distance_file']
                assert hashlib.sha256(path.read_bytes()).hexdigest()==saved['distance_file_sha256']
                distance=np.load(path);geometry_source={'original_geometry_sha256':meta['geometry_sha256']}
        elif application:
            # The stored array has 24x256 probabilities; the producing script
            # explicitly uses t=np.linspace(0,5,256). Keep this inferred provenance.
            if data['transmittance'].shape==(24,256):
                distance=np.linspace(0,5,256);geometry_source='inferred from applications.py fixed float64 ray grid, not stored geometry'
        if distance is None:
            skipped.append({'artifact':str(p.relative_to(ROOT)),'reason':'no verified or documented distance grid'})
            continue
        name=hashlib.sha256(distance.tobytes()).hexdigest()[:16]+'.npy'
        np.save(OUT/name,distance)
        if isinstance(geometry_source,dict):
            geometry_index[geometry_source['original_geometry_sha256']]={'distance_file':name,
                'distance_file_sha256':hashlib.sha256((OUT/name).read_bytes()).hexdigest()}
        entries={}
        for key in (['transmittance'] if application else keys):
            probability=data[key] if application else data[key].mean(0)
            old,new=scores(probability,distance)
            entries[key]={'legacy':old.tolist(),'numeric':new.tolist(),
                          'legacy_selected':int(old.argmax()),'numeric_selected':int(new.argmax()),
                          'selection_changed':int(old.argmax())!=int(new.argmax()),
                          'numeric_regret_of_legacy_selection':float(new.max()-new[int(old.argmax())]),
                          'numeric_ties_within_1e-12':np.flatnonzero(new.max()-new<=1e-12).tolist()}
            if application and 'scores' in data:
                error=float(np.max(np.abs(old-data['scores'])))
                entries[key]['legacy_max_error_vs_stored']=error
                assert error<1e-10,(p,'legacy score recomputation differs from stored scores',error)
        pairs=[]
        reference_key='reference_rays' if 'reference_rays' in entries else None
        if reference_key is None and isinstance(meta.get('reference'),str) and meta['reference']+'_rays' in entries:
            reference_key=meta['reference']+'_rays'
        external=None
        if reference_key is None and not application:
            ref=meta.get('args',{}).get('reference')
            if ref:
                candidates=output_index.get(ref,[])
                direct=ROOT/ref
                if not Path(ref).is_absolute() and (direct/'fields.npz').exists(): candidates=[direct]
                if len(candidates)==1 and (candidates[0]/'fields.npz').exists():
                    r=np.load(candidates[0]/'fields.npz')
                    if 'rays' in r and r['rays'].shape[1:]==data['rays'].shape[1:]:
                        old,new=scores(r['rays'].mean(0),distance)
                        external={'legacy':old,'numeric':new,'path':str((candidates[0]/'fields.npz').relative_to(ROOT))}
        for key,entry in entries.items():
            if key==reference_key: continue
            ref=entries[reference_key] if reference_key else external
            if ref is None: continue
            old,new=np.asarray(ref['legacy']),np.asarray(ref['numeric'])
            old_regret=float(old.max()-old[entry['legacy_selected']])
            new_regret=float(new.max()-new[entry['numeric_selected']])
            threshold=2*float(distance[1]-distance[0])
            pairs.append({'candidate':key,'reference':reference_key or external['path'],
                          'legacy_regret':old_regret,'numeric_regret':new_regret,'threshold':threshold,
                          'legacy_regret_gate':old_regret<=threshold,'numeric_regret_gate':new_regret<=threshold})
        records.append({'artifact':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                        'distance_file':name,'geometry_provenance':geometry_source,'scores':entries,'comparisons':pairs})
    changed=[{'artifact':r['artifact'],'key':key,**entry} for r in records for key,entry in r['scores'].items() if entry['selection_changed']]
    reclassified=[{'artifact':r['artifact'],**p} for r in records for p in r['comparisons'] if p['legacy_regret_gate']!=p['numeric_regret_gate']]
    result={'records':records,'skipped':skipped,'changed_selections':changed,'reclassified_regret_gates':reclassified,
            'unresolved':['Sequential scan stores decisions but no per-step transmittance; regenerate its trajectory under numeric scoring.'],
            'scope':'diagnostic score audit only; raw fields/events and prior timing records unchanged'}
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    index_path.write_text(json.dumps(geometry_index,indent=2)+'\n')
    text=f'''# Numeric next-view score audit

NumPy trapezoidal integration of a Boolean mask undercounts consecutive true
intervals: the mask `[True, True]` on `[0, 1]` integrates to 0.5, whereas its
numeric indicator integrates to 1.0. The next-view diagnostic now converts
the 5%–95% uncertainty mask to floating point before integration. New paired
diagnostics record quality-gate version 3. This changes score units and can
change selection/regret; it does not change fields or ray/collision probabilities.

This audit recomputes **{len(records)} archived artifacts**, including all arrays
in each, without overwriting source records. **{len(changed)} selections change**
and **{len(reclassified)} two-ray-step regret gates change classification**.
**{len(skipped)} artifacts lack a verified/documented distance grid** and are
listed explicitly. Full scores, source hashes, distance arrays and provenance
are in [results.json](results.json). Application-demo grids inferred from the
fixed producing script are distinguished from stored/hashed geometry.

The sequential scan saved decisions but no per-step transmittance, so this
audit cannot reconstruct its adaptive trajectory. It requires a rerun; prior
scan-quality claims must not be treated as revalidated. That work remains in
[issue #7](https://github.com/alecjacobson/fast-stochastic-psr/issues/7).

Run `./reproduction/run_env.sh -m reproduction.audit_view_scores` to reproduce
the audit. Conditioning distance grids were recovered from original scratch
geometry only after checking its saved hash. This report archives those arrays
and [their hash index](geometry_index.json), so later runs can use them when
scratch geometry is unavailable. [Issue #28](https://github.com/alecjacobson/fast-stochastic-psr/issues/28)
tracks the correction and remaining coverage. This diagnostic is a proxy
acquisition score, not a validation of physical scanning performance.
'''
    (OUT/'README.md').write_text(text)
    print(json.dumps({'artifacts':len(records),'changed_selections':changed,'reclassified':reclassified,'skipped':skipped},indent=2))


if __name__=='__main__': main()
