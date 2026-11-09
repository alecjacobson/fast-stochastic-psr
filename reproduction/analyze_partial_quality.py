"""Apply the preregistered validation gates without reading any test fields."""
import json
from pathlib import Path
import numpy as np
from reproduction.partial_quality import OUT, occupancy, roots, CONTACTS
from reproduction.stanford_partial_data import digest


def balanced(record):
    return np.mean([record['geometry'][s]['mae'] for s in ['observed','occluded']])


def qualifies(candidate, baseline):
    checks = {'balanced_geometry': bool(balanced(candidate) <= .75*balanced(baseline))}
    for s in ['observed', 'occluded']:
        for k in ['mae','missing_fraction']:
            checks[s+'_'+k] = candidate['geometry'][s][k] <= baseline['geometry'][s][k]
    for s in ['all', 'occluded']:
        checks[s+'_brier'] = candidate['occupancy'][s]['brier'] <= baseline['occupancy'][s]['brier']+.005
    return dict(passes=all(checks.values()),checks=checks)


def analyze():
    panel = dict(np.load(OUT/'validation-panel.npz'))
    all_rows=[]
    for n in [768,16384]:
        rows=[]
        for path in sorted(OUT.glob(f'screen-n{n}-*.json')):
            data=json.loads(path.read_text())
            if data['status']!='completed':raise ValueError(path)
            assert data['gpu_audit']['sampled_isolation_pass']
            assert data['panel_sha256']==digest(OUT/'validation-panel.npz')
            assert data['field_sha256']==digest(path.with_suffix('.npz'))
            fields=dict(np.load(path.with_suffix('.npz')))
            for k, c in enumerate(CONTACTS):
                r=data['records'][k]
                assert r['contacts']==c
                assert occupancy(fields['mean'][:,k],fields['variance'][:,k],panel)==r['occupancy']
                assert roots(fields['probe_mean'][:,k],panel)[0]==r['geometry']
                rows.append(dict(n=n,ell=data['ell'],sigma=data['sigma'],source=str(path),**r))
        if len(rows)!=36:raise ValueError(f'incomplete factorial at {n}: {len(rows)}')
        baseline=next(r for r in rows if (r['ell'],r['sigma'],r['contacts'])==(.48,.2,32))
        for r in rows:r.update(qualification=qualifies(r,baseline),balanced_root_mae=float(balanced(r)))
        rows.sort(key=lambda r:(r['balanced_root_mae'],r['occupancy']['all']['brier']))
        all_rows+=rows
        print('N',n,'baseline',balanced(baseline),baseline['occupancy']['all']['brier'])
        for r in rows[:8]:print(r['ell'],r['sigma'],r['contacts'],r['balanced_root_mae'],r['occupancy']['all']['brier'],r['qualification']['passes'])
    promoted=[r for r in all_rows if r['n']==16384 and r['qualification']['passes']][:3]
    output=OUT/'screen-selection.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(dict(protocol_sha256=digest(OUT/'PROTOCOL.md'),rows=all_rows,promotion_order=promoted),indent=2,allow_nan=False)+'\n')


def promote():
    selection=json.loads((OUT/'screen-selection.json').read_text())
    baseline_path=OUT/'validation-old-70652-reference.json'
    baseline=json.loads(baseline_path.read_text())
    decisions=[]
    for i,candidate in enumerate(selection['promotion_order'],1):
        path=OUT/f'validation-candidate{i}-70652-reference.json'
        if not path.exists():raise ValueError(f'Run the next preregistered candidate: {candidate["ell"],candidate["sigma"],candidate["contacts"]}')
        r=json.loads(path.read_text())
        for p,record in [(baseline_path,baseline),(path,r)]:
            assert record['status']=='completed' and record['gpu_audit']['sampled_isolation_pass']
            assert record['field_sha256']==digest(p.with_suffix('.npz'))
            assert record['arguments']['group']=='validation' and record['arguments']['n']==70652
        assert (r['arguments']['ell'],r['arguments']['sigma'],r['arguments']['contacts'])==(candidate['ell'],candidate['sigma'],candidate['contacts'])
        quality=qualifies(r,baseline);decisions.append(dict(candidate=i,qualification=quality))
        if quality['passes']:
            chosen={k:candidate[k] for k in ['ell','sigma','contacts']}
            from datetime import datetime, timezone
            chosen.update(frozen_at_utc=datetime.now(timezone.utc).isoformat(),heldout_scope_sha256=digest(OUT/'HELDOUT-SCOPE.md'),normal_noise_variance=chosen['sigma']**2,scalar_noise_variance=.0001,amplitude=1/3,
                validation_record_sha256=digest(path),baseline_record_sha256=digest(baseline_path),
                screen_selection_sha256=digest(OUT/'screen-selection.json'),protocol_sha256=digest(OUT/'PROTOCOL.md'),
                decisions=decisions,scope='Frozen using validation views only; relative quality gates, no claim of externally certified geometry')
            output=OUT/'selected.json'
            if output.exists():raise FileExistsError(output)
            output.write_text(json.dumps(chosen,indent=2,allow_nan=False)+'\n')
            print('PROMOTED',chosen,flush=True)
            return
    raise RuntimeError('No promoted candidate passes the large-N validation gates')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',nargs='?',choices=['screen','promote'],default='screen')
    if p.parse_args().action=='screen':analyze()
    else:promote()
