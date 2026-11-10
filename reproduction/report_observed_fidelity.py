"""Replay observed-fidelity scores and verify retained experimental evidence."""
import json
from pathlib import Path

import numpy as np

from reproduction.observed_fidelity import OUT, fidelity, write
from reproduction.partial_quality import occupancy, roots
from reproduction.stanford_partial_data import digest


def main():
    validation=dict(np.load(OUT/'validation-panel.npz'))
    rows=[]
    for path in sorted(OUT.glob('screen*.json')):
        r=json.loads(path.read_text());f=np.load(path.with_suffix('.npz'))
        assert r['status']=='completed' and r['gpu_audit']['sampled_isolation_pass']
        assert r['field_sha256']==digest(path.with_suffix('.npz'))
        assert r['panel_sha256']==digest(OUT/'validation-panel.npz')
        for k,one in enumerate(r['records']):
            assert fidelity(f['fit_mean'][:,k],f['gradient'][:,:,k],validation)[0]==one['fidelity']
            assert occupancy(f['mean'][:,k],f['variance'][:,k],validation)==one['occupancy']
            assert roots(f['reference_mean'][:,k],validation)[0]==one['geometry']
            if not r['arguments']['control']:
                rows.append(dict(ell=r['arguments']['ell'],sigma=r['arguments']['sigma'],source=str(path),column=k,**one))
    selection=json.loads((OUT/'selection.json').read_text())
    assert len(rows)==32 and rows==selection['rows']
    passing=sorted([r for r in rows if r['fidelity']['passes']],key=lambda r:(r['fidelity']['p95'],r['fidelity']['normal_median_degrees']))
    assert passing[:3]==selection['promotion_order']
    large=[]
    for name in ['validation-candidate1','validation-refinement-check','test-side-selected']:
        path=OUT/(name+'.json');r=json.loads(path.read_text());f=np.load(path.with_suffix('.npz'))
        assert r['status']=='completed' and r['gpu_audit']['sampled_isolation_pass']
        assert r['field_sha256']==digest(path.with_suffix('.npz'))
        group=r['arguments']['group'];panel=dict(np.load(OUT/(group+'-panel.npz')))
        assert r['panel_sha256']==digest(OUT/(group+'-panel.npz'))
        assert fidelity(f['fit_mean'],f['gradient'],panel)[0]==r['fidelity']
        assert occupancy(f['mean'],f['variance'],panel)==r['occupancy']
        assert roots(f['reference_mean'],panel)[0]==r['geometry']
        assert all(h[-1]['relative_residual'] <= 1e-10 for h in r['refinement_history'])
        large.append(dict(name=name,configuration=r['configuration'],fidelity=r['fidelity'],geometry=r['geometry'],occupancy=r['occupancy']))
    chosen=selection['promotion_order'][0];ref=np.load(OUT/'validation-refinement-check.npz');screen=np.load(Path(chosen['source']).with_suffix('.npz'))
    agreement={key:float(np.max(abs(ref[key]-screen[key][:,chosen['column']]))) for key in ['mean','variance']}
    assert agreement['mean']<1e-9 and agreement['variance']<1e-10
    frozen=json.loads((OUT/'selected.json').read_text())
    assert frozen['validation_sha256']==digest(OUT/'validation-candidate1.json')
    assert frozen['protocol_sha256']==digest(OUT/'PROTOCOL.md')
    control=json.loads((OUT/'test-side-control-fit.json').read_text());cf=np.load(OUT/'test-side-control-fit.npz')
    panel=dict(np.load(OUT/'test_side-panel.npz'))
    assert control['field_sha256']==digest(OUT/'test-side-control-fit.npz')
    assert control['source_record_sha256']==digest(control['source_record'])
    assert control['panel_sha256']==digest(OUT/'test_side-panel.npz')
    assert fidelity(cf['fit_mean'],cf['gradient'],panel)[0]==control['fidelity']
    assert roots(cf['reference_mean'],panel)[0]==control['geometry']
    displays={}
    for tag in ['control','selected']:
        for resolution in [160,240]:
            path=OUT/f'display-{tag}-{resolution}.json';r=json.loads(path.read_text())
            assert r['field_sha256']==digest(path.with_suffix('.npz')) and r['gpu_audit']['sampled_isolation_pass']
            assert r['saved_mean_agreement']<1e-10
            displays[f'{tag}-{resolution}']=dict(grid=r['grid'],statistics=r['statistics'],boundary_edges=r['boundary_edges'])
    import igl
    convergence={}
    for tag in ['control','selected']:
        a=np.load(OUT/f'display-{tag}-160.npz');b=np.load(OUT/f'display-{tag}-240.npz')
        values={}
        for name,source,target in [('coarse_to_fine',a,b),('fine_to_coarse',b,a)]:
            d=np.sqrt(igl.point_mesh_squared_distance(source['vertices'],target['vertices'],target['faces'])[0])
            values[name]=dict(median=float(np.median(d)),p95=float(np.quantile(d,.95)),maximum=float(np.max(d)))
        convergence[tag]=values
    write(OUT/'summary.json',dict(screen_candidates=len(rows),screen_passing=len(passing),large=large,
                                 refinement_float64_agreement=agreement,displays=displays,display_convergence=convergence))
    print('Replay passed: 32 screening configurations, selection, large references, isolation, display records.',flush=True)


if __name__=='__main__':main()
