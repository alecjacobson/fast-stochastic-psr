"""Check visual provenance, numerical agreement and unchanged benchmark evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.special import ndtr

from geospsr.nonperiodic import FreeSpacePoissonMatern
from reproduction.partial_competitor_common import gates
from reproduction.partial_occupancy import ROOT

OUT=Path('reproduction/results/partial_renderings')
BASE=Path('reproduction/results/mixed_h2')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    scene=dict(np.load(OUT/'scene.npz'));h=dict(np.load(OUT/'h2-fields.npz'))
    contract=json.loads((OUT/'contract.json').read_text());record=json.loads((OUT/'h2-fields.json').read_text())
    check=json.loads((OUT/'reference-check.json').read_text());ref=dict(np.load(OUT/'reference-check.npz'))
    assert sha(OUT/'scene.npz')==contract['scene_sha256']
    assert sha(OUT/'contract.json')==record['contract_sha256']
    assert sha(OUT/'h2-fields.npz')==record['field_sha256']
    assert sha(OUT/'reference-check.npz')==check['field_sha256']
    assert scene['points'].shape==(70652,3) and scene['contacts'].shape==(32,3)
    assert len(h['mean'])==128+2*72**2
    np.testing.assert_array_equal(h['queries'][:128],scene['benchmark_queries'])
    np.testing.assert_array_equal(h['queries'][128:],scene['slice_queries'].reshape(-1,3))
    assert np.isfinite(h['mean']).all() and np.isfinite(h['variance']).all() and np.all(h['variance']>0)
    fit=json.loads((ROOT/'controls.json').read_text())['fits']['spatial'];model=FreeSpacePoissonMatern(.48,1/3)
    ids=ref['indices'];np.testing.assert_array_equal(h['queries'][ids],ref['queries'])
    error=gates(h['mean'][ids],h['variance'][ids],ref,model,fit)
    assert error['passes'] and check['errors']==error
    for r in [record,check]:
        audit=r['gpu_audit'];assert audit['sampled_isolation_pass'] and not audit['errors'] and not audit['foreign_process_samples']
        assert sha(OUT/'sources'/(r['source_sha256']+'.py'))==r['source_sha256']
    for history in check['histories']:
        last=history[-1];assert last['relative_residual']<=1e-9
        assert last['change']['cdf']<=1e-6 and last['change']['calibrated_cdf']<=1e-6
    assert record['folded_mean_max_error']<1e-10
    volume=np.load(OUT/'mean-volume.npz');mesh=np.load(OUT/'mean-mesh.npz');mr=json.loads((OUT/'mean-mesh.json').read_text())
    assert sha(OUT/'mean-volume.npz')==record['mean_volume_sha256']==mr['field_sha256']
    assert sha(OUT/'mean-mesh.npz')==mr['mesh_sha256']
    axes=[volume['axis_'+a] for a in 'xyz']
    coords=np.array([(mesh['vertices'][:,i]-a[0])/(a[1]-a[0]) for i,a in enumerate(axes)])
    iso_error=float(np.max(abs(map_coordinates(volume['mean'],coords,order=1,mode='nearest'))))
    assert iso_error<1e-6,iso_error
    rgba=np.load(OUT/'probe-colors.npz')['h2']
    import matplotlib.pyplot as plt
    np.testing.assert_array_equal(rgba,plt.get_cmap('viridis')(ndtr(-h['mean'][:128]/np.sqrt(h['variance'][:128]))))
    render_names=[kind+'-'+view for kind in ['reference','input','mean'] for view in ['observed','back']]
    render_names+=['input-observed-planes','probes-h2-observed']
    for name in render_names:
        r=json.loads((OUT/(name+'.json')).read_text())
        assert r['source_sha256']==sha(OUT/'sources'/(r['source_sha256']+'.py'))
        assert r['scene_sha256']==sha(OUT/'scene.npz') and not r['preview']
        assert r['geometry_sha256']==sha(OUT/('mean-mesh.npz' if r['kind']=='mean' else 'scene.npz'))
        assert (OUT/(name+'.png')).stat().st_size>10000
        for a in r['plane_label_anchors']:assert 0<a[0]<1 and 0<a[1]<1
        points=(mesh['vertices'] if r['kind']=='mean' else scene['points'] if r['kind']=='input'
                else scene['benchmark_queries'] if r['kind'].startswith('probes') else scene['vertices'])
        v=points[:,[0,2,1]]*np.array([1,-1,1])-r['target']
        direction=np.array(r['target'])-r['camera'];direction/=np.linalg.norm(direction)
        right=np.cross(direction,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,direction)
        im=plt.imread(OUT/(name+'.png'));aspect=im.shape[1]/im.shape[0]
        radius=.018 if r['kind'].startswith('probes') else .0033 if r['kind']=='input' else 0
        assert np.max(abs(v@right))+radius<r['orthographic_scale']*aspect/2,name
        assert np.max(abs(v@up))+radius<r['orthographic_scale']/2,name
    # All historical evidence remains byte-for-byte identical.
    historical=0
    for folder in ['mixed_h2','partial_competitors','real_partial']:
        root=OUT.parent/folder;m=json.loads((root/'manifest.json').read_text())
        for name,digest in m['artifacts'].items():assert sha(root/name)==digest,(folder,name)
        historical+=len(m['artifacts'])
    sources=['reproduction/compute_partial_figures.py','reproduction/render_partial_figures.py',
             'reproduction/compose_partial_figures.py','reproduction/audit_partial_figures.py','tests/test_partial_figures.py']
    for name in sources:
        data=Path(name).read_bytes();(OUT/'sources'/(sha(name)+'.py')).write_bytes(data)
    summary=dict(base_commit='75b087f73514d135d7d297f1e6a0baa15effef6a',checked_queries=len(ids),
        errors=error,mean_isosurface_max_trilinear_residual=iso_error,unchanged_historical_artifacts=historical,
        full_kernel_check_scope='Fixed 128 benchmark queries + 64 on each slice, not a uniform field certificate',
        test='Contact-folded mean agrees with independent augmented Gaussian posterior; one test passed',
        sources={name:sha(name) for name in sources},
        artifacts={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()
                   and p.name!='manifest.json' and '-preview.' not in p.name})
    (OUT/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Audited',len(ids),'full-kernel checks,',len(render_names),'renders,',historical,'unchanged historical artifacts')
    print('Maximum display-mesh zero-level residual:',iso_error)


if __name__=='__main__':main()
