"""Fixed-model uncertainty diagnostics for the PR127 partial scan."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import ndtr, log_ndtr

from geospsr.nonperiodic import FreeSpacePoissonMatern
from reproduction.stanford_partial_data import CACHE, digest
from reproduction.partial_occupancy import load, farthest_contacts

OUT=Path('reproduction/results/epistemic_revision')
PREVIOUS=Path('reproduction/results/observed_fidelity')


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def snapshot():
    folder=OUT/'sources';folder.mkdir(exist_ok=True);result={}
    for p in [__file__,'reproduction/observed_fidelity_large.py','reproduction/partial_quality_large.py',
              'reproduction/partial_direct_gpu.py','reproduction/fused_mean.py','reproduction/gpu_audit.py',
              'geospsr/nonperiodic.py','geospsr/scalar_contacts.py','geospsr/nonperiodic_dense_gpu.py']:
        h=digest(p);(folder/(h+'.py')).write_bytes(Path(p).read_bytes());result[p]=h
    return result


def probabilities(mean,variance):
    mean,variance=np.asarray(mean),np.asarray(variance)
    if mean.shape!=variance.shape or not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance<=0):
        raise ValueError('finite means and strictly positive finite variances required')
    return ndtr(-mean/np.sqrt(variance))


def summary(mean,variance,prior,inside=None):
    p=probabilities(mean,variance)
    result=dict(count=len(p),variance_over_prior_quantiles=np.quantile(variance/prior,[0,.1,.5,.9,1]).tolist(),
                probability_quantiles=np.quantile(p,[0,.1,.5,.9,1]).tolist(),
                fraction_probability_02_08=float(np.mean((p>=.2)&(p<=.8))),
                fraction_probability_045_055=float(np.mean((p>=.45)&(p<=.55))),
                fraction_confident_95=float(np.mean((p<.05)|(p>.95))))
    if inside is not None:
        inside=np.asarray(inside,dtype=bool);truthp=np.where(inside,p,1-p);wrong=(p>=.5)!=inside
        z=-mean/np.sqrt(variance)
        result.update(brier=float(np.mean((p-inside)**2)),log_score=float(-np.where(inside,log_ndtr(z),log_ndtr(-z)).mean()),
                      wrong_count=int(wrong.sum()),confident_wrong_count=int(np.sum(truthp<.05)),
                      truth_probability_on_errors=np.quantile(truthp[wrong],[0,.5,1]).tolist() if wrong.any() else None)
        bins=[]
        for i in range(10):
            mask=(p>=i/10)&((p<(i+1)/10) if i<9 else (p<=1))
            bins.append(dict(count=int(mask.sum()),mean_probability=float(p[mask].mean()) if mask.any() else None,
                             fraction_inside=float(inside[mask].mean()) if mask.any() else None))
        result['reliability_bins']=bins
    return result


def prepare():
    path=OUT/'queries.npz'
    if path.exists():raise FileExistsError(path)
    mesh=np.load(PREVIOUS/'display-selected-240.npz');v,f=mesh['vertices'],mesh['faces']
    edges=np.sort(np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1)
    edge,count=np.unique(edges,axis=0,return_counts=True);boundary=v[np.unique(edge[count==1])]
    assert len(boundary)==192 and np.ptp(boundary[:,0])==0
    center=boundary.mean(0)
    axis_x=np.linspace(center[0],-.30,33);axis_z=np.linspace(-.72,-.12,33)
    xx,zz=np.meshgrid(axis_x,axis_z,indexing='xy')
    grid=np.column_stack((xx.ravel(),np.full(xx.size,center[1]),zz.ravel()))
    # Angular ordering makes the 16 rim anchors spatially interpretable.
    order=np.argsort(np.arctan2(boundary[:,2]-center[2],boundary[:,1]-center[1]))
    rim=boundary[order[np.linspace(0,len(boundary)-1,16,dtype=int)]]
    radial=rim-center;radial[:,0]=0;radial/=np.linalg.norm(radial,axis=1)[:,None]
    rim_offsets=np.array([-.10,-.03,0,.03,.10]);rim_profiles=rim[:,None,:]+radial[:,None,:]*rim_offsets[None,:,None]
    panel=dict(np.load(PREVIOUS/'test_side-panel.npz'))
    anchors=farthest_contacts(panel['fit_points'],16)
    _,indices=cKDTree(panel['fit_points']).query(anchors);normals=panel['fit_normals'][indices]
    observed_offsets=np.array([-.10,-.03,-.01,-.003,0,.003,.01,.03,.10])
    observed_profiles=anchors[:,None,:]+normals[:,None,:]*observed_offsets[None,:,None]
    check_indices=np.linspace(0,len(panel['queries'])-1,32,dtype=int)
    arrays=[grid,rim_profiles.reshape(-1,3),observed_profiles.reshape(-1,3),panel['queries'][check_indices]]
    stops=np.r_[0,np.cumsum([len(a) for a in arrays])]
    np.savez_compressed(path,queries=np.concatenate(arrays),stops=stops,axis_x=axis_x,axis_z=axis_z,
                        plane_y=center[1],boundary=boundary,center=center,rim_anchors=rim,rim_offsets=rim_offsets,
                        observed_anchors=anchors,observed_offsets=observed_offsets,observed_normals=normals,check_indices=check_indices)
    write(OUT/'queries.json',dict(protocol_sha256=digest(OUT/'PROTOCOL.md'),query_sha256=digest(path),
        selected_sha256=digest(PREVIOUS/'selected.json'),mesh_sha256=digest(PREVIOUS/'display-selected-240.npz'),
        old_panel_sha256=digest(PREVIOUS/'test_side-panel.npz'),source_sha256=digest(__file__),
        partitions={name:[int(a),int(b)] for name,a,b in zip(['plane','rim_profiles','observed_profiles','retained_check'],stops[:-1],stops[1:])},
        scope='Post-hoc diagnostic of the existing mean protrusion; locations fixed before new variance, no model retuning.'))
    print('PREPARED',stops.tolist(),center.tolist(),flush=True)


def replay_panel():
    panel=dict(np.load(PREVIOUS/'test_side-panel.npz'));fields=np.load(PREVIOUS/'test-side-selected.npz')
    cfg=json.loads((PREVIOUS/'selected.json').read_text())['configuration'];prior=cfg['ell']**2/3
    x=np.load(CACHE/'test_side.npy')[:,:3];distance=cKDTree(x).query(panel['queries'])[0]
    masks=dict(all=np.ones(len(distance),bool),observed_surface_region=panel['observed'],unobserved_surface_region=panel['occluded'],
               within_002_of_scan=distance<.02,beyond_010_from_scan=distance>.10)
    rows={name:summary(fields['mean'][mask],fields['variance'][mask],prior,panel['inside'][mask]) for name,mask in masks.items()}
    old=np.load(PREVIOUS.parent/'quality_controls/test_side-selected-70652-reference.npz')
    old_rows={name:summary(old['mean'][mask],old['variance'][mask],.48**2/3,panel['inside'][mask]) for name,mask in masks.items()}
    write(OUT/'retained-panel.json',dict(current=rows,previous=old_rows,
        field_sha256=digest(PREVIOUS/'test-side-selected.npz'),panel_sha256=digest(PREVIOUS/'test_side-panel.npz'),
        old_field_sha256=digest(PREVIOUS.parent/'quality_controls/test_side-selected-70652-reference.npz'),source_sha256=digest(__file__),
        scope='512 retained spatial queries from one object; reliability bins have counts but no iid confidence intervals or population-calibration claim.'))
    print('PANEL',rows['unobserved_surface_region'],flush=True)


def reference():
    import cupy as cp
    from reproduction.observed_fidelity_large import StreamingReference
    from reproduction.fused_mean import FusedMean
    from reproduction.gpu_audit import GpuProcessAudit
    path=OUT/'reference.json'
    if path.exists():raise FileExistsError(path)
    spec=json.loads((OUT/'queries.json').read_text());q=np.load(OUT/'queries.npz')
    assert spec['query_sha256']==digest(OUT/'queries.npz')
    frozen=json.loads((PREVIOUS/'selected.json').read_text());cfg=frozen['configuration']
    assert spec['selected_sha256']==digest(PREVIOUS/'selected.json')
    x,y,_,_=load('test_side',70652);c=np.load(PREVIOUS/'test_side-panel.npz')['contacts'][:cfg['contacts']]
    model=FreeSpacePoissonMatern(cfg['ell'],1/3)
    record=dict(configuration=cfg,queries_sha256=digest(OUT/'queries.npz'),selected_sha256=digest(PREVIOUS/'selected.json'),
                source_sha256=snapshot(),scope='Full-kernel reference uncertainty diagnostic; no H2 action or competitive timing claim.')
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id,interval=.5,cuda_device=cp.cuda.Device().id).start();start=perf_counter()
    try:
        posterior=StreamingReference(model,x,y,c,cfg['sigma']**2,cfg['scalar_variance'],block=192)
        mean,variance=posterior.predict(q['queries']);probabilities(mean,variance)
        record['refinement_history']=posterior.system.histories
        previous=np.load(PREVIOUS/'test-side-selected.npz');a,b=spec['partitions']['retained_check'];ids=q['check_indices']
        errors=dict(mean_max=float(np.max(abs(mean[a:b]-previous['mean'][ids]))),
                    variance_over_prior_max=float(np.max(abs(variance[a:b]-previous['variance'][ids]))/model.prior_variance))
        assert errors['mean_max']<=1e-9 and errors['variance_over_prior_max']<=1e-7
        evaluator=FusedMean(model,x,c,previous['coefficients'],previous['contact_weights'])
        record['folded_mean_max_difference']=float(np.max(abs(evaluator(q['queries'])-mean)))
        assert record['folded_mean_max_difference']<=1e-9
        np.savez_compressed(path.with_suffix('.npz'),mean=mean,variance=variance)
        record.update(status='completed',retained_agreement=errors,field_sha256=digest(path.with_suffix('.npz')))
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        record.update(reference_seconds=perf_counter()-start,gpu_audit=audit.stop());write(path,record)
    print('REFERENCE',record['reference_seconds'],errors,flush=True)


def mean_context():
    import igl
    from reproduction.fused_mean import FusedMean
    path=OUT/'mean-context.npz'
    if path.exists():raise FileExistsError(path)
    panel=np.load(OUT/'queries.npz');f=np.load(PREVIOUS/'test-side-selected.npz')
    cfg=json.loads((PREVIOUS/'selected.json').read_text())['configuration'];model=FreeSpacePoissonMatern(cfg['ell'],1/3)
    x=np.load(CACHE/'test_side.npy')[:,:3];c=np.load(PREVIOUS/'test_side-panel.npz')['contacts'][:cfg['contacts']]
    ax=np.linspace(panel['axis_x'][0],panel['axis_x'][-1],257);az=np.linspace(panel['axis_z'][0],panel['axis_z'][-1],257)
    xx,zz=np.meshgrid(ax,az,indexing='xy');q=np.column_stack((xx.ravel(),np.full(xx.size,float(panel['plane_y'])),zz.ravel()))
    mean=FusedMean(model,x,c,f['coefficients'],f['contact_weights'])(q).reshape(xx.shape)
    ref=np.load(CACHE/'reference.npz');signed=igl.signed_distance(q,ref['vertices'],ref['faces'],igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)[0].reshape(xx.shape)
    near=x[abs(x[:,1]-float(panel['plane_y']))<.01]
    np.savez_compressed(path,axis_x=ax,axis_z=az,mean=mean,reference_signed=signed,near_plane_input=near)
    write(OUT/'mean-context.json',dict(field_sha256=digest(path),query_spec_sha256=digest(OUT/'queries.json'),
        coefficient_source_sha256=digest(PREVIOUS/'test-side-selected.npz'),reference_mesh_sha256=digest(CACHE/'reference.npz'),
        source_sha256={p:digest(p) for p in [__file__,'reproduction/fused_mean.py']},
        scope='Fine direct mean/reference contours only; new variances are evaluated on the explicitly plotted33-grid nodes.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','panel','reference','mean-context']);a=p.parse_args()
    dict(prepare=prepare,panel=replay_panel,reference=reference,**{'mean-context':mean_context})[a.action]()
