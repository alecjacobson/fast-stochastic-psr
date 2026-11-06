"""Plots and pooled diagnostics for the real partial-scan occupancy pilot."""
import json
from pathlib import Path

import igl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr

from geospsr.nonperiodic import FreeSpacePoissonMatern, DenseFreeSpacePosterior
from geospsr.scalar_contacts import ContactConditionedDensePosterior
from reproduction.partial_occupancy import ROOT, load
from reproduction.stanford_partial_data import CACHE, digest

GROUPS = ['test_side', 'test_back', 'test_other_side']
LABELS = ['Side views', 'Back views', 'Other side views']


def pooled(records, key, stratum='all'):
    count = sum(r[key][stratum]['count'] for r in records)
    return {metric:sum(r[key][stratum][metric]*r[key][stratum]['count'] for r in records)/count
            for metric in ['brier', 'log_loss']}


def plot_quality():
    controls = json.loads((ROOT/'controls.json').read_text())
    exact = [json.loads(p.read_text()) for p in sorted((ROOT/'exact').glob('*2048.json'))]
    cc = [r for r in controls['tests'] if r['n'] == 2048]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, stratum, title in zip(axes, ['all', 'observed', 'occluded'], ['All queries', 'Observed region', 'Occluded region']):
        methods = [('Posterior', exact, 'posterior'), ('Constant variance', exact, 'constant_same_mean'),
                   ('Contact-only variance', cc, 'contact_only_variance'),
                   ('Calibrated posterior', cc, 'calibrated_spatial'),
                   ('Calibrated mean only', cc, 'calibrated_mean_only'),
                   ('Calibrated contact-only', cc, 'calibrated_contact_only')]
        values = [pooled(records, key, stratum)['brier'] for _, records, key in methods]
        ax.barh(np.arange(len(values)), values, color=['#0072B2','#888888','#E69F00','#009E73','#aaaaaa','#CC79A7'])
        ax.set_yticks(np.arange(len(values)), [m[0] for m in methods]); ax.invert_yaxis()
        ax.set_title(title); ax.set_xlabel('Brier score (lower is better)'); ax.grid(axis='x', alpha=.2)
    fig.suptitle('Real Armadillo partial scans: pooled held-out views, N = 2,048\nCalibration fits use a different view group; reconstructed-mesh reference labels', fontsize=12)
    fig.tight_layout(); fig.savefig(ROOT/'quality.png', dpi=170); fig.savefig(ROOT/'quality.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, group, title in zip(axes, GROUPS, LABELS):
        raw = next(r for r in exact if r['group'] == group)
        calibrated = next(r for r in cc if r['group'] == group)
        for label, scores, color in [('Posterior',raw['posterior']['all'],'#0072B2'),
                ('Calibrated posterior',calibrated['calibrated_spatial']['all'],'#009E73'),
                ('Calibrated contact-only',calibrated['calibrated_contact_only']['all'],'#CC79A7')]:
            b = [v for v in scores['calibration'] if v['count'] >= 20]
            ax.plot([v['probability'] for v in b], [v['frequency'] for v in b], 'o-', label=label, color=color, ms=4)
        ax.plot([0,1],[0,1],'k--',lw=.8); ax.set(xlim=(0,1),ylim=(0,1),title=title,
            xlabel='Predicted inside probability',ylabel='Reference inside fraction'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); fig.suptitle('Reliability diagnostics on spatial queries (not independent repeated scans)',fontsize=11)
    fig.tight_layout(); fig.savefig(ROOT/'reliability.png',dpi=170); fig.savefig(ROOT/'reliability.pdf'); plt.close(fig)
    return dict(raw={k:pooled(exact,k) for k in ['posterior','prior_variance','constant_same_mean','independently_tuned_constant']},
                controls={k:pooled(cc,k) for k in ['contact_only_variance','calibrated_spatial','calibrated_mean_only','calibrated_contact_only']})


def plot_slices():
    mesh = np.load(CACHE/'reference.npz'); vertices, faces = mesh['vertices'], mesh['faces']
    selected = json.loads((ROOT/'selected.json').read_text())
    fit = json.loads((ROOT/'controls.json').read_text())['fits']['spatial']
    model = FreeSpacePoissonMatern(selected['ell'],1/3)
    xs = np.linspace(vertices[:,0].min()-.15,vertices[:,0].max()+.15,96)
    zs = np.linspace(vertices[:,2].min()-.15,vertices[:,2].max()+.15,96)
    xx, zz = np.meshgrid(xs,zs); q = np.column_stack((xx.ravel(),np.full(xx.size,.3),zz.ravel()))
    signed, *_ = igl.signed_distance(q, vertices, faces, igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 10.5))
    for row,(group,title) in enumerate(zip(GROUPS,LABELS)):
        path = ROOT/(group+'-slice.npz')
        x, normals, contacts, _ = load(group,2048)
        fingerprint = digest(ROOT/'selected.json')+digest(ROOT/'dataset.json')+digest('geospsr/scalar_contacts.py')
        record = np.load(path) if path.exists() else None
        if (record is not None and 'fingerprint' in record and str(record['fingerprint']) == fingerprint
                and np.array_equal(record['queries'], q)):
            mean,variance=record['mean'],record['variance']
        else:
            base=DenseFreeSpacePosterior(model,x,normals,selected['sigma']**2)
            mean,variance=ContactConditionedDensePosterior(base,contacts,np.zeros(len(contacts)),.01**2).predict(q)
            np.savez_compressed(path,queries=q,mean=mean,variance=variance,reference_distance=signed,fingerprint=fingerprint)
        z=-mean/np.sqrt(variance); raw=ndtr(z); calibrated=ndtr(np.exp(fit['log_scale'])*z+fit['bias'])
        for col,(value,label,cmap) in enumerate([(raw,'Raw inside probability','RdBu_r'),
                                               (calibrated,'Calibrated probability','RdBu_r'),
                                               (np.sqrt(variance),'Scalar standard deviation','viridis')]):
            ax=axes[row,col]
            image=ax.pcolormesh(xs,zs,value.reshape(xx.shape),shading='auto',cmap=cmap,
                               vmin=0,vmax=1 if col<2 else .2)
            ax.contour(xs,zs,signed.reshape(xx.shape),levels=[0],colors='black',linewidths=.8)
            near=abs(x[:,1]-.3)<.025
            ax.scatter(x[near,0],x[near,2],s=3,color='black',alpha=.6)
            ax.set_aspect('equal');ax.set_title(title+': '+label,fontsize=10);ax.set_xlabel('x');ax.set_ylabel('z')
            fig.colorbar(image,ax=ax,shrink=.7)
    fig.suptitle('Real partial scans: chest cross-section, y = 0.3\nBlack curve: reconstructed reference boundary; dots: nearby raw scan points',fontsize=12)
    fig.tight_layout();fig.savefig(ROOT/'slices.png',dpi=170);fig.savefig(ROOT/'slices.pdf');plt.close(fig)
    fig=plt.figure(figsize=(12,4))
    for i,(group,title) in enumerate(zip(GROUPS,LABELS)):
        ax=fig.add_subplot(1,3,i+1,projection='3d'); cloud=np.load(CACHE/(group+'.npy'))
        ax.scatter(*vertices[::100][:,[0,2,1]].T,c='lightgray',s=.2,alpha=.3)
        ax.scatter(*cloud[::30][:,[0,2,1]].T,c=cloud[::30,2],cmap='viridis',s=.6)
        ax.set_box_aspect(np.ptp(vertices[:,[0,2,1]],axis=0));ax.view_init(elev=12,azim=-65)
        ax.set_title(title);ax.set_axis_off()
    fig.suptitle('Three held-out groups of actual laser range scans; complete reference in gray')
    fig.tight_layout();fig.savefig(ROOT/'scans.png',dpi=190);fig.savefig(ROOT/'scans.pdf');plt.close(fig)


def main():
    summary=plot_quality();plot_slices()
    direct=[]
    for path in sorted((ROOT/'direct').glob('*.json')):
        if path.name.endswith('.status.json'):continue
        r=json.loads(path.read_text());direct.append(r)
    # A dense finite-panel field is an independent full-kernel reference for
    # larger runs. Compare every query, not merely a handful of probes.
    comparisons=[]
    for r in direct:
        n=r['arguments']['n'];method=r['arguments']['method']
        if method == 'dense':
            # Dense is the larger-N reference. Do not count comparison with
            # itself as an independent accuracy check.
            continue
        refpath=ROOT/'direct'/f'dense-{n}.npz'
        if not refpath.exists():continue
        reference=np.load(refpath);candidate=np.load(Path(r['arguments']['output']).with_suffix('.npz'))
        v,c=reference['variance'],candidate['variance'];m,mm=reference['mean'],candidate['mean']
        prior=json.loads((ROOT/'selected.json').read_text())['ell']**2/3
        errors=dict(variance_over_prior=float(np.max(abs(c-v))/prior),cdf=float(np.max(abs(ndtr(-mm/np.sqrt(c))-ndtr(-m/np.sqrt(v))))))
        comparisons.append(dict(n=n,method=method,**errors,passes=errors['variance_over_prior']<=1e-4 and errors['cdf']<=1e-3,
            total_seconds=r['total_seconds'],shortcut_total_seconds=r['shortcut_total_seconds']))
    summary['direct_comparisons']=comparisons
    summary['direct_timings']=[dict(n=r['arguments']['n'],method=r['arguments']['method'],
        total_seconds=r['total_seconds'],shortcut_total_seconds=r['shortcut_total_seconds'],
        queries=r['queries'],cpu_oracle_pass=r.get('passes_exact_panel'),
        sampled_gpu_isolation=r['production_gpu_audit']['sampled_isolation_pass']) for r in direct]
    summary['artifact_sha256']={str(p.relative_to(ROOT)):digest(p) for p in ROOT.rglob('*')
                               if p.is_file() and p.name not in ['summary.json','README.md','manifest.json']}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='artifact_sha256'},indent=2),flush=True)


if __name__=='__main__':main()
