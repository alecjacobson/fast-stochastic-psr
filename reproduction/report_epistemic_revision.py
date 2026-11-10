"""Replay and plot fixed-model, post-hoc partial-scan uncertainty diagnostics."""
import argparse
import json

import numpy as np
from scipy.spatial import cKDTree

from reproduction.epistemic_revision import OUT, PREVIOUS, probabilities, summary, write
from reproduction.stanford_partial_data import CACHE, digest


def read(name):
    return json.loads((OUT/name).read_text())


def report():
    import igl
    spec=read('queries.json');record=read('reference.json');context=read('mean-context.json')
    assert record['status']=='completed' and record['gpu_audit']['sampled_isolation_pass']
    assert record['refinement_history'] and all(h[-1]['relative_residual']<=1e-10 for h in record['refinement_history'])
    assert record['retained_agreement']['mean_max']<=1e-9
    assert record['retained_agreement']['variance_over_prior_max']<=1e-7
    assert record['folded_mean_max_difference']<=1e-9
    for name,expected in [('queries.npz',spec['query_sha256']),('reference.npz',record['field_sha256']),
                          ('mean-context.npz',context['field_sha256']),('PROTOCOL.md',spec['protocol_sha256'])]:
        assert digest(OUT/name)==expected, name
    assert record['queries_sha256']==spec['query_sha256']
    assert record['selected_sha256']==spec['selected_sha256']==digest(PREVIOUS/'selected.json')
    assert spec['mesh_sha256']==digest(PREVIOUS/'display-selected-240.npz')
    assert spec['old_panel_sha256']==digest(PREVIOUS/'test_side-panel.npz')
    assert context['query_spec_sha256']==digest(OUT/'queries.json')
    assert context['coefficient_source_sha256']==digest(PREVIOUS/'test-side-selected.npz')
    assert context['reference_mesh_sha256']==digest(CACHE/'reference.npz')
    retained=read('retained-panel.json')
    assert retained['field_sha256']==digest(PREVIOUS/'test-side-selected.npz')
    assert retained['panel_sha256']==digest(PREVIOUS/'test_side-panel.npz')
    assert retained['old_field_sha256']==digest(PREVIOUS.parent/'quality_controls/test_side-selected-70652-reference.npz')
    for source,sha in record['source_sha256'].items():
        assert digest(OUT/'sources'/(sha+'.py'))==sha, source
    data=np.load(OUT/'queries.npz');fields=np.load(OUT/'reference.npz')
    mean,variance=fields['mean'],fields['variance'];q=data['queries']
    prior=record['configuration']['ell']**2/3
    p=probabilities(mean,variance)
    assert np.all(variance<=prior*(1+1e-8))
    x=np.load(CACHE/'test_side.npy')[:,:3];distance=cKDTree(x).query(q)[0]
    mesh=np.load(CACHE/'reference.npz')
    signed=igl.signed_distance(q,mesh['vertices'],mesh['faces'],igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)[0]
    np.savez_compressed(OUT/'diagnostics.npz',probability=p,variance_over_prior=variance/prior,
                        scan_distance=distance,reference_signed=signed)

    def subset(ids,labels=False):
        ids=np.asarray(ids)
        if not np.count_nonzero(ids) and ids.dtype==bool:return dict(count=0)
        result=summary(mean[ids],variance[ids],prior,signed[ids]<0 if labels else None)
        result['scan_distance_quantiles']=np.quantile(distance[ids],[0,.1,.5,.9,1]).tolist()
        return result

    rows={name:subset(np.arange(a,b)) for name,(a,b) in spec['partitions'].items()}
    plane=np.zeros(len(q),bool);a,b=spec['partitions']['plane'];plane[a:b]=True
    masks=dict(plane_far_from_scan=plane&(distance>.1),
               plane_scored_margin_005=plane&(abs(signed)>.005),
               plane_mean_inside_reference_outside=plane&(mean<0)&(signed>.005))
    rows.update({name:subset(mask,labels=True) for name,mask in masks.items()})
    for name in ['rim','observed']:
        a,b=spec['partitions'][name+'_profiles'];offsets=data[name+'_offsets']
        indices=np.arange(a,b).reshape(16,len(offsets))
        rows[name+'_by_offset']={str(offset):subset(indices[:,j]) for j,offset in enumerate(offsets)}
    write(OUT/'summary.json',dict(rows=rows,prior_variance=prior,reference_sha256=digest(OUT/'reference.json'),
        diagnostics_sha256=digest(OUT/'diagnostics.npz'),reference_mesh_sha256=digest(CACHE/'reference.npz'),
        source_sha256=digest(__file__),
        scope='Post-hoc descriptive queries, not new held-out selection. Reference labels are same-acquisition context; margin .005 is a sensitivity exclusion, not a registration certificate.'))
    print(json.dumps(rows,indent=2))


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    data=np.load(OUT/'queries.npz');diag=np.load(OUT/'diagnostics.npz')
    context=np.load(OUT/'mean-context.npz');spec=read('queries.json')
    ax,az=data['axis_x'],data['axis_z'];xx,zz=np.meshgrid(ax,az)

    def save(fig,name):
        fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight',facecolor='white')
        fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',facecolor='white');plt.close(fig)

    def plane(axis,field,title,cmap):
        value=field[:xx.size].reshape(xx.shape)
        im=axis.pcolormesh(xx,zz,value,shading='nearest',cmap=cmap,vmin=0,vmax=1,rasterized=True)
        axis.scatter(xx,zz,s=1,color='black',alpha=.16,linewidths=0)
        axis.contour(context['axis_x'],context['axis_z'],context['mean'],levels=[0],colors=['#111111'],linewidths=1.5)
        signed=context['reference_signed']
        if signed.min()<0<signed.max():
            axis.contour(context['axis_x'],context['axis_z'],signed,levels=[0],colors=['#d832a0'],linestyles='--',linewidths=1.6)
        points=context['near_plane_input'];axis.scatter(points[:,0],points[:,2],s=3,c='#19b9e4',linewidths=0)
        axis.set(xlim=(ax[0],ax[-1]),ylim=(az[0],az[-1]),aspect='equal',xlabel='x',ylabel='z',title=title)
        return im

    fig=plt.figure(figsize=(13,10.5));gs=fig.add_gridspec(2,2,height_ratios=[1,1.05],hspace=.28,wspace=.20)
    renders={kind:plt.imread(OUT/'render'/(kind+'-back-planes.png')) for kind in ['input','mean']}
    # A common viewport removes transparent studio margins without changing
    # either camera, relative scale, geometry, or scientific image pixels.
    visible=np.logical_or.reduce([im[:,:,3]>.01 for im in renders.values()])
    iy,ix=np.nonzero(visible);pad=50
    for col,(kind,title) in enumerate([('input','70,652 measured points'),('mean','Posterior mean zero surface / P(inside) = 0.5')]):
        axis=fig.add_subplot(gs[0,col]);axis.imshow(renders[kind])
        axis.set(xlim=(ix.min()-pad,ix.max()+pad),ylim=(iy.max()+pad,iy.min()-pad))
        axis.axis('off');axis.set_title(title,pad=5)
    for col,(key,title,cmap,label) in enumerate([
        ('probability','Marginal inside probability','RdBu_r','P(inside)'),
        ('variance_over_prior','Retained latent uncertainty','viridis','Posterior variance / prior variance')]):
        axis=fig.add_subplot(gs[1,col]);im=plane(axis,diag[key],title,cmap)
        fig.colorbar(im,ax=axis,fraction=.045,pad=.035,label=label,ticks=[0,.25,.5,.75,1])
    handles=[Line2D([],[],color='#111111',lw=1.5,label='Fine mean zero contour'),
             Line2D([],[],color='#d832a0',ls='--',label='Same-acquisition reference'),
             Line2D([],[],color='#19b9e4',marker='.',ls='',label='Input within 0.01 of plane')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,frameon=False)
    fig.suptitle('The unseen mean protrusion needs an uncertainty view',fontsize=16,y=.99)
    fig.text(.5,.955,f'Frozen model; cross-section y = {float(data["plane_y"]):.4f}; 33 × 33 full-kernel marginal evaluations',ha='center',color='.3')
    save(fig,'uncertainty-context')

    fig,axes=plt.subplots(2,2,figsize=(11.5,7.5),sharex='col',sharey='row',layout='constrained')
    for col,(name,color,title) in enumerate([('observed','#14658c','16 observed-surface anchors'),('rim','#a04916','16 unseen-rim anchors')]):
        a,b=spec['partitions'][name+'_profiles'];offset=data[name+'_offsets']
        for row,(key,ylabel) in enumerate([('probability','P(inside)'),('variance_over_prior','Variance / prior')]):
            axis=axes[row,col];values=diag[key][a:b].reshape(16,len(offset))
            axis.plot(offset,values.T,color=color,alpha=.20,lw=.8,marker='.',ms=3)
            axis.plot(offset,np.median(values,axis=0),color=color,lw=2.2,marker='o',ms=4,label='Median over anchors')
            axis.axvline(0,color='.5',lw=.7,ls=':');axis.set(ylim=(-.03,1.03),ylabel=ylabel)
            if row==0:axis.set_title(title);axis.axhline(.5,color='.65',lw=.7,ls='--')
            else:axis.set_xlabel('Input-normal offset' if name=='observed' else 'In-plane radial offset');axis.legend(frameon=False)
    fig.suptitle('Confidence away from the zero contour: individual profiles and their median',fontsize=14)
    save(fig,'uncertainty-profiles')

    panel=read('retained-panel.json');fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for axis,group,title in zip(axes,['all','unobserved_surface_region'],['All 512 retained queries','159 queries in unobserved reference regions']):
        axis.plot([0,1],[0,1],color='.6',ls='--',lw=1)
        for model,color,label in [('previous','#777777','Previous smooth model'),('current','#14658c','Observed-fidelity model')]:
            result=panel[model][group];bins=[v for v in result['reliability_bins'] if v['count']]
            x=[v['mean_probability'] for v in bins];y=[v['fraction_inside'] for v in bins]
            axis.plot(x,y,marker='o',color=color,label=f'{label}; Brier {result["brier"]:.4f}')
            for vx,vy,b in zip(x,y,bins):axis.annotate(str(b['count']),(vx,vy),xytext=(0,6 if model=='current' else -13),textcoords='offset points',ha='center',fontsize=8,color=color)
        axis.set(xlim=(-.03,1.03),ylim=(-.09,1.12),xlabel='Mean predicted inside probability',ylabel='Fraction inside reference',title=title)
        axis.legend(loc='lower right',fontsize=8,frameon=False)
    fig.suptitle('One-object reliability diagnostics — labels give bin counts; no iid confidence intervals',fontsize=12)
    save(fig,'retained-reliability')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['report','plots'])
    {'report':report,'plots':plots}[parser.parse_args().action]()
