"""Compose paper-style figures from computed fields and actual Cycles renders."""
import argparse
import json
from pathlib import Path

import igl
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import ndtr

from reproduction.stanford_partial_data import digest

OUT=Path('reproduction/results/partial_renderings')
TIMINGS=Path('reproduction/results/mixed_h2')
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11,'axes.titlesize':13,
                     'mathtext.fontset':'dejavuserif','pdf.fonttype':42,'ps.fonttype':42})


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=220,facecolor='white')
    fig.savefig(OUT/(name+'.pdf'),facecolor='white',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)


def prepare():
    scene=np.load(OUT/'scene.npz');h=np.load(OUT/'h2-fields.npz')
    dense=np.load(TIMINGS/'reference-70652.npz')
    np.savez_compressed(OUT/'probe-colors.npz',h2=plt.get_cmap('viridis')(ndtr(-h['mean'][:128]/np.sqrt(h['variance'][:128]))),
        dense=plt.get_cmap('viridis')(ndtr(-dense['mean']/np.sqrt(dense['variance']))))
    q=scene['slice_queries'].reshape(-1,3)
    signed,_,closest,_=igl.signed_distance(q,scene['vertices'],scene['faces'],igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)
    distance=cKDTree(scene['points']).query(closest)[0]
    np.savez_compressed(OUT/'slice-reference.npz',distance=signed,distance_to_scan=distance)
    p=ndtr(-h['mean'][128:]/np.sqrt(h['variance'][128:]))
    rows=[]
    for i,height in enumerate(scene['heights']):
        selection=slice(i*72**2,(i+1)*72**2)
        v=h['variance'][128:][selection];d=distance[selection];near=abs(signed[selection])<.06
        valid=abs(signed[selection])>.005;inside=signed[selection]<0;pp=p[selection]
        rows.append(dict(height=float(height),variance_quantiles=np.quantile(v,[0,.1,.5,.9,1]).tolist(),
            probability_quantiles=np.quantile(p[selection],[0,.1,.5,.9,1]).tolist(),
            reference_boundary_margin=.005,scored_points=int(valid.sum()),reference_inside_points=int((valid&inside).sum()),
            inside_probability_below_05=int((valid&inside&(pp<.05)).sum()),
            outside_probability_above_95=int((valid&~inside&(pp>.95)).sum()),
            raw_brier=float(np.mean((pp[valid]-inside[valid])**2)),
            observed_near_boundary_count=int((near&(d<.025)).sum()),
            occluded_near_boundary_count=int((near&(d>.075)).sum()),
            observed_near_boundary_median_variance=float(np.median(v[near&(d<.025)])),
            occluded_near_boundary_median_variance=float(np.median(v[near&(d>.075)]))))
    (OUT/'slice-statistics.json').write_text(json.dumps(rows,indent=2)+'\n')
    if (OUT/'mean-mesh.npz').exists():
        mesh=np.load(OUT/'mean-mesh.npz')
        a,*_=igl.signed_distance(mesh['vertices'],scene['vertices'],scene['faces'],igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)
        b,*_=igl.signed_distance(scene['vertices'],mesh['vertices'],mesh['faces'],igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)
        coverage=cKDTree(scene['points']).query(scene['vertices'])[0]
        values={'mean_vertices_to_reference':np.abs(a),'reference_vertices_to_mean':np.abs(b),
            'observed_reference_to_mean':np.abs(b[coverage<.025]),'unobserved_reference_to_mean':np.abs(b[coverage>.075])}
        record={key:dict(count=len(value),median=float(np.median(value)),p95=float(np.quantile(value,.95))) for key,value in values.items()}
        record['scope']='Vertex-to-triangle distances in normalized coordinates; same-acquisition reference and finite display grid, not independent physical truth.'
        (OUT/'geometry-statistics.json').write_text(json.dumps(record,indent=2)+'\n')


def image(ax,name):
    ax.imshow(plt.imread(OUT/(name+'.png')));ax.set_axis_off()


def geometry():
    fig,axes=plt.subplots(2,3,figsize=(11,8.5),gridspec_kw={'wspace':.01,'hspace':.01})
    for row,view in enumerate(['observed','back']):
        for col,kind in enumerate(['reference','input','mean']):
            image(axes[row,col],kind+'-'+view)
            if row==0:axes[row,col].set_title(['Complete reference mesh','Actual partial scan','H² posterior mean'][col],pad=1)
        axes[row,0].text(.03,.5,['View toward observed side','Opposite view'][row],rotation=90,
            va='center',ha='right',transform=axes[row,0].transAxes,fontsize=11)
    fig.subplots_adjust(left=.05,right=.99,bottom=.075,top=.93)
    fig.text(.5,.035,'70,652 oriented observations + 32 scalar contacts (red).  Reference is context; right: computed μ = 0 surface.',ha='center',fontsize=10)
    save(fig,'geometry')


def slices():
    scene=np.load(OUT/'scene.npz');h=np.load(OUT/'h2-fields.npz');ref=np.load(OUT/'slice-reference.npz')
    x,z=scene['axis_x'],scene['axis_z'];shape=(2,len(z),len(x))
    mean=h['mean'][128:].reshape(shape);variance=h['variance'][128:].reshape(shape)
    probability=ndtr(-mean/np.sqrt(variance));distance=ref['distance'].reshape(shape)
    fig=plt.figure(figsize=(15.5,7.6));grid=fig.add_gridspec(2,4,width_ratios=[1.20,1,1,1],left=.015,right=.985,bottom=.15,top=.90,wspace=.13,hspace=.18)
    cloud=fig.add_subplot(grid[:,0]);image(cloud,'input-observed-planes');cloud.set_title('Observed points + slice locations',fontsize=12,pad=8)
    anchors=json.loads((OUT/'input-observed-planes.json').read_text())['plane_label_anchors']
    for i,(anchor,color) in enumerate(zip(anchors,['#296494','#b35019'])):
        cloud.text(anchor[0]+.015,anchor[1],['A','B'][i],transform=cloud.transAxes,color=color,fontsize=16,fontweight='bold',va='center')
    minmean,maxmean=float(mean.min()),float(mean.max());varmax=float(variance.max())
    scales=[colors.TwoSlopeNorm(vcenter=0,vmin=minmean,vmax=maxmean),colors.LogNorm(float(variance.min()),varmax),colors.Normalize(0,1)]
    fields=[mean,variance,probability];cmaps=['RdBu','plasma','viridis']
    labels=[r'Mean $\mu$ (negative inside)',r'Variance $\sigma^2$ (log scale)',r'Inside probability $\Phi(-\mu/\sigma)$']
    axs=[];images=[]
    for row in range(2):
        rowaxes=[]
        for col in range(3):
            ax=fig.add_subplot(grid[row,col+1]);rowaxes.append(ax)
            im=ax.pcolormesh(x,z,fields[col][row],shading='nearest',cmap=cmaps[col],norm=scales[col],rasterized=True)
            ax.contour(x,z,distance[row],levels=[0],colors='#141414',linewidths=.8)
            ax.contour(x,z,mean[row],levels=[0],colors='white',linewidths=1,linestyles='--')
            nearby=abs(scene['points'][:,1]-scene['heights'][row])<.012
            points=scene['points'][nearby]
            ax.scatter(points[:,0],points[:,2],s=.65,c='#b7b7b7',alpha=.65,linewidths=0,rasterized=True)
            ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(labels[col],pad=12,fontsize=12);images.append(im)
            if col==0:ax.text(-.045,.5,('%s  ·  y = %.2f'%('AB'[row],scene['heights'][row])),rotation=90,va='center',ha='right',transform=ax.transAxes,fontsize=11,color=['#296494','#b35019'][row])
        axs.append(rowaxes)
    for col in range(3):
        pos=axs[1][col].get_position();cax=fig.add_axes([pos.x0,pos.y0-.042,pos.width,.018])
        cb=fig.colorbar(images[col],cax=cax,orientation='horizontal');cb.ax.tick_params(labelsize=9)
        if col==0:
            cb.set_ticks([minmean,0,maxmean]);cb.set_ticklabels(['%.3f'%minmean,'0','%.3f'%maxmean])
    fig.text(.5,.035,'Shared scales across both slices.  Black: reference boundary; white dashed: posterior mean zero set; gray dots: nearby measurements.',ha='center',fontsize=10)
    fig.text(.5,.008,'Additional spatial evaluations for illustration; the 17.82 s benchmark covers a separate batch of 128 marginal queries.',ha='center',fontsize=10)
    save(fig,'posterior-slices')


def agreement():
    h=np.load(OUT/'h2-fields.npz');d=np.load(TIMINGS/'reference-70652.npz')
    hp=ndtr(-h['mean'][:128]/np.sqrt(h['variance'][:128]));dp=ndtr(-d['mean']/np.sqrt(d['variance']))
    primary=json.loads((TIMINGS/'primary.json').read_text())
    fig=plt.figure(figsize=(13.7,4.8));grid=fig.add_gridspec(1,3,width_ratios=[1,1,1.15],left=.025,right=.98,bottom=.19,top=.87,wspace=.30)
    ax=fig.add_subplot(grid[0,0]);image(ax,'probes-h2-observed');ax.set_title('The 128 benchmark queries',fontsize=12)
    cax=fig.add_axes([.055,.135,.20,.025]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0,1),cmap='viridis'),cax=cax,orientation='horizontal');cb.set_label('Marginal inside probability',fontsize=10)
    ax=fig.add_subplot(grid[0,1]);ax.plot([0,1],[0,1],color='#a2a2a2',lw=1,zorder=0)
    ax.scatter(dp,hp,c=hp,cmap='viridis',vmin=0,vmax=1,s=22,edgecolors='#252525',linewidths=.3)
    ax.set(xlabel='Full-kernel reference',ylabel='Mixed H²',xlim=(-.025,1.025),ylim=(-.025,1.025));ax.set_aspect('equal');ax.set_title('Probability agreement',fontsize=12)
    ax.text(.04,.94,'max |Δp| = %.2g'%np.max(abs(hp-dp)),transform=ax.transAxes,va='top',fontsize=10)
    ax=fig.add_subplot(grid[0,2])
    for method,color,label in [('h2','#176c91','Mixed H²'),('dense','#cc6e1f','Mixed Cholesky')]:
        pp=primary[method]['points'];n=[v['n'] for v in pp];t=[v['seconds'] for v in pp]
        ax.loglog(n,t,'o-',color=color,label=label,lw=1.8,ms=5)
        ax.fill_between(n,[v['minimum'] for v in pp],[v['maximum'] for v in pp],color=color,alpha=.15)
    ax.set(xlabel='Normal observations N',ylabel='Fresh setup + 128 queries (s)');ax.grid(which='both',alpha=.15);ax.legend(fontsize=9,loc='upper left')
    ax.set_title('Measured runtime crossover',fontsize=12)
    ax.text(.98,.03,'70,652 points\n17.82 s vs 22.46 s\n1.26× speedup',ha='right',transform=ax.transAxes,fontsize=10)
    fig.text(.5,.025,'Timings reused from PR122: three-run medians, mean + variance + CDF, unchanged numerical gates. Both methods fit in GPU memory.',ha='center',fontsize=10)
    save(fig,'query-agreement')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','geometry','slices','agreement','all'])
    a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='geometry':geometry()
    elif a.action=='slices':slices()
    elif a.action=='agreement':agreement()
    else:geometry();slices();agreement()
