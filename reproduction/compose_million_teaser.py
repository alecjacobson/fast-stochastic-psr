"""Compose computed fields and Blender point cloud in the paper's Figure 1 style."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,Normalize
from matplotlib.patches import ConnectionPatch,FancyArrowPatch
from matplotlib import patheffects

root=Path(__file__).resolve().parents[1];out=root/'reproduction/results/teaser'
f=np.load(out/'fields.npz');meta=json.loads((out/'million.json').read_text())
projection=json.loads((out/'cloud_projection.json').read_text())
cloud=plt.imread(out/'cloud_render.png')
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral','DejaVu Serif'],
                     'mathtext.fontset':'stix','font.size':18,'svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(16,6.1),facecolor='white')
# The two planes in the 3D render have exactly the same coordinates as these fields.
cloudrect=[.002,.075,.405,.87]
cloudax=fig.add_axes(cloudrect);cloudax.imshow(cloud);cloudax.axis('off')
cols=[.425,.601,.815];width=.161;rows=[.56,.105];height=.422
mean=f['mean'];var=f['variance'];prob=f['probability']
mn=TwoSlopeNorm(vmin=float(mean.min()),vcenter=0,vmax=float(mean.max()))
vn=Normalize(0,float(var.max()));pn=Normalize(0,1)
axes={}
for row,h in enumerate(f['heights']):
    for col,(values,cmap,norm,title) in enumerate([(mean,'RdBu',mn,'Mean'),(var,'plasma',vn,'Variance'),(prob,'viridis',pn,r'$\mathbb{P}(x\in\Omega)$')]):
        ax=fig.add_axes([cols[col],rows[row],width,height])
        image=ax.imshow(values[row],origin='lower',extent=[-1.7,1.7,-1.7,1.7],cmap=cmap,norm=norm,interpolation='bilinear',aspect='equal')
        if col==0:
            contours=ax.contour(f['axis'],f['axis'],values[row],levels=[0],colors='white',linewidths=1.15)
            contours.set_path_effects([patheffects.Stroke(linewidth=1.85,foreground='#263b4b'),patheffects.Normal()])
        if row==0:
            ax.text(.05,.97,title,transform=ax.transAxes,ha='left',va='top',fontsize=19,color='black' if col==1 else 'white')
        ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values():sp.set_color('#25333b');sp.set_linewidth(.7)
        axes[row,col]=ax
    axes[row,0].text(.035,.035,f'$y={h:g}$',va='bottom',ha='left',transform=axes[row,0].transAxes,fontsize=12,color='white')
# Project actual slice corners into the point-cloud axes and connect to row edges.
for row,anchor in enumerate(projection['anchors']):
    con=ConnectionPatch(xyA=anchor,coordsA=cloudax.transAxes,
                        xyB=(0,.50),coordsB=axes[row,0].transAxes,
                        color='#222',linewidth=.8,zorder=0)
    fig.add_artist(con)
# A short curved arrow links the GP fields to the marginal statistical query.
arrow=FancyArrowPatch((.775,.76),(.805,.76),transform=fig.transFigure,
                      connectionstyle='arc3,rad=-.6',arrowstyle='-|>',mutation_scale=13,linewidth=.9,color='#222')
fig.add_artist(arrow)
fig.text(.195,.055,'Input point cloud',ha='center',va='center',fontsize=20)
fig.text(.195,-.005,'1,000,000 observations',ha='center',va='bottom',fontsize=15,color='#36454f')
fig.text(.59,.045,'NUFFT stochastic reconstruction',ha='center',va='center',fontsize=19)
fig.text(.895,.045,'Statistical query',ha='center',va='center',fontsize=19)
# Compact numerical scales, outside the image panels. Shared scales across slices.
for i,(norm,cmap,ticks) in enumerate([(mn,'RdBu',[float(mean.min()),0,float(mean.max())]),(vn,'plasma',[0,float(var.max())]),(pn,'viridis',[0,.5,1])]):
    cax=fig.add_axes([cols[i]+.015,.005,width-.03,.012])
    cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',ticks=ticks)
    cb.ax.tick_params(labelsize=9,pad=1,length=2)
    if i<2:cb.ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1e'))
    cb.outline.set_linewidth(.4)
fig.savefig(root/'assets/scorpion-million.png',dpi=220,bbox_inches='tight',pad_inches=.07)
fig.savefig(root/'assets/scorpion-million.pdf',bbox_inches='tight',pad_inches=.07)
fig.savefig(root/'assets/scorpion-million.svg',bbox_inches='tight',pad_inches=.07)
svg=root/'assets/scorpion-million.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
print('Saved assets/scorpion-million.{png,pdf,svg}')
