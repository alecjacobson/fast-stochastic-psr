"""Figures from retained observed-fidelity measurements."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from reproduction.observed_fidelity import OUT, OLD


def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/(name+'.'+ext),dpi=190,bbox_inches='tight')
    plt.close(fig)


def validation():
    selection=json.loads((OUT/'selection.json').read_text());rows=selection['rows']
    control=json.loads((OUT/'screen-n16384-ell0.48-sigma0.5.json').read_text())['records'][0]
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for c,marker in [(512,'o'),(2048,'s')]:
        for sigma,color in [(.02,'#217f9b'),(.05,'#ad6547')]:
            rr=sorted([r for r in rows if r['contacts']==c and r['sigma']==sigma and abs(r['scalar_variance_over_prior']-1e-4)<1e-10],key=lambda r:r['ell'])
            label=f'C={c}, σ={sigma}'
            for ax,key in zip(axes,['median','p95','normal_p95_degrees']):
                ax.plot([r['ell'] for r in rr],[r['fidelity'][key] for r in rr],marker=marker,color=color,linestyle='-' if c==2048 else '--',label=label)
    for ax,key,limit in zip(axes,['median','p95','normal_p95_degrees'],[.0033029994296663997,2*.0033029994296663997,40]):
        ax.scatter([.48],[control['fidelity'][key]],c='#555555',marker='x',s=55,label='Previous setting')
        ax.axhline(limit,color='#777777',lw=1,ls=':',label='Fidelity gate')
        ax.set_xscale('log');ax.set_xlabel('Length scale ℓ');ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
        ax.set_xticks([.015,.03,.06,.12,.48],['.015','.03','.06','.12','.48'])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[0].set_ylabel('Median input crossing displacement');axes[1].set_ylabel('95th percentile displacement');axes[2].set_ylabel('95th percentile normal error (degrees)')
    axes[1].legend(fontsize=7);fig.suptitle('Fit the observed surface first · 16,384 normals · new scalar variance/prior = 10⁻⁴')
    save(fig,'validation-fidelity')


def final():
    old=json.loads((OUT/'test-side-control-fit.json').read_text())
    new=json.loads((OUT/'test-side-selected.json').read_text())
    config=new['configuration']
    of=np.load(OUT/'test-side-control-fit.npz');nf=np.load(OUT/'test-side-selected.npz')
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),layout='constrained')
    for fields,label,color in [(of,'Previous ℓ=.48','#8b8b8b'),(nf,f"New ℓ={config['ell']}",'#217f9b')]:
        for ax,key in zip(axes[:2],['root_error','normal_angle']):
            values=np.sort(fields[key]);ax.plot(values,np.arange(1,len(values)+1)/len(values),label=label,color=color)
    axes[0].axvline(new['fidelity']['spacing'],color='#333333',ls=':',label='Median sample spacing')
    axes[0].set_xlim(0,.031);axes[0].set_xlabel('Input displacement (missing roots → .03)');axes[0].set_ylabel('Cumulative fraction of probes');axes[0].legend(fontsize=8)
    axes[1].set_xlim(0,80);axes[1].set_xlabel('Gradient/input-normal angle (degrees)')
    oldq=json.loads((OLD/'test_side-selected-70652-reference.json').read_text())
    for k,(record,label,color) in enumerate([(oldq,'Previous','#8b8b8b'),(new,'New setting','#217f9b')]):
        axes[2].bar(np.arange(2)+(k-.5)*.35,[record['occupancy'][s]['brier'] for s in ['observed','occluded']],width=.34,label=label,color=color)
    axes[2].set_xticks([0,1],['Observed','Unobserved']);axes[2].set_ylabel('Inside-probability Brier score');axes[2].legend(fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('70,652 raw range observations · improved fit, separate completion tradeoff')
    save(fig,'fit-and-probability')
    fig,axes=plt.subplots(2,4,figsize=(12,8.5),gridspec_kw=dict(wspace=.01,hspace=.01))
    for i,view in enumerate(['observed','back']):
        for j,path in enumerate([OLD.parent/'partial_renderings'/f'reference-{view}.png',
                                  OUT/'render-input'/f'input-{view}.png',
                                  OUT/'render-control'/f'mean-{view}.png',OUT/'render-selected'/f'mean-{view}.png']):
            axes[i,j].imshow(plt.imread(path));axes[i,j].set_axis_off()
    for ax,title in zip(axes[0],['Reference','70,652 raw\nrange observations','Previous mean\nℓ=.48 · σ=.5 · 512 contacts',
        f"New mean\nℓ={config['ell']} · σ={config['sigma']} · {config['contacts']:,} contacts"]):ax.set_title(title,fontsize=10)
    fig.text(.5,.04,'Matched cameras and uniform 240×288×240 grids · full-kernel means · no mesh repair',ha='center',fontsize=9)
    save(fig,'geometry-comparison')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['validation','final']);args=p.parse_args()
    if args.action=='validation':validation()
    else:final()
