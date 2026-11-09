"""Plots of retained issue124 measurements; no new model evaluations."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reproduction.partial_quality import OUT


def screening():
    rows=json.loads((OUT/'screen-selection.json').read_text())['rows']
    fig,axes=plt.subplots(2,3,figsize=(10.2,6.3),sharex=True,sharey=True,layout='constrained')
    ell=[.06,.12,.24,.48];sigma=[.05,.2,.5]
    for i,n in enumerate([768,16384]):
        for j,c in enumerate([32,128,512]):
            a=np.array([[next(r['balanced_root_mae'] for r in rows if (r['n'],r['ell'],r['sigma'],r['contacts'])==(n,e,s,c)) for s in sigma] for e in ell])
            ax=axes[i,j];im=ax.imshow(a,origin='lower',vmin=.03,vmax=.095,cmap='viridis_r',aspect='auto')
            for y in range(4):
                for x in range(3):
                    ax.text(x,y,f'{a[y,x]:.3f}',ha='center',va='center',fontsize=8,color='white' if a[y,x]>.067 else '#161616')
            ax.set_xticks(range(3),sigma);ax.set_yticks(range(4),ell)
            ax.set_title(f'N = {n:,} · {c} contacts',fontsize=10)
            if i==1:ax.set_xlabel('Normal-noise σ')
            if j==0:ax.set_ylabel('Lengthscale ℓ')
    fig.colorbar(im,ax=axes,label='Balanced surface-probe MAE (lower is better)',shrink=.9,pad=.02)
    fig.suptitle('Validation geometry: contact coverage matters more than shortening ℓ',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(OUT/f'validation-grid.{ext}',dpi=180)
    plt.close(fig)


def occupancy_geometry():
    rows=json.loads((OUT/'screen-selection.json').read_text())['rows']
    fig,axes=plt.subplots(1,2,figsize=(9.6,4),sharex=True,sharey=True,layout='constrained')
    for ax,n in zip(axes,[768,16384]):
        for c,color in [(32,'#8b8b8b'),(128,'#d09024'),(512,'#2385a1')]:
            ss=[r for r in rows if r['n']==n and r['contacts']==c]
            ax.scatter([r['balanced_root_mae'] for r in ss],[r['occupancy']['all']['brier'] for r in ss],color=color,label=f'{c} contacts',s=28)
        old=next(r for r in rows if (r['n'],r['ell'],r['sigma'],r['contacts'])==(n,.48,.2,32))
        selected=next(r for r in rows if (r['n'],r['ell'],r['sigma'],r['contacts'])==(n,.48,.5,512))
        ax.scatter(old['balanced_root_mae'],old['occupancy']['all']['brier'],marker='x',c='black',s=65,label='Old setting')
        ax.scatter(selected['balanced_root_mae'],selected['occupancy']['all']['brier'],marker='*',c='#b84036',s=150,label='Promoted setting')
        ax.set_title(f'{n:,} validation observations');ax.set_xlabel('Balanced surface-probe MAE ↓');ax.grid(alpha=.18)
    axes[0].set_ylabel('Raw probability Brier score ↓');axes[1].legend(fontsize=8,loc='upper right')
    for ext in ['png','pdf']:fig.savefig(OUT/f'validation-tradeoff.{ext}',dpi=180)
    plt.close(fig)




def final_figures():
    data=json.loads((OUT/'summary.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    colors=['#8b8b8b','#217f9b']
    ax=axes[0]
    for k,tag in enumerate(['old','selected']):
        stats=json.loads((OUT/f'display-{tag}-120.json').read_text())['statistics']
        ax.bar(np.arange(2)+(k-.5)*.32,[stats[s]['median'] for s in ['observed','occluded']],width=.31,color=colors[k],label=tag.capitalize())
    ax.set_xticks([0,1],['Observed','Unobserved']);ax.set_ylabel('Median distance to mean surface ↓');ax.set_title('Reference-to-mean geometry');ax.legend(fontsize=8)
    ax=axes[1]
    heldout=[r for r in data['quality'] if r['group']!='validation']
    for k,(key,label,color) in enumerate([('old_occupancy','Old full variance','#8b8b8b'),('new_occupancy','Selected full variance','#217f9b'),('new_contact_only','Selected contact-only variance','#dc9a35')]):
        ax.bar(np.arange(3)+(k-1)*.25,[r[key]['all']['brier'] for r in heldout],width=.24,label=label,color=color)
    ax.set_xticks(range(3),['Side\n70,652','Back\n16,384','Other side\n16,384']);ax.set_ylabel('Raw probability Brier ↓');ax.set_title('Held-out inside probabilities');ax.legend(fontsize=7)
    ax=axes[2]
    labels=list(data['primary']);rows=[data['primary'][label] for label in labels]
    med=np.array([r['seconds_median'] for r in rows]);err=np.array([[m-r['seconds_range'][0],r['seconds_range'][1]-m] for m,r in zip(med,rows)]).T
    ax.bar(range(len(rows)),med,yerr=err,color=['#217f9b','#a16450'][:len(rows)],capsize=4)
    ax.set_xticks(range(len(rows)),labels);ax.set_ylabel('Fresh task seconds ↓');ax.set_title('128 mean + variance queries')
    for i,(m,r) in enumerate(zip(med,rows)):ax.text(i,m+max(med)*.025,f'{m:.2f} s\n{r["gpu_gib_max"]:.2f} GiB',ha='center',va='bottom',fontsize=8)
    ax.set_ylim(0,max(med)*1.3)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15)
    for ext in ['png','pdf']:fig.savefig(OUT/f'quality-and-runtime.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(12,8.3),gridspec_kw={'wspace':.015,'hspace':.015})
    for i,view in enumerate(['observed','back']):
        paths=[OUT.parent/'partial_renderings'/f'reference-{view}.png',OUT/'render-selected'/f'input-{view}.png',
               OUT/'render-old'/f'mean-{view}.png',OUT/'render-selected'/f'mean-{view}.png']
        for j,path in enumerate(paths):axes[i,j].imshow(plt.imread(path));axes[i,j].set_axis_off()
    for ax,title in zip(axes[0],['Reference','Observed scan\n512 contacts','Old mean\nσ=.2 · 32 contacts','Selected mean\nσ=.5 · 512 contacts']):ax.set_title(title,fontsize=10)
    fig.text(.5,.025,'70,652 observations · ℓ=.48 · matched 80×96×80 display grids · actual full-kernel posterior means',ha='center',fontsize=10)
    for ext in ['png','pdf']:fig.savefig(OUT/f'geometry-comparison.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def scaling():
    data=json.loads((OUT/'summary.json').read_text())['scaling']
    fig,axes=plt.subplots(1,2,figsize=(9.6,3.7),layout='constrained')
    for method,label,color,marker in [('h2-joint','H² joint correction','#217f9b','o'),('corrected-wide','Mixed Cholesky','#a16450','s')]:
        rows=data[method]['points'];n=np.array([r['n'] for r in rows]);t=np.array([r['seconds'] for r in rows]);memory=[r['gpu_gib'] for r in rows]
        error=np.array([[v-r['seconds_range'][0],r['seconds_range'][1]-v] for v,r in zip(t,rows)]).T
        axes[0].errorbar(n,t,yerr=error,marker=marker,color=color,label=f"{label} (slope {data[method]['empirical_loglog_slope']:.2f})",capsize=4)
        axes[1].plot(n,memory,marker=marker,color=color,label=label)
    for ax in axes:
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Normal observations N');ax.grid(which='both',alpha=.17)
        ax.set_xticks(n,[f'{v:,}' for v in n]);ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator());ax.legend(fontsize=8)
    axes[0].set_ylabel('Fresh mean + variance task (seconds)');axes[1].set_ylabel('Sampled peak GPU process memory (GiB)')
    axes[0].set_title('512 contacts + 128 fixed queries');axes[1].set_title('Both methods fit at the measured crossover')
    for ext in ['png','pdf']:fig.savefig(OUT/f'quality-crossover.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',nargs='?',choices=['validation','final'],default='validation')
    if p.parse_args().action=='validation':screening();occupancy_geometry()
    else:final_figures();scaling()
