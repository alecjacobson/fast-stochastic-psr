"""Preparation, remaining estimate cost, and all-seed timing ranges."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent/'results/packed_factor_sources'


def main():
    r=json.loads((ROOT/'summary.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4))
    methods=['generic','packed','packed_planar'];labels=['Generic','Packed sources','Packed + planar']
    for ax,case in zip(axes,'AB'):
        for i,method in enumerate(methods):
            m=r[case][method];t=np.array(m['first_estimate_seconds']);index=np.argsort(t)[1]
            prep=m['preparation_seconds'][index];total=t[index]
            ax.barh(i,prep,color='#376787',label='Preparation' if i==0 else None)
            ax.barh(i,total-prep,left=prep,color='#9ac6df',label='Evaluation / output' if i==0 else None)
            ax.errorbar(total,i,xerr=[[total-t.min()],[t.max()-total]],fmt='none',ecolor='black',capsize=4)
            ax.scatter(m['deterministic_median_seconds'],i,marker='D',color='#b24d36',zorder=3,label='Fresh deterministic median' if i==0 else None)
        ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.set_xlabel('Variance-stage sum including preparation (s)')
        ax.set_title(f'{case}: all 20,000 slice queries');ax.grid(axis='x',alpha=.2)
        ax.set_xlim(left=0)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
    fig.suptitle('Fixed settings; medians and complete three-seed ranges',fontsize=12)
    fig.tight_layout(rect=(0,.1,1,.96));fig.savefig(ROOT/'timings.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
