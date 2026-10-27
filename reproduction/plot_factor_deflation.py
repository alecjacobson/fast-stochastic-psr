"""All predeclared ranks and all seed ranges, including preparation costs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent/'results/factor_deflation'


def main():
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharey=True)
    for col,case in enumerate('AB'):
        r=json.loads((ROOT/f'pilot/{case}/results.json').read_text())
        for rank in [0]+r['ranks']:
            checks=[[next(c for c in run['checkpoints'] if (c['rank'],c['count'])==(rank,count))
                     for count in r['looks']] for run in r['runs']]
            errors=np.array([[100*c['metrics']['contrast_relative_l2'] for c in row] for row in checks])
            times=np.median([[c['first_estimate_seconds'] for c in row] for row in checks],axis=0)
            for row,x in enumerate([r['looks'],times]):
                ax=axes[row,col]
                line,=ax.loglog(x,np.median(errors,axis=0),'--' if rank==0 else '-o',markersize=4,
                               label='Raw moments' if rank==0 else f'Rank {rank}')
                ax.fill_between(x,errors.min(axis=0),errors.max(axis=0),color=line.get_color(),alpha=.18)
        for row in range(2):
            ax=axes[row,col];ax.axhline(1,color='black',ls=':',lw=1);ax.grid(alpha=.2,which='both')
            ax.legend(fontsize=8,ncol=2)
            if col==0:ax.set_ylabel('Variance error / reference spatial contrast (%)')
        axes[0,col].set_title(f'{case}, N = {r["N"]:,}, 20,000 queries')
        axes[0,col].set_xlabel('Evaluation draws (sketch/factor preparation separate)')
        axes[1,col].set_xlabel('First-estimate stage sum including preparation (s)')
        axes[1,col].axvline(r['deterministic_median_seconds'],color='grey',ls=':',lw=1)
    fig.suptitle('Factor-space calibration: medians and three-seed error ranges',fontsize=12)
    fig.tight_layout();fig.savefig(ROOT/'convergence.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
