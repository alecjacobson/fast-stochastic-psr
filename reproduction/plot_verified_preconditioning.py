"""Measured stage sums; explicitly distinguish the rejected original budget."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    root=Path(__file__).resolve().parent/'results/verified_preconditioning'
    original=json.loads((root/'summary.json').read_text());strict=json.loads((root/'strict/summary.json').read_text())
    labels={'plain':'Plain packed CG','float32':'Float32, two passes','float32_onepass':'Float32, one pass','fused16':'Fused float16'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),sharex=True)
    max_total=0
    for ax,record,keys,title in [
        (axes[0],original,['plain','float32','fused16'],'Original threshold: 2e−5'),
        (axes[1],strict,['plain','float32','float32_onepass','fused16'],'Tighter threshold: 2e−6')]:
        for row,key in enumerate(keys):
            r=record['runs'][key]
            if not r['complete']:
                ax.text(0,row,'run failed');continue
            t=r['seconds'];setup=t['coarse_setup'];solve=t['eight_solves'];total=setup+solve
            max_total=max(max_total,total)
            ax.barh(row,setup,color='#d8a24e',label='Coarse setup' if row==0 else None)
            ax.barh(row,solve,left=setup,color='#3c708c',label='Eight verified solves' if row==0 else None,
                    hatch='////' if not r['quality_pass'] else None,edgecolor='#243744')
            ax.text(total+2,row,f'{total:.1f} s'+('  FAIL' if not r['quality_pass'] else ''),va='center',fontsize=9)
        ax.set_yticks(range(len(keys)),[labels[k] for k in keys]);ax.invert_yaxis()
        ax.set_title(title,fontsize=11);ax.set_xlabel('Setup + conditional solves (seconds)')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_xlim(0,max_total*1.3)
    axes[0].legend(handles=[Patch(facecolor='#d8a24e',label='Coarse setup'),
                            Patch(facecolor='#3c708c',label='Eight verified solves')],loc='lower right',fontsize=8)
    fig.suptitle('Three million observations: setup changes the first-use comparison',fontsize=13)
    note='Single eight-draw runs. Prepared coherent RHS; mean fit, prior generation and queries excluded.\nHatching: covariance gate failed. '+('All four tighter runs pass numerical and sampled-isolation gates.' if strict['timing_ratios_eligible'] else 'Tighter-run timing ratios are ineligible.')
    fig.text(.5,.015,note,ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.94))
    fig.savefig(root/'conditioning.png',dpi=180);fig.savefig(root/'conditioning.pdf');plt.close(fig)


if __name__=='__main__':main()
