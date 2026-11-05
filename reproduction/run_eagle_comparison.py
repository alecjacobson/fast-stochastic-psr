"""Bounded, sequential GPU/CPU runs for missing Eagle comparison baselines."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def main(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for method, sizes in [('dense',[2048,4096,8192,32768]),('nufft-hybrid',[2048,4096,8192]),('sdd',[2048,8192,32768])]:
        for n in sizes:
            extra=['--eta','.03','--budget','120'] if method=='sdd' else []
            jobs.append((f'periodic/{method}-{n}', ['--method',method,'--n',str(n)]+extra))
    for n in [2048,8192,32768]:
        jobs.append((f'periodic/sdd-{n}-eta01',['--method','sdd','--n',str(n),'--eta','.1','--budget','120']))
        jobs.append((f'periodic/nufft-cg-{n}-tuned',['--method','nufft-cg','--n',str(n),
                    '--variance-rtol','1e-3','--preconditioner-block','128','--rhs-block-size','4']))
        jobs.append((f'periodic/nufft-cg-{n}-wide',['--method','nufft-cg','--n',str(n),
                    '--variance-rtol','1e-3','--preconditioner-block','128','--rhs-block-size','32']))
    for n in [2048,8192,32768]:
        jobs.append((f'original/full-grid16-{n}',['--method','original','--n',str(n),'--grid','16']))
    for grid in [12,20]:
        jobs.append((f'original/full-grid{grid}-8192',['--method','original','--n','8192','--grid',str(grid)]))
    for name, args in jobs:
        out=root/(name+'.json');out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():
            d=json.loads(out.read_text())
            if d.get('complete') or d.get('status','').startswith('censored'):
                # Existing pilot records use distinct names. Do not silently
                # replace measured results when regenerating a plot.
                continue
        command=[sys.executable,'-m','reproduction.benchmark_eagle_comparison',*args,'--output',str(out)]
        start=perf_counter();status=dict(command=command)
        with out.with_suffix('.log').open('w') as log:
            try:
                done=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600)
                status['returncode']=done.returncode
            except subprocess.TimeoutExpired:
                status['timeout_seconds']=600
        status['wall_seconds']=perf_counter()-start
        out.with_suffix('.status.json').write_text(json.dumps(status,indent=2)+'\n')
        print(name,status,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='reproduction/results/eagle_comparison')
    main(p.parse_args().root)
