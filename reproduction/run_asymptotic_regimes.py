"""Fixed-code geometric sweeps, bounded by overlap wins or resources (#104)."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter


def jobs():
    for n in [2048,4096,8192,16384,32768,65536,77824,98304]:
        yield f'periodic/dense-{n}', ['--method','dense','--n',str(n)]
    for method in ['sdd','nufft-cg','nufft-hybrid']:
        for n in ([2048,8192] if method=='nufft-hybrid' else [2048,8192,32768]):
            extra=['--eta','.1'] if method=='sdd' else ['--rhs-block-size','4'] if method=='nufft-cg' else ['--moment-dtype','float64']
            yield f'periodic/{method}-{n}', ['--method',method,'--n',str(n),*extra]
    for n in [2048,8192,32768]:
        yield f'original/full-grid16-{n}', ['--method','original','--n',str(n),'--grid','16']
    for grid in [12,20,24,28]:
        yield f'original/full-grid{grid}-8192', ['--method','original','--n','8192','--grid',str(grid)]
    for rank in [64,256,1024,2048,3072]:
        yield f'original/subspace{rank}-grid16-8192', ['--method','original','--n','8192','--grid','16','--subspace',str(rank)]
    for cutoff in [75,100,150]:
        yield f'bandwidth/dense-2048-cutoff{cutoff}', ['--method','dense','--n','2048','--cutoff',str(cutoff)]


def main(root,group):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    source=hashlib.sha256(Path('reproduction/benchmark_eagle_comparison.py').read_bytes()).hexdigest()
    for name,args in jobs():
        if group and not name.startswith(group):continue
        out=root/(name+'.json');out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():
            d=json.loads(out.read_text())
            if d.get('source_sha256')!=source:raise ValueError('Stale implementation: '+str(out))
            for key,value in zip(args[::2],args[1::2]):
                old=d['arguments'][key[2:].replace('-','_')]
                if old!=type(old)(value):raise ValueError('Changed settings: '+str(out))
            status_path=out.with_suffix('.status.json')
            if status_path.exists():
                status=json.loads(status_path.read_text())
                relevant=['reproduction/variance_series.py'] if d['arguments']['method']=='nufft-hybrid' else ['reproduction/scaling_reference.py'] if d['arguments']['method']=='dense' else []
                for dependency in relevant:
                    recorded=status.get('dependency_sha256',{}).get(dependency)
                    if recorded and recorded!=hashlib.sha256(Path(dependency).read_bytes()).hexdigest():
                        raise ValueError('Changed dependency: '+dependency)
            if d.get('complete') or d.get('status','').startswith('censored') or d.get('failure'):continue
        command=[sys.executable,'-m','reproduction.benchmark_eagle_comparison',*args,'--output',str(out)]
        start=perf_counter();status=dict(command=command,source_sha256=source,
            dependency_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [Path('reproduction/scaling_reference.py'),Path('reproduction/variance_series.py')]})
        def host_cap():resource.setrlimit(resource.RLIMIT_AS,(12*2**30,12*2**30))
        with out.with_suffix('.log').open('w') as log:
            try:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600,
                                      preexec_fn=host_cap if name.startswith('original/') else None)
                status['returncode']=result.returncode
            except subprocess.TimeoutExpired:status['timeout_seconds']=600
        status['wall_seconds']=perf_counter()-start
        out.with_suffix('.status.json').write_text(json.dumps(status,indent=2)+'\n')
        print(name,status,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='reproduction/results/asymptotic_regimes');p.add_argument('--group')
    a=p.parse_args();main(a.root,a.group)
