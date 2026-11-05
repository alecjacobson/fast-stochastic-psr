"""Hold full-Eagle solver/cache settings fixed at the smaller overlap sizes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path('reproduction/results/poissonrecon_large/eagle')
TEMPLATES={'h2-streamed':ROOT/'eight-large/h2-streamed-796825-0.json',
           'sparse-streamed':ROOT/'large-control/sparse-streamed-796825-0.json'}


def main(root,sizes):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    for method,template in TEMPLATES.items():
        for n in sizes:
            path=root/f'{method}-{n}.json'
            a=json.loads(template.read_text())['arguments'];a.update(n=n,output=str(path))
            args=[]
            for k,v in a.items():
                if k=='point_cloud_sha256' or v is None or v is False:continue
                args.append('--'+k.replace('_','-'))
                if v is not True:args.append(str(v))
            if not path.exists():
                with path.with_suffix('.log').open('w') as log:
                    result=subprocess.run([sys.executable,'-m','reproduction.benchmark_hierarchical_variance',*args],
                                          stdout=log,stderr=subprocess.STDOUT,timeout=600)
                if result.returncode:raise RuntimeError(str(path))
            reference=ROOT/f'references/{n}.json'
            if not reference.exists():
                reference=root/f'references/{n}.json';reference.parent.mkdir(exist_ok=True)
            subprocess.run([sys.executable,'-m','reproduction.audit_hierarchical_variance',str(path),
                            '--reference',str(reference),'--random-queries','8'],check=True)
            print(path,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='reproduction/results/asymptotic_regimes/free-space')
    p.add_argument('--sizes',type=int,nargs='+',default=[8192,32768,77824,196608])
    a=p.parse_args();main(a.root,a.sizes)
