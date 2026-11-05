"""Pin full-query checks for the scaling figures to exact raw payload hashes."""
import argparse
import json
from pathlib import Path

from reproduction.verify_eagle_comparison import main as periodic
from reproduction.compare_hierarchical_application import compare


def main(root):
    root=Path(root);periodic(root);out=root/'checks';out.mkdir(exist_ok=True)
    old=Path('reproduction/results/poissonrecon_large/eagle');checks=[];diagnostics=[]
    for n in [8192,32768,77824,196608]:
        h=root/f'free-space/h2-streamed-{n}.json';s=root/f'free-space/sparse-streamed-{n}.json'
        baseline=old/(f'tuned/dense-float32-{n}-0.json' if n<196608 else 'cached-control-15/sparse-gpu-196608-0.json')
        pairs=[(h,s,'pair')] if n==8192 else [(h,s,'pair'),(h,baseline,'h2'),(s,baseline,'locality')]
        for candidate,ref,label in pairs:
            target=out/f'{label}-{n}.json';compare(str(candidate),str(ref),str(target))
            (diagnostics if n==196608 and label!='pair' else checks).append(str(target))
    checks.append(str(old/'application-796825.json'))
    (root/'diagnostic_checks_free.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    (root/'full_query_checks_free.json').write_text(json.dumps(checks,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='reproduction/results/asymptotic_regimes')
    main(p.parse_args().root)
