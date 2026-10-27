"""Check fixed-setting paired batch16 and batch4 variance outputs."""
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'results/factor_deflation'


def main():
    result={}
    for case in 'AB':
        old=ROOT/f'confirmation/{case}';new=ROOT/f'batch16/{case}'
        a=json.loads((old/'results.json').read_text());b=json.loads((new/'results.json').read_text())
        if not a['complete'] or not b['complete']:raise ValueError('incomplete run')
        for key in ['hashes','seeds','ranks','looks','estimator']:
            if a[key]!=b[key]:raise ValueError(f'{key} changed')
        ref=np.load(old/'reference.npz')['variance'];cnorm=float(np.linalg.norm(ref-ref.mean()))
        rank,count=a['ranks'][0],a['looks'][0];key=f'variance_{rank}_{count}'
        errors=[]
        for run in a['runs']:
            name=f"seed_{run['seed']}.npz";v0=np.load(old/name)[key];v1=np.load(new/name)[key]
            errors.append(float(np.linalg.norm(v1-v0)/cnorm))
        result[case]={'matched_settings_and_seeds':True,'contrast_relative_discrepancy':errors,
                      'gate':.001,'pass':bool(max(errors)<=.001),
                      'batch4_result_sha256':hashlib.sha256((old/'results.json').read_bytes()).hexdigest(),
                      'batch16_result_sha256':hashlib.sha256((new/'results.json').read_bytes()).hexdigest()}
    (ROOT/'batch_comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
