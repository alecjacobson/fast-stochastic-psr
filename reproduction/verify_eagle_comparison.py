"""Replay all periodic candidate fields against a same-model dense field."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
from cupyx.scipy.special import ndtr

from reproduction.benchmark_hierarchical_variance import read_field, query_batches


def main(root):
    root=Path(root);results=[]
    for path in sorted((root/'periodic').glob('*.json')):
        if path.name.endswith(('.status.json','.audit.json')):continue
        d=json.loads(path.read_text())
        if not d.get('complete') or not d.get('quality',{}).get('passes'):continue
        a=d['arguments'];reference=root/'periodic'/f"dense-{a['n']}.json"
        if a['method']=='dense':continue
        base=json.loads(reference.read_text())
        if not base.get('complete') or not base.get('quality',{}).get('passes'):
            raise ValueError('Missing passing dense reference '+str(reference))
        for key in ['point_sha256','normal_sha256','queries_total','model']:
            if d[key]!=base[key]:raise ValueError('Mismatched '+key)
        f,ref=read_field(d['field']),read_field(base['field'])
        pe=ve=prior_error=constant_error=0.;winners_match=True;positive=True
        contact_variance=float(f.predict([[1,0,0]])[1][0])
        for q in query_batches(SimpleNamespace(**a)):
            m,v=f.predict_gpu(q);rm,rv=ref.predict_gpu(q)
            positive &= bool(cp.all(v>0) and cp.all(cp.isfinite(m)) and cp.all(cp.isfinite(v)))
            p=ndtr((d['level']-m)/cp.sqrt(v));rp=ndtr((base['level']-rm)/cp.sqrt(rv))
            pe=max(pe,float(cp.max(abs(p-rp))));ve=max(ve,float(cp.max(abs(v-rv)))/f.prior_variance)
            prior_error=max(prior_error,float(cp.max(abs(p-ndtr((d['level']-m)/cp.sqrt(f.prior_variance))))))
            constant_error=max(constant_error,float(cp.max(abs(p-ndtr((d['level']-m)/cp.sqrt(contact_variance))))))
            winners_match &= int(cp.argmax((4*p*(1-p)).reshape(-1,64).mean(1)))==int(cp.argmax((4*rp*(1-rp)).reshape(-1,64).mean(1)))
        row=dict(candidate=str(path),reference=str(reference),
                 candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                 queries=a['queries']*a['batches'],probability_max_error=pe,variance_max_error_over_prior=ve,
                 passes=bool(positive and pe<=1e-3 and ve<=1e-4),all_batch_winners_match=bool(winners_match),
                 shortcut_prior_max_probability_difference=prior_error,
                 shortcut_constant_contact_variance_max_probability_difference=constant_error)
        results.append(row);print(path.name,row['passes'],pe,ve,flush=True)
        del f,ref;cp.get_default_memory_pool().free_all_blocks()
    (root/'full_query_checks.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='reproduction/results/eagle_comparison')
    main(p.parse_args().root)
