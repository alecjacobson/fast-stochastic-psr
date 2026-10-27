"""Replay stored factor controls and moments; raw draws are not regenerated."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from reproduction.common_rhs_integrity import array_digest
from reproduction.benchmark_correction_contrast import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    a=p.parse_args();r=json.loads((a.output/'results.json').read_text())
    if not r['complete']:raise ValueError('incomplete run')
    expected={'contrast_relative_l2':.01,'probability_rms':.0005,'probability_max':.005}
    if r['gates']!=expected or r['control_precision_gate']!=.001:raise ValueError('gates changed')
    root=Path(__file__).resolve().parents[1]
    data=np.load(root/f"reproduction/results/matched/{r['case']}/input.npz")
    if r['hashes']!={label:array_digest(data[key]) for label,key in [('points','points'),('normals','normals'),('queries','query')]}:
        raise ValueError('source/query mismatch')
    original=root/f"reproduction/results/correction_contrast/{r['case']}/reference.npz"
    if hashlib.sha256(original.read_bytes()).hexdigest()!=r['reference_payload_sha256']:
        raise ValueError('reference payload mismatch')
    ref=np.load(a.output/'reference.npz');truth,mean,prior=ref['variance'],ref['mean'],float(ref['prior_variance'])
    old=np.load(original)
    if not np.array_equal(ref['query'],data['query']) or not np.array_equal(truth,old['variance']) or prior!=float(old['prior_variance']):
        raise ValueError('reference arrays mismatch')
    cnorm=float(np.linalg.norm(truth-truth.mean()));hashes=0;estimates=0;max_control_error=0.
    good=lambda m:all(m[k]<=v for k,v in expected.items()) and m['nonpositive_count']==0
    for run in r['runs']:
        controls={0:np.zeros_like(truth)}
        arrays=np.load(a.output/f"seed_{run['seed']}.npz")
        for rank,setting in run['settings'].items():
            values={}
            for key,digest in setting['hashes'].items():
                value=np.load(a.archive/f"{run['seed']}_{rank}_{key}.npy",mmap_mode='r')
                if array_digest(value)!=digest:raise ValueError('archive hash mismatch')
                hashes+=1;values[key]=value
            v=values['basis'];u=values['factor_basis'].astype('d')
            if v.shape!=(3*r['N'],int(rank)) or u.shape!=(r['queries'],int(rank)):
                raise ValueError('factor/basis shape mismatch')
            if np.linalg.norm(v.T@v-np.eye(int(rank)))>1e-8:
                raise ValueError('basis orthogonality mismatch')
            d=np.sum(u*u,axis=1)
            error=float(np.linalg.norm(d-values['control'])/cnorm);max_control_error=max(max_control_error,error)
            if error>1e-9:raise ValueError('control replay mismatch')
            precision=float(np.linalg.norm(values['control']-values['reference_control'])/cnorm)
            if not np.isclose(precision,setting['precision_discrepancy_over_contrast'],rtol=1e-8,atol=1e-12):raise ValueError('precision check mismatch')
            if bool(precision<=.001)!=setting['precision_pass']:raise ValueError('precision gate mismatch')
            controls[int(rank)]=values['control']
        for check in run['checkpoints']:
            rank,count=check['rank'],check['count'];key=f'{rank}_{count}'
            variance=prior-(controls[rank]+arrays['residual_second_moment_'+key])
            if not np.array_equal(variance,arrays['variance_'+key]):raise ValueError('variance replay mismatch')
            probability=ndtr(-mean/np.sqrt(np.maximum(variance,np.finfo(float).tiny)))
            if not np.array_equal(probability,arrays['probability_'+key]):raise ValueError('probability replay mismatch')
            m=metrics(variance,truth,mean,prior)
            if m!=check['metrics']:raise ValueError('metric mismatch')
            passed=good(m) and (rank==0 or run['settings'][str(rank)]['precision_pass'])
            if bool(passed)!=check['quality_pass']:raise ValueError('quality gate mismatch')
            setup=(sum(run['settings'][str(rank)][k] for k in ['sketch_seconds','adjoint_seconds','qr_seconds','forward_seconds','control_seconds'])+r['seconds']['action_setup']) if rank else 0.
            total=setup+r['seconds']['query_plan']+check['shared_generation_query_seconds']+check['reduction_seconds']+check['output_seconds']
            if not np.isclose(total,check['first_estimate_seconds'],rtol=1e-12,atol=1e-12):raise ValueError('cost accounting mismatch')
            estimates+=1
    candidates=[]
    for rank in r['ranks']:
        for count in r['looks']:
            checks=[next(c for c in run['checkpoints'] if (c['rank'],c['count'])==(rank,count)) for run in r['runs']]
            if all(c['quality_pass'] for c in checks):candidates.append({'rank':rank,'count':count,'median_first_estimate_seconds':float(np.median([c['first_estimate_seconds'] for c in checks]))})
    selected=min(candidates,key=lambda c:c['median_first_estimate_seconds']) if candidates else None
    if selected!=r['selected']:raise ValueError('selection mismatch')
    result={'complete':True,'archive_arrays_checked':hashes,'estimates_replayed':estimates,
            'variance_probability_replay_exact':True,'maximum_control_error_over_contrast':max_control_error,
            'selection_and_cost_accounting_checked':True,
            'scope':'saved basis/factor controls and residual moments; does not regenerate sketch/evaluation draws or rerun factor actions'}
    (a.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
