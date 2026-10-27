"""Replay fixed-panel covariance, moments and finite-grid event controls."""
import argparse,json
from pathlib import Path
import numpy as np
from reproduction.common_rhs_integrity import array_digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['A','B'],required=True)
    p.add_argument('--panel',choices=['moments','applications'],required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/f'reproduction/results/query_posterior/{args.case}_{args.panel}'
    r=json.loads((out/'results.json').read_text());assert r['complete'] and r['quality_pass'] and r['gpu_process_audit']['sampled_isolation_pass']
    data=np.load(root/f'reproduction/results/matched/{args.case}/input.npz');c=np.load(out/'controls.npz');o=np.load(out/'outputs.npz')
    for key,value in [('points',data['points']),('normals',data['normals']),('queries',c['queries'])]:assert array_digest(value)==r['hashes'][key]
    cov=c['covariance'];ref=c['reference_covariance'];mean=c['reference_mean'];m=len(mean)
    error=float(np.linalg.norm(cov-ref)/np.linalg.norm(ref))
    assert error<1e-7 and np.max(np.abs(np.diag(cov-ref))/np.diag(ref))<1e-7
    np.testing.assert_allclose(error,r['covariance_relative_l2'],rtol=1e-9,atol=1e-16)
    for solve in r['solve_records']:assert solve['converged'] and max(solve['relative_residual'])<=1e-10
    eig=np.linalg.eigvalsh((cov+cov.T)*.5)
    budget=r['query_record']['psd']['roundoff_projection_budget']
    assert eig.min()>=-budget
    results={'passed':False,'scope':'retained covariance, input hashes, fields/events and moment summaries; not regeneration of all Fourier priors or full training solves',
             'covariance_relative_l2':error,'minimum_cpu_covariance_eigenvalue':float(eig.min()),'event_confidence':{}}
    target=ref[:min(32,m),:min(32,m)];mu=mean[:min(32,m)]
    for side in ['query','reference','source']:
        fields=o[side+'_fields'];count=len(fields)
        se=np.sqrt(np.diag(target)/count)
        cse=np.sqrt((target**2+np.diag(target)[:,None]*np.diag(target)[None,:])/(count-1))
        observed=[float(np.max(np.abs(fields.mean(0)-mu)/se)),float(np.max(np.abs(np.cov(fields.T)-target)/cse))]
        expected=[r['source_moment_sanity'][side]['max_mean_standardized_error'],r['source_moment_sanity'][side]['max_covariance_standardized_error']]
        np.testing.assert_allclose(observed,expected,rtol=1e-9,atol=1e-9)
        assert max(observed)<6
        # Hoeffding for each Bernoulli marginal, then a union bound over the
        # fixed finite family. No independence between queries is assumed.
        family=769 if args.panel=='applications' else 1
        epsilon=float(np.sqrt(np.log(2*family/.05)/(2*count)))
        entry={'samples':count,'family_size':family,'confidence':.95,'simultaneous_probability_radius':epsilon,
               'qualification':'iid Gaussian draws from the represented covariance; numerical/model bias and spatial discretization are not included',
               'joint_inside_probability':float(o[side+'_collision'].mean())}
        if args.panel=='applications':
            rays=o[side+'_rays'].mean(0);distance=np.load(root/f'reproduction/results/analytic_posterior/{args.case}/geometry.npz')['distance']
            lo=np.maximum(0,rays-epsilon);hi=np.minimum(1,rays+epsilon)
            scores=np.trapezoid(((rays>=.05)&(rays<=.95)).astype(float),distance,axis=1)
            score_lo=np.trapezoid(((lo>=.05)&(hi<=.95)).astype(float),distance,axis=1)
            score_hi=np.trapezoid(((lo<=.95)&(hi>=.05)).astype(float),distance,axis=1)
            winner=int(scores.argmax())
            entry.update(ray_ids=[0,8,17],sphere_id=8,scores=scores.tolist(),score_lower=score_lo.tolist(),score_upper=score_hi.tolist(),
                         best_ray=int([0,8,17][winner]),best_is_certified=bool(score_lo[winner]>np.delete(score_hi,winner).max()))
        results['event_confidence'][side]=entry
    for key in ['rays','collision']:
        disagreements=int(np.count_nonzero(o['query_'+key]!=o['reference_'+key]))
        summary=next(d for d in r['query_draws'] if d['count']==r['samples'])
        assert disagreements==summary['ray_disagreements' if key=='rays' else 'collision_disagreements']
        assert disagreements/o['query_'+key].size<.0001 if o['query_'+key].size else disagreements==0
    results['passed']=True
    (out/'validation.json').write_text(json.dumps(results,indent=2)+'\n');print(args.case,args.panel,results,flush=True)


if __name__=='__main__':main()
