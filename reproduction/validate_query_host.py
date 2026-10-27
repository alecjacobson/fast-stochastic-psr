"""Replay host-storage covariance and coherent finite-grid event controls."""
import argparse,json
from pathlib import Path
import numpy as np
from reproduction.common_rhs_integrity import array_digest


def validate(folder):
    folder=Path(folder);r=json.loads((folder/'results.json').read_text());a=np.load(folder/'controls.npz')
    assert r['complete'] and r['gpu_process_audit']['sampled_isolation_pass']
    root=Path(__file__).resolve().parents[1];case=r['args']['case'];panel=r['args']['panel']
    if case=='B':
        data=np.load(root/'reproduction/results/matched/B/input.npz');x,y=data['points'],data['normals']
        ref=np.load(root/f'reproduction/results/query_posterior/B_{panel}/controls.npz')
        target,mean=ref['reference_covariance'],ref['reference_mean']
    else:
        x=np.load(root/'reproduction/results/range_compact/scale1m/control.npz')['points']
        y=np.load(root/'reproduction/results/analytic_refinement/million/controls.npz')['normals']
        ref=np.load(root/'reproduction/results/query_error/million_c100/controls.npz');target,mean=ref['covariance'],ref['mean']
        for k in ['upper','shift']:
            value=np.linalg.norm(a[k]-ref[k])/np.linalg.norm(ref[k])
            np.testing.assert_allclose(value,r[k+'_relative_l2_difference'],rtol=1e-8,atol=1e-14)
            # Separate-run finite-NUFFT solves can change tiny residuals; the
            # storage identity is checked below on exactly the same vectors.
    if 'same_vector_products' in r:
        for k in ['upper','shift']:
            target_product=a['same_vector_reference_'+k]
            metric=np.linalg.norm(a[k]-target_product)/max(np.linalg.norm(target_product),1e-300)
            np.testing.assert_allclose(metric,r['same_vector_products'][k+'_relative_l2'],rtol=1e-8,atol=1e-15)
            assert metric<=1e-11
    for key,value in [('points',x),('normals',y),('queries',a['queries'])]:assert array_digest(value)==r['hashes'][key]
    np.testing.assert_array_equal(a['reference_covariance'],target);np.testing.assert_array_equal(a['reference_mean'],mean)
    s=a['covariance'];u=a['upper']
    metrics={'covariance_relative_l2':float(np.linalg.norm(s-target)/np.linalg.norm(target)),
        'marginal_variance_relative_max':float(np.max(np.abs(np.diag(s-target))/np.diag(target))),
        'mean_error_over_prior_std':float(np.max(np.abs(a['mean']-mean)/np.sqrt(np.diag(a['prior']))))}
    for k,v in metrics.items():
        np.testing.assert_allclose(v,r[k],rtol=1e-8,atol=1e-14)
    quality=all(v<=r['gates'][k] for k,v in metrics.items())
    assert np.linalg.eigvalsh((s+s.T)*.5).min()>=-r['query_record']['psd']['roundoff_projection_budget']
    assert np.linalg.eigvalsh((u+u.T)*.5).min()>=-100*np.finfo(float).eps*len(u)*np.linalg.norm(u)
    np.testing.assert_allclose(np.diag(u),np.sum(a['residual_norms']**2,axis=1)/.02,rtol=1e-11,atol=1e-28)
    result={'passed':True,'metrics':metrics,'scope':'retained covariance, input hashes and events; not regeneration of every analytic action or all field values'}
    if r['args'].get('adaptive_marginal'):
        ratio=np.diag(u)/(np.diag(s)-np.diag(u))
        assert np.all(np.diag(s)>np.diag(u)) and ratio.max()<=r['gates']['marginal_variance_relative_max']
        batch=r['query_record']['batch']
        for begin,solve in zip(range(0,len(s),batch),r['finite_solves'],strict=True):
            assert solve['converged']
            np.testing.assert_allclose(ratio[begin:begin+batch].max(),solve['budget_history'][-1]['maximum_relative_marginal_bound'],rtol=1e-8,atol=1e-14)
        trace=float(np.trace(u));norm=float(np.linalg.norm(s));assert norm>trace
        bound=trace/(norm-trace)
        assert metrics['covariance_relative_l2']<=bound+1e-12 and bound<=r['gates']['covariance_relative_l2']
        result['solve_only_budgets']={'maximum_relative_marginal':float(ratio.max()),'relative_covariance_frobenius':bound,
            'qualification':'exact-arithmetic solve bounds; numerical and model errors excluded'}
    if panel=='applications':
        o=np.load(folder/'outputs.npz');result['events']={};result['confidence']={}
        for k in ['rays','collision']:
            count=int(np.count_nonzero(o['query_'+k]!=o['reference_'+k]));rate=count/o['query_'+k].size
            assert count==r['events'][k]['disagreements']
            quality=quality and rate<r['gates']['paired_event_disagreement_rate']
            result['events'][k]={'disagreements':count,'rate':rate}
        distance=np.load(root/'reproduction/results/analytic_posterior/B/geometry.npz')['distance']
        epsilon=float(np.sqrt(np.log(2*769/.05)/(2*r['samples'])))
        for side in ['query','reference']:
            prob=o[side+'_rays'].mean(0);lo=np.maximum(0,prob-epsilon);hi=np.minimum(1,prob+epsilon)
            scores=np.trapezoid(((prob>=.05)&(prob<=.95)).astype(float),distance,axis=1)
            lower=np.trapezoid(((lo>=.05)&(hi<=.95)).astype(float),distance,axis=1)
            upper=np.trapezoid(((lo<=.95)&(hi>=.05)).astype(float),distance,axis=1)
            winner=int(scores.argmax())
            result['confidence'][side]={'sphere8_probability':float(o[side+'_collision'].mean()),'scores':scores.tolist(),
                'score_lower':lower.tolist(),'score_upper':upper.tolist(),'best_ray':int([0,8,17][winner]),
                'best_is_certified':bool(lower[winner]>np.delete(upper,winner).max()),'simultaneous_probability_radius95':epsilon,
                'qualification':'finite-grid iid sampling only; excludes numerical/model bias and spatial discretization'}
        observed=np.linalg.norm(o['query_fields']-o['reference_fields'])/np.linalg.norm(o['reference_fields'])
        np.testing.assert_allclose(observed,r['paired_field_relative_l2_first32'],rtol=1e-8,atol=1e-14)
    assert quality==r['quality_pass'];result['quality_pass']=quality
    (folder/'validation.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',nargs='+');a=p.parse_args()
    for folder in a.folder:print(folder,json.dumps(validate(folder)))
