"""CPU replay of projected covariance, budgets, and finite-grid event evidence."""
import argparse,json
from pathlib import Path
import numpy as np
from reproduction.query_projection import assemble_projected
from reproduction.common_rhs_integrity import array_digest


def validate(folder):
    folder=Path(folder);r=json.loads((folder/'results.json').read_text());a=np.load(folder/'controls.npz');o=np.load(folder/'outputs.npz')
    assert r['complete'] and r['gpu_process_audit']['sampled_isolation_pass']
    root=Path(__file__).resolve().parents[1]
    if r['args']['case']=='B':
        source=np.load(root/'reproduction/results/matched/B/input.npz');x,y=source['points'],source['normals']
    else:
        x=np.load(root/'reproduction/results/range_compact/scale1m/control.npz')['points']
        y=np.load(root/'reproduction/results/analytic_refinement/million/controls.npz')['normals']
    original_queries=np.load(root/'reproduction/results/query_posterior/B_applications/controls.npz')['queries']
    for key,value in [('points',x),('normals',y),('queries',original_queries)]:assert array_digest(value)==r['hashes'][key]
    # The model explicitly promotes query coordinates to double. Hash the
    # original supplied bytes and independently verify the exact promotion.
    assert a['queries'].dtype==np.dtype('float64')
    np.testing.assert_array_equal(a['queries'],original_queries.astype('float64'))
    s,upper,budget=assemble_projected(a['prior'],a['basis'],a['cross_action'],a['weight_residual'],a['tail'],a['solve_upper'])
    assert np.linalg.norm(s-a['covariance'])/np.linalg.norm(s)<1e-11
    np.testing.assert_allclose(upper,a['upper'],rtol=1e-10,atol=1e-24)
    for k in ['relative_covariance_frobenius_bound','maximum_relative_marginal_bound']:
        np.testing.assert_allclose(budget[k],r['query_record']['budget'][k],rtol=1e-9,atol=1e-18)
    budget_pass=bool(budget['relative_covariance_frobenius_bound']<=1e-7 and budget['maximum_relative_marginal_bound']<=1e-7)
    assert budget_pass==r['budget_pass']
    projection=np.eye(len(s))-a['basis']@a['basis'].T
    raw_tail=projection@a['prior']@projection
    assert np.linalg.norm(raw_tail-a['tail'])<=r['query_record']['basis']['prior_roundoff_allowance']*len(s)
    assert np.linalg.eigvalsh(s).min()>=-r['query_record']['psd']['roundoff_projection_budget']
    result={'replay_pass':True,'budget_pass':budget_pass,'budget':budget,
        'scope':'retained covariance/budgets, input hashes and all event arrays; not regeneration of every solve/NUFFT or a global floating-point certificate'}
    quality=budget_pass
    if r['args']['case']=='B':
        ref=np.load(root/'reproduction/results/query_posterior/B_applications/controls.npz');target=ref['reference_covariance'];mu=ref['reference_mean']
        np.testing.assert_array_equal(target,a['reference_covariance']);np.testing.assert_array_equal(mu,a['reference_mean'])
        error=s-target;metrics={'covariance_relative_l2':float(np.linalg.norm(error)/np.linalg.norm(target)),
            'maximum_marginal_relative_error':float(np.max(np.abs(np.diag(error))/np.diag(target))),
            'mean_error_over_prior_std':float(np.max(np.abs(a['mean']-mu)/np.sqrt(np.diag(a['prior']))))}
        for k,v in metrics.items():
            np.testing.assert_allclose(v,r[k],rtol=1e-6,atol=1e-13);quality=quality and v<=r['gates'][k]
        result['metrics']=metrics
        result['upper_minus_error_min_eigenvalue']=float(np.linalg.eigvalsh(upper-error).min())
        assert result['upper_minus_error_min_eigenvalue']>=-r['query_record']['basis']['prior_roundoff_allowance']
        result['events']={}
        for k in ['rays','collision']:
            diff=int(np.count_nonzero(o['query_'+k]!=o['reference_'+k]));rate=diff/o['query_'+k].size
            assert diff==r['events'][k]['disagreements'];quality=quality and rate<r['gates']['event_disagreement_rate']
            result['events'][k]={'disagreements':diff,'rate':rate}
    else:
        metric=np.linalg.norm(a['sampled_direct_residual']-a['sampled_captured_residual'],axis=1)/np.maximum(np.linalg.norm(a['sampled_rhs'],axis=1),1e-300)
        np.testing.assert_allclose(metric,r['direct_validation']['per_component_basis_column'],rtol=1e-9,atol=1e-18)
        result['maximum_direct_row_discrepancy_over_rhs']=float(metric.max())
        quality=quality and metric.max()<=1e-7 and r['mean_weights_relative_l2']<=1e-7
    distance=np.load(root/'reproduction/results/analytic_posterior/B/geometry.npz')['distance']
    epsilon=float(np.sqrt(np.log(2*769/.05)/(2*r['samples'])))
    rays=o['query_rays'].mean(0);lo=np.maximum(0,rays-epsilon);hi=np.minimum(1,rays+epsilon)
    score=np.trapezoid(((rays>=.05)&(rays<=.95)).astype(float),distance,axis=1)
    lower=np.trapezoid(((lo>=.05)&(hi<=.95)).astype(float),distance,axis=1)
    upper_score=np.trapezoid(((lo<=.95)&(hi>=.05)).astype(float),distance,axis=1)
    winner=int(score.argmax())
    result['finite_grid_sampling']={'sphere8_probability':float(o['query_collision'].mean()),'best_ray':int([0,8,17][winner]),
        'scores':score.tolist(),'score_lower':lower.tolist(),'score_upper':upper_score.tolist(),
        'best_is_certified':bool(lower[winner]>np.delete(upper_score,winner).max()),
        'simultaneous_probability_radius95':epsilon,'qualification':'iid represented Gaussian on finite grid; numerical/model bias and geometry discretization excluded'}
    fields=o['query_fields'];target=s[:32,:32];mu=a['mean'][:32]
    se=np.sqrt(np.diag(target)/len(fields));cse=np.sqrt((target**2+np.diag(target)[:,None]*np.diag(target)[None,:])/(len(fields)-1))
    result['moment_sanity']={'mean_max_standardized_error':float(np.max(np.abs(fields.mean(0)-mu)/se)),
        'covariance_max_standardized_error':float(np.max(np.abs(np.cov(fields.T)-target)/cse))}
    assert max(result['moment_sanity'].values())<6
    assert bool(quality)==r['quality_pass'];result['quality_pass']=bool(quality)
    (folder/'validation.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',nargs='+');args=p.parse_args()
    for folder in args.folder:print(folder,json.dumps(validate(folder)))
