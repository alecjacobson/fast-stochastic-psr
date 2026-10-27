"""CPU replay of retained query-covariance budgets and independent controls."""
import argparse,json
from pathlib import Path
import numpy as np
from reproduction.query_error import residual_envelopes,relative_covariance_budget
from reproduction.common_rhs_integrity import array_digest


def validate(folder):
    folder=Path(folder);r=json.loads((folder/'results.json').read_text());a=np.load(folder/'controls.npz')
    assert r['complete'] and r['gpu_process_audit']['sampled_isolation_pass']
    assert array_digest(a['queries'])==r['hashes']['queries'] and array_digest(a['rows'])==r['hashes']['rows']
    root=Path(__file__).resolve().parents[1]
    if r['case']=='B':
        source=np.load(root/'reproduction/results/matched/B/input.npz');x,y=source['points'],source['normals']
    else:
        x=np.load(root/'reproduction/results/range_compact/scale1m/control.npz')['points']
        y=np.load(root/'reproduction/results/analytic_refinement/million/controls.npz')['normals']
    assert array_digest(x)==r['hashes']['points'] and array_digest(y)==r['hashes']['normals']
    s=a['covariance'];u=a['upper'];shift=a['shift'];noise=.02
    for matrix in [s,u]:
        assert np.isfinite(matrix).all()
        assert np.linalg.eigvalsh(matrix).min()>=-100*np.finfo(float).eps*len(matrix)*np.linalg.norm(matrix)
    np.testing.assert_allclose(s+shift,a['raw_covariance'],rtol=1e-12,atol=1e-20)
    np.testing.assert_allclose(np.diag(u),np.sum(a['residual_norms']**2,axis=1)/noise,rtol=1e-12)
    budget=relative_covariance_budget(s,u);saved=r['query_record']['residual_audit']['relative_budget']
    np.testing.assert_allclose(budget['upper_relative_to_exact_covariance'],saved['upper_relative_to_exact_covariance'],rtol=1e-10)
    budget_pass=bool(budget['informative'] and budget['upper_relative_to_exact_covariance']<=r['gates']['relative_covariance_solve_budget'])
    assert budget_pass==r['budget_pass']
    first,second=residual_envelopes(a['cross_norms'],a['residual_norms'],noise)
    assert np.all(np.abs(u)<=second+1e-22)
    result={'replay_pass':True,'budget_pass':budget_pass,'relative_solve_budget':budget,
            'scope':'retained small-matrix and sampled-row replay; does not regenerate every training action or certify floating-point bounds'}
    if r['case']=='B':
        target=a['reference_covariance'];error=s-target;exact=a['exact_remaining_error'];scale=np.linalg.norm(target)
        ref=np.load(root/'reproduction/results/query_posterior/B_moments/controls.npz')
        np.testing.assert_array_equal(target,ref['reference_covariance'])
        discrepancy=np.linalg.norm(error-exact)/scale
        assert discrepancy<=1e-10
        assert np.linalg.eigvalsh(u-exact).min()>=-1e-10*scale
        assert np.all(np.abs(error)<=second+1e-10*scale)
        assert np.all(np.abs(a['raw_covariance']-target)<=first+1e-10*scale)
        eig,vec=np.linalg.eigh(target)
        relative=(vec.T@error@vec)/np.sqrt(eig[:,None]*eig[None,:])
        observed=float(np.max(np.abs(np.linalg.eigvalsh((relative+relative.T)*.5))))
        assert observed<=budget['upper_relative_to_exact_covariance']+1e-10
        result.update(corrected_covariance_relative_l2=float(np.linalg.norm(error)/scale),identity_discrepancy_relative_l2=float(discrepancy),
                      observed_relative_operator_error=observed)
    else:
        rhs=a['sampled_rhs'];ra=a['sampled_range_residual'];rd=a['sampled_direct_residual'];source=a['sampled_assembled_rhs']
        denom=np.maximum(np.linalg.norm(rhs,axis=0),1e-300)
        for name,value in [('residual',rd-ra),('cross',rhs-source),('action',rd-ra-(rhs-source))]:
            metric=np.linalg.norm(value,axis=0)/denom
            np.testing.assert_allclose(metric,r['direct_validation'][name+'_discrepancy_over_rhs_per_column'],rtol=1e-10,atol=1e-18)
            assert metric.max()<=1e-9
            result[name+'_discrepancy_over_rhs_max']=float(metric.max())
        assert r['mean_weights_relative_l2']<=1e-7
    assert r['validation_pass']
    result['quality_pass']=budget_pass
    (folder/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',nargs='+');a=p.parse_args()
    for folder in a.folder:print(folder,json.dumps(validate(folder)))
