"""CPU diagnostic of ideal sign-event budgets on retained B controls.

This does not replace numerical gates or certify total application error.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from reproduction.event_probability_budget import (
    sign_probability_budget, prefix_probability_budget, scanning_score_intervals)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def evaluate(folder):
    folder=Path(folder);root=Path(__file__).resolve().parents[1]
    record=json.loads((folder/'results.json').read_text())
    if record['args']['case']!='B' or not record['complete']:
        raise ValueError('completed B controls with independent reference required')
    with np.load(folder/'controls.npz') as a, np.load(folder/'outputs.npz') as o:
        mean=a['mean'];s=np.diag(a['covariance']);u=np.diag(a['upper'])
        observed_mean_error=np.abs(mean-a['reference_mean'])
        p=o['query_rays'].mean(0);collision=float(o['query_collision'].mean())
    distance=np.load(root/'reproduction/results/analytic_posterior/B/geometry.npz')['distance']
    if mean.shape!=(1024,) or p.shape!=(3,256):raise ValueError('unexpected retained panel')
    epsilon=float(np.sqrt(np.log(2*769/.05)/(2*record['samples'])))
    result={'scope':'ideal PSD solve/truncation law-error diagnostics; no total numerical, physical-model or continuous-geometry certificate',
        'rank':record['query_record']['basis']['rank'],'original_quality_pass':record['quality_pass'],
        'mean_reference_difference_max':float(observed_mean_error.max()),
        'mean_reference_qualification':'observed difference from independent dense numerical reference, not a certified mean-error enclosure',
        'simultaneous_mc_radius95':epsilon,'mc_events':769,'samples':record['samples'],
        'mc_scope':'iid represented Gaussian; Hoeffding plus union over 768 ray prefixes and one collision event',
        'threshold_grid':{'kind':'geomspace','minimum':.5,'maximum':30.,'count':128},
        'bound_method':'approximate_marginal_density_plus_gaussian_difference_tail',
        'coupling':'existence Gaussian independent-perturbation coupling; not the shared-principal-root comparison',
        'excluded_errors':['prior/cross/analytic-action floating point','mean reference numerical error',
                           'covariance/root projection','continuous geometry discretization','physical/model calibration'],
        'source_sha256':{p: digest(root/p) for p in ['reproduction/event_probability_budget.py','reproduction/evaluate_event_probability_budget.py']},
        'artifact_sha256':{p:digest(folder/p) for p in ['results.json','controls.npz','outputs.npz']},'scenarios':{}}
    for name,d in [('equal_means',0.),('observed_dense_mean_difference',observed_mean_error)]:
        node=sign_probability_budget(mean,s,u,d)['node_bound']
        prefix=prefix_probability_budget(node[:768].reshape(3,256))
        sphere=float(min(1,node[768:].sum()))
        scores=scanning_score_intervals(p,distance,prefix,epsilon)
        result['scenarios'][name]={'maximum_node_bound':float(node.max()),
            'node_bound':node.tolist(),'ray_prefix_bound':prefix.tolist(),
            'whole_ray_bounds':prefix[:,-1].tolist(),'sphere8_bound':sphere,
            'sphere8_probability':collision,'sphere8_interval95':[max(0,collision-epsilon-sphere),min(1,collision+epsilon+sphere)],
            'score':scores['score'].tolist(),'score_lower':scores['score_lower'].tolist(),
            'score_upper':scores['score_upper'].tolist(),'winner_ray':[0,8,17][scores['winner']],
            'winner_separated_under_assumptions':scores['winner_separated']}
    (folder/'event_budget.json').write_text(json.dumps(result,indent=2)+'\n')
    return {k:{j:v for j,v in case.items() if j not in ['node_bound','ray_prefix_bound']} for k,case in result['scenarios'].items()}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',nargs='+');args=p.parse_args()
    for folder in args.folder:print(folder,json.dumps(evaluate(folder)),flush=True)
