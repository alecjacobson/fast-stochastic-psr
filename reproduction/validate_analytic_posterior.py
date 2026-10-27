"""Replay paired outputs and first-draw full analytic controls."""
import argparse,json
from pathlib import Path
import cupy as cp
import numpy as np
from geospsr.spectral_tail import analytic_covariance_rows64
from geospsr.training_spectrum import FourierCovariance,finite_covariance_rows
from geospsr.square_root import FixedChebyshevRoot
from reproduction.common_rhs_integrity import array_digest
from reproduction.paired_applications import field_quality,quality


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['A','B'],required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'reproduction/results/analytic_posterior'/args.case
    r=json.loads((out/'results.json').read_text());assert r['complete'] and r['gpu_process_audit']['sampled_isolation_pass']
    data=np.load(root/f'reproduction/results/matched/{args.case}/input.npz');x,y=data['points'],data['normals']
    assert array_digest(x)==r['input_hashes']['points'] and array_digest(y)==r['input_hashes']['normals']
    geometry=np.load(out/'geometry.npz')
    for key,expected in r['geometry_hashes'].items():assert array_digest(geometry[key])==expected
    outputs=np.load(out/'outputs.npz')
    ref={k:outputs['paired_reference_'+k] for k in ['fields','rays','collision']}
    actual={k:outputs['production_'+k] for k in ['fields','rays','collision']}
    fields=field_quality(cp.asarray(ref['fields']),cp.asarray(actual['fields']))
    event=quality(ref,actual,geometry['distance'])
    assert fields==r['field_quality'] and event==r['event_quality']
    assert fields['pass'] and event['compatibility_pass'] and event['uncertainty_coverage_pass']
    first=np.load(out/'first_draw.npz');a=analytic_covariance_rows64(x,x);a.flat[::len(x)+1]+=.02
    weights,rhs=cp.asarray(first['production_weights']),cp.asarray(first['production_rhs'])
    residual=(cp.linalg.norm(a@weights-rhs,axis=0)/cp.linalg.norm(rhs,axis=0)).get()
    assert residual.max()<=1e-8
    np.testing.assert_allclose(residual,r['draws'][0]['independent_solve_residual_per_rhs'],rtol=1e-5,atol=1e-13)
    low=finite_covariance_rows(x,x,FourierCovariance(50,dtype='float64'),dtype='float64')
    tail=a-(low+low.T)*.5;white=cp.asarray(first['white_noise'])
    if args.case=='A':
        eig,vec=cp.linalg.eigh(tail);expected=(vec*cp.sqrt(eig))@(vec.T@white)
    else:
        p=FixedChebyshevRoot(r['root_bound']['lower'],r['root_bound']['upper'],1e-7)
        expected=p.apply(lambda v:tail@v,white,xp=cp)
    tail_error=float(cp.linalg.norm(cp.asarray(first['production_tail'])-expected)/cp.linalg.norm(expected))
    assert tail_error<=1e-7
    targets=np.load(out/'moment_target.npz');cov=targets['covariance'];mean=targets['mean'];ids=targets['query_indices']
    for side in ['production','paired_reference','dense_cholesky']:
        draws=outputs[side+'_fields'][:,ids]
        mean_z=float(np.max(np.abs(draws.mean(0)-mean)/np.sqrt(np.diag(cov)/len(draws))))
        cov_z=float(np.max(np.abs(np.cov(draws.T)-cov)/np.sqrt((cov**2+np.diag(cov)[:,None]*np.diag(cov)[None,:])/(len(draws)-1))))
        np.testing.assert_allclose([mean_z,cov_z],[r['moment_sanity'][side]['max_mean_standardized_error'],r['moment_sanity'][side]['max_covariance_standardized_error']],rtol=1e-12,atol=1e-12)
    result={'passed':True,'scope':'all retained fields/events and moment summaries; regenerated first-draw dense analytic residual and paired root; no fresh full-ensemble regeneration or absolute application accuracy claim',
            'first_draw_relative_residual':residual.tolist(),'first_draw_tail_relative_l2':tail_error}
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(args.case,result,flush=True)


if __name__=='__main__':main()
