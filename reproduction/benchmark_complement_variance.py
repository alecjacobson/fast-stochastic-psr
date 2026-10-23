"""Released-A marginal variance pilot; no timing or full-application claim."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cupy as cp
import cupyx.scipy.linalg as cla
import numpy as np
from scipy.special import ndtr
from geospsr.spectral import FourierPoisson
from geospsr.spectral_tail import analytic_covariance_rows64
from reproduction.posterior import Posterior, sync
from reproduction.complement_variance import estimate, pilot_weights
from reproduction.gpu_audit import GpuProcessAudit


def error_metrics(value, exact, mean):
    valid = value > 0
    probability = np.full_like(value, np.nan)
    probability[valid] = ndtr(-mean[valid]/np.sqrt(value[valid]))
    target = ndtr(-mean/np.sqrt(exact))
    return {'relative_variance_l2': float(np.linalg.norm(value-exact)/np.linalg.norm(exact)),
            'maximum_pointwise_relative_variance_error': float(np.max(np.abs(value/exact-1))),
            'nonpositive_estimates': int((~valid).sum()),
            'probability_rms': float(np.sqrt(np.mean((probability-target)**2))) if valid.all() else None,
            'probability_max': float(np.max(np.abs(probability-target))) if valid.all() else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    record = {'complete': False, 'scope': 'released-A marginal estimator pilot, no speedup claim',
              'model': 'analytic vector covariance / cutoff50 scalar, corrected noise variance .02',
              'pilot_draws': 32, 'evaluation_draws': 128, 'seeds': [107,211,307],
              'seconds': {}, 'runs': [],
              'gates': {'numerical_component_relative_l2': .0005, 'variance_relative_l2': .01,
                        'probability_rms': .005, 'probability_max': .02}}
    audit = GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    def timed(name, fn):
        sync(); start = time.perf_counter(); value = fn(); sync()
        record['seconds'][name] = time.perf_counter()-start
        print(name, record['seconds'][name], flush=True)
        return value
    try:
        root = Path(__file__).resolve().parents[1]
        data = np.load(root/'reproduction/results/matched/A/input.npz')
        x, y = data['points'], data['normals']
        ids = np.linspace(0, len(data['query'])-1, 128, dtype=int)
        observed = np.linspace(0, len(x)-1, 128, dtype=int)
        q = np.concatenate([data['query'][ids], x[observed]])
        record['N'], record['query_count'] = len(x), len(q)
        record['hashes'] = {k: hashlib.sha256(a.tobytes()).hexdigest()
                            for k,a in [('points',x),('normals',y),('query',q)]}
        model = timed('fit', lambda: Posterior(x,y,upsampfac=1.25))
        # Dense float64 analytic factorization and higher-accuracy NUFFT cross
        # sums form the reference. The separate small test uses independent real
        # features; this large spectral reference is not fully independent code.
        op = FourierPoisson(50, dtype='float64', eps=1e-12)
        matrix = analytic_covariance_rows64(x,x)+.02*cp.eye(len(x),dtype='d')
        chol = timed('reference_factor', lambda: cp.linalg.cholesky(matrix))
        def solve(b):
            return cla.solve_triangular(chol.T,cla.solve_triangular(chol,b,lower=True),lower=False)
        def cross_values():
            pieces = []
            for start in range(0,len(q),32):
                differences = q[start:start+32,None].astype('d')-x[None].astype('d')
                pieces.append(op.cross_values(differences.reshape(-1,3)).reshape(-1,len(x),3))
            return cp.concatenate(pieces)
        cross = timed('reference_cross', cross_values)
        rhs = cross.transpose(1,0,2).reshape(len(x),-1)
        reduction = (rhs*solve(rhs)).sum(0).reshape(len(q),3).sum(1)
        norm = (op.ns**2).sum(0)
        v0 = float(cp.sum(cp.where(norm>0,op.rho/cp.where(norm>0,norm,1),0)))
        exact = v0-cp.asnumpy(reduction)
        mean = cp.asnumpy(cp.einsum('qnd,nd->q',cross,solve(cp.asarray(y,dtype='d'))))-float(model.offset)
        assert np.all(exact > 0)
        reference_correction = cp.asnumpy(reduction)
        record['posterior_to_prior_variance_range'] = [float(exact.min()/v0),float(exact.max()/v0)]
        record['oracle_variance_reduction_range'] = [float(np.min(1+(exact/reference_correction)**2)),
                                                    float(np.max(1+(exact/reference_correction)**2))]
        np.savez_compressed(args.output/'reference.npz',query=q,variance=exact,mean=mean,
                            prior_variance=v0,correction_variance=reference_correction)
        for seed in record['seeds']:
            timed(f'draw_{seed}',lambda: model.draw(160,seed=seed,batch=4,rng_backend='gpu',
                  joint_solve=True,retain_components=True,retain_rhs=True))
            def query_components():
                centered = model.sample_coeff-model.mean_coeff
                correction = model.sample_prior_coeff-centered
                return (cp.asnumpy(model.op.evaluate_batched(q,centered,batch=4)),
                        cp.asnumpy(model.op.evaluate_batched(q,correction,batch=4)))
            g,c = timed(f'components_{seed}',query_components)
            # Check numerical conditioning and query bias on these same draws.
            # RHS is normals-z, so z=normals-RHS; no new randomness is introduced.
            z = (model.normals[None]-model.sample_rhs).transpose(1,0,2).reshape(len(x),-1).astype('d')
            exact_c = cp.asnumpy(cp.einsum('qnd,snd->sq',cross,solve(z).reshape(len(x),160,3).transpose(1,0,2)))
            prior = cp.asnumpy(op.evaluate_batched(q,model.sample_prior_coeff,batch=4))
            exact_g = prior-exact_c
            component_errors = {'posterior': float(np.linalg.norm(g-exact_g)/np.linalg.norm(exact_g)),
                                'correction': float(np.linalg.norm(c-exact_c)/np.linalg.norm(exact_c))}
            beta = pilot_weights(g[:32],c[:32])
            estimates = estimate(g[32:],c[32:],v0,beta)
            metrics = {key:error_metrics(estimates[key],exact,mean) for key in ['posterior','correction','combined']}
            uncertainty = {'empty_intervals':int(estimates['empty_interval'].sum()),
                           'queries_covered':int(np.sum((estimates['lower']<=exact)&(exact<=estimates['upper']))),
                           'maximum_width_over_true_variance':float(np.max((estimates['upper']-estimates['lower'])/exact)),
                           'scope':'95% simultaneous component-intersection under exact iid Gaussian law; numerical bias excluded'}
            good = metrics['combined']
            quality = (max(component_errors.values())<.0005 and good['nonpositive_estimates']==0
                       and good['relative_variance_l2']<.01 and good['probability_rms']<.005
                       and good['probability_max']<.02)
            # This checks empirical independence only to its Monte Carlo noise.
            scale = np.sqrt(exact*reference_correction/128)
            cross_z = float(np.max(np.abs(estimates['posterior_correction_second_moment'])/scale))
            record['runs'].append({'seed':seed,'component_relative_l2':component_errors,
                'metrics':metrics,'intervals':uncertainty,'pass':quality,
                'maximum_component_cross_second_moment_standard_errors':cross_z,
                'pilot_posterior_weight_range':[float(beta.min()),float(beta.max())]})
            np.savez_compressed(args.output/f'seed_{seed}.npz',posterior_draws=g,correction_draws=c,**estimates)
            print('seed',seed,record['runs'][-1],flush=True)
        record['all_quality_gates_pass'] = all(v['pass'] for v in record['runs'])
        record['complete'] = True
    except Exception as exc:
        record['failure'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        record['gpu_process_audit'] = audit.stop()
        (args.output/'results.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__ == '__main__': main()
