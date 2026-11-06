"""Choose physical lengthscales on held-out geometry before solver timings."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from reproduction.noise_selection import AMPLITUDE,HALFWIDTH,SIZES,LENGTHS,NOISES,VALIDATION_SEEDS,TUNING_SEED,TEST_SEEDS,NOISE_REPLICATES
from reproduction.run_matched_crossover import invoke
from reproduction.run_matched_lengthscales import digest


def select(root,sizes=SIZES):
    root=Path(root);(root/'validation').mkdir(parents=True,exist_ok=True)
    protocol=dict(geometry='full ellipsoid axes (1,.8,.6), Gaussian noise on normal components; positions exact',
                  amplitude=AMPLITUDE,noise_standard_deviations=NOISES,likelihood_variance='sigma squared',
                  halfwidth=HALFWIDTH,sizes=sizes,lengthscale_grid=LENGTHS,upper_extensions=[.96,1.92],lower_extension=.0075,
                  validation_seeds=VALIDATION_SEEDS,noise_replicates_per_validation_geometry=NOISE_REPLICATES,
                  tuning_seed=TUNING_SEED,test_seeds=TEST_SEEDS,
                  validation_point_seed=6101,test_point_seed=6102,tuning_point_seed=6103,
                  selection='Minimum pooled squared surface-root error over validation scans; missing roots penalized by full box width',
                  queries=1000000,numerical_gates=dict(cdf=.001,variance_over_prior=.0001),
                  selection_solver='float64 dense through8192; H2 tol1e-7 with solve1e-10 above; direct mean-cache interpolation probes',
                  selection_is_mean_only=True,selection_and_tuning_excluded_from_production_cost=True,
                  locality_cutoff_candidates=[1,2,3,4,6,8,10,12,14,16,20],
                  locality_cutoff_grid_note='Added 1,2,3 after lengthscale validation, before solver tuning or test timings',
                  locality_storage='Test cached sparse if graph bytes < 0.6 device capacity; freeze faster passing storage on tuning scan',
                  largest_n_precision_check='Selected and runner-up, both geometry seeds; refine hierarchy until winner preserved and score change <1 percent; 1e-9 vs 1e-11 after initial 1.44 percent change',
                  additional_mean_geometry_check='Anchored mean error / minimum gradient on full-kernel probes <= 0.1 validation RMS; linearized diagnostic, unchanged CDF/variance gates',
                  hierarchy_tolerance_candidates=[1e-5,1e-6,1e-7,1e-8,1e-9],
                  hierarchy_extension_note='Added 1e-8 and 1e-9 after 32k tuning mean-error rejection, before test runs at 32k or 131k',
                  large_n_reference_mean_bound=1e-9,variance_iteration_candidates=[8,16,32],
                  largest_n_shared_precision=dict(cross_tolerance=1e-10,solve_tolerance=1e-10),
                  largest_n_precision_note='Tightened after a mean/cross precision floor on the tuning scan, before its test runs',
                  largest_n_test_seeds=TEST_SEEDS[:1],
                  locality_stop='After a three-scan 32k crossover, use the user-authorized competitor stopping rule; largest cases compare H2 with stronger full-kernel dense actions',
                  dense_action_control=dict(seeds=TEST_SEEDS[:1],cells=[[8192,.2],[32768,.2],[131072,.05],[131072,.2]]))
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    output=root/'selection.json';selected=json.loads(output.read_text()) if output.exists() else []
    for n in sizes:
        for sigma in NOISES:
            curves=[]
            def evaluate(ell):
                records=[]
                for seed in VALIDATION_SEEDS:
                    path=root/f'validation/n{n}-s{sigma:g}-ell{ell:g}-seed{seed}.json'
                    if not path.exists() and not path.with_suffix('.status.json').exists():
                        status=invoke([sys.executable,'-m','reproduction.noise_selection','--n',str(n),'--sigma',str(sigma),
                            '--ell',str(ell),'--seed',str(seed),'--output',str(path)],path.with_suffix('.log'))
                        path.with_suffix('.status.json').write_text(json.dumps(status,indent=2)+'\n')
                    if not path.exists():
                        control=path.with_name(path.stem+'-residual-control.json')
                        if not control.exists():
                            status=invoke([sys.executable,'-m','reproduction.noise_selection_control','--n',str(n),
                                '--sigma',str(sigma),'--ell',str(ell),'--seed',str(seed),'--output',str(control)],control.with_suffix('.log'))
                            control.with_suffix('.status.json').write_text(json.dumps(status,indent=2)+'\n')
                            if status.get('returncode')!=0:raise RuntimeError('Controlled selection fit failed '+str(control))
                        path=control
                    d=json.loads(path.read_text())
                    for source,checksum in d['source_sha256'].items():
                        if digest(source)!=checksum:raise ValueError('Stale selection source '+source)
                    records.append(dict(path=str(path),sha256=digest(path),
                        noisy_rms=float(np.sqrt(np.mean([m['penalized_surface_rms']**2 for m in d['noisy_metrics']]))),
                        replicate_rms=[m['penalized_surface_rms'] for m in d['noisy_metrics']],
                        clean_rms=d['clean_same_likelihood_metrics']['penalized_surface_rms'],
                        missing_roots=sum(m['missing_roots'] for m in d['noisy_metrics']),interpolation_error=d['interpolation_max_mean_error']))
                curves.append(dict(lengthscale=ell,records=records,
                    score=float(np.sqrt(np.mean([r['noisy_rms']**2 for r in records]))),
                    clean_score=float(np.sqrt(np.mean([r['clean_rms']**2 for r in records])))))
                print('validation',n,sigma,ell,curves[-1]['score'],flush=True)
            for ell in LENGTHS:evaluate(ell)
            for ell in [.96,1.92]:
                if min(curves,key=lambda r:r['score'])['lengthscale']!=max(r['lengthscale'] for r in curves):break
                evaluate(ell)
            if min(curves,key=lambda r:r['score'])['lengthscale']==min(LENGTHS):evaluate(.0075)
            best=min(curves,key=lambda r:r['score']);clean=min(curves,key=lambda r:r['clean_score'])
            row=dict(n=n,normal_noise=sigma,lengthscale=best['lengthscale'],validation_rms=best['score'],
                     clean_same_likelihood_best_lengthscale=clean['lengthscale'],curves=curves,
                     boundary_selected=best['lengthscale'] in [min(r['lengthscale'] for r in curves),max(r['lengthscale'] for r in curves)])
            selected=[r for r in selected if (r['n'],r['normal_noise'])!=(n,sigma)]+[row]
            output.write_text(json.dumps(selected,indent=2)+'\n');print('SELECTED',n,sigma,best['lengthscale'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='reproduction/results/noise_selected')
    p.add_argument('--sizes',type=int,nargs='+',default=SIZES);a=p.parse_args();select(a.root,a.sizes)
