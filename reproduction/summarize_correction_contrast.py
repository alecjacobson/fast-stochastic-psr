"""Recheck full-slice marginal quality without conflating it with event quality."""
import json
import hashlib
import re
from pathlib import Path
import numpy as np
from scipy.special import ndtr

OUT=Path(__file__).resolve().parent/'results/correction_contrast'


def main():
    rows=[]; costs=[]; result={}; curves={}
    for case in ['A','B']:
        raw=json.loads((OUT/case/'results.json').read_text())
        assert raw['complete'] and raw['queries']==20000
        assert raw['seeds']==[401,503,607] and raw['checkpoints']==[128,512,2048,8192]
        data=np.load(OUT/case/'reference.npz'); exact,mean=data['variance'],data['mean']
        source=np.load(OUT.parent/'matched'/case/'input.npz')
        for key,name in [('points','points'),('normals','normals'),('queries','query')]:
            assert hashlib.sha256(source[name].tobytes()).hexdigest()==raw['hashes'][key]
        np.testing.assert_array_equal(data['query'],source['query'])
        truth=ndtr(-mean/np.sqrt(exact)); curves[case]=[]; checkpoints={n:[] for n in raw['checkpoints']}
        reference_contrast=np.linalg.norm(exact-exact.mean())
        deterministic=np.linalg.norm(data['deterministic']-exact)/reference_contrast
        np.testing.assert_allclose(deterministic,raw['deterministic_quality']['contrast_relative_l2'],rtol=1e-12)
        for run in raw['runs']:
            saved=np.load(OUT/case/f'seed_{run["seed"]}.npz'); series=[]
            for item in run['checkpoints']:
                value=saved[f'variance_{item["count"]}']
                error=np.linalg.norm(value-exact)/reference_contrast
                prob=ndtr(-mean/np.sqrt(np.maximum(value,np.finfo(float).tiny)))
                rms=float(np.sqrt(np.mean((prob-truth)**2))); maximum=float(np.max(np.abs(prob-truth)))
                np.testing.assert_allclose([error,rms,maximum],
                    [item['metrics'][k] for k in ['contrast_relative_l2','probability_rms','probability_max']],rtol=1e-12)
                passed=bool(error<=.01 and rms<=.0005 and maximum<=.005 and np.all(value>0))
                assert passed==item['quality_pass']
                checkpoints[item['count']].append(passed)
                series.append([item['count'],error,maximum])
            last=run['checkpoints'][-1]; m=last['metrics']
            rows.append(f'| {case} | {run["seed"]} | {100*m["contrast_relative_l2"]:.3f}% | '
                        f'{100*m["probability_rms"]:.4f} / {100*m["probability_max"]:.4f} | '
                        f'{"pass" if last["quality_pass"] else "FAIL"} |')
            curves[case].append(series)
        passing=[n for n,v in checkpoints.items() if all(v)]
        baseline=raw['deterministic_quality']
        baseline_pass=all(baseline[k]<=v for k,v in raw['gates'].items()) and baseline['nonpositive_count']==0
        common=raw['seconds']['fit']+raw['seconds']['mean_queries']
        duration=[r['checkpoints'][-1]['elapsed_seconds'] for r in raw['runs']]
        result[case]={'all_seed_passing_counts':passing,'baseline_quality_pass':baseline_pass,
                      'baseline_contrast_relative_l2':deterministic,
                      'sampled_isolation_pass':raw['gpu_process_audit']['sampled_isolation_pass'],
                      'common_fit_mean_seconds':common,'deterministic_seconds':raw['deterministic_median_seconds'],
                      'stochastic_8192_seconds_range':[min(duration),max(duration)],
                      'timing_ratio_eligible':bool(passing and baseline_pass and raw['gpu_process_audit']['sampled_isolation_pass'])}
        costs.append(f'| {case} | {common:.3f} | {raw["deterministic_median_seconds"]:.3f} | '
                     f'{min(duration):.3f}–{max(duration):.3f} | {100*deterministic:.6f}% |')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
    for case,color in [('A','#2266aa'),('B','#bb4422')]:
        values=np.array(curves[case]);counts=values[0,:,0]
        for ax,col in zip(axes,[1,2]):
            for seed in values:ax.plot(counts,100*seed[:,col],color=color,alpha=.25,lw=1)
            ax.plot(counts,100*np.median(values[:,:,col],axis=0),'-o',color=color,label=case)
            ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks(counts,labels=[str(int(n)) for n in counts])
            ax.set_xlabel('Independent correction draws');ax.grid(alpha=.2)
    axes[0].axhline(1,color='black',ls='--',label='1% gate');axes[0].set_ylabel('Error / spatial contrast (%)')
    axes[1].axhline(.5,color='black',ls='--',label='0.5 pp gate');axes[1].set_ylabel('Maximum inside-probability error (pp)')
    for ax in axes:ax.legend(fontsize=8)
    fig.savefig(OUT/'convergence.png',dpi=180);fig.savefig(OUT/'convergence.pdf');plt.close(fig)
    tests=OUT/'tests.log';matches=re.findall(r'(\d+) passed',tests.read_text()) if tests.exists() else []
    test_text=f'All **{matches[-1]} tests pass** ([log](tests.log)).' if matches else 'The two new correction-sampling tests pass; full regression result pending.'
    all_failed=all(not v['all_seed_passing_counts'] for v in result.values())
    headline=('**Neither A nor B reaches the 1% spatial-contrast gate at any tested count through 8,192 draws.**'
              if all_failed else 'At least one dataset reaches all declared marginal gates; see the per-case summary.')
    (OUT/'README.md').write_text(f'''# Correction sampling at matched spatial-contrast quality

{headline}
The deterministic streamed cross-factor baseline passes on both. No matched-quality
speedup follows from this study. [Protocol](PROTOCOL.md) · [Raw summary](summary.json).

## Method and scope

For `A = L Lᵀ`, independent Gaussian vectors `xi` give `alpha = L⁻ᵀ xi` with
covariance `A⁻¹`. Fourier aggregation and evaluation give correction fields
`c = C alpha` with covariance `C A⁻¹ Cᵀ`. Estimate marginal variance with
`V0 − mean(c²)`, using the known zero correction mean. This avoids constructing
the full Fourier prior. It does **not** produce posterior samples for correlated
ray or collision queries. The Cholesky factor still requires quadratic storage.

Source plans are reused within each stream; the fixed-query plan is shared
across batches. Only eight coefficient fields and query-count-sized running
second moments are required, apart from transient blocks and transform workspaces.
Retaining yielded blocks defeats this bound. Negative estimates are preserved.
No antithetic partners, pilot fitting or query points are counted as extra draws.

The experiment uses every one of the 20,000 slice queries in each committed A/B
input, full analytic training covariance, scalar/cross cutoff 50 and noise .02.
It evaluates the direct Fourier model, not the released interpolant. These
results do not replace the existing matched teaser images. Probabilities use a
shared fitted mean, so their error isolates the variance estimator.

## Fixed prospective quality checks

Three new seeds (401, 503, 607) each run through 128, 512, 2,048 and 8,192 draws.
All seeds and failures are retained. A pass requires variance error below 1%
of the reference's centered spatial contrast, inside-probability RMS below
.05 pp and maximum below .5 pp, with positive variance estimates everywhere.
These marginal gates do not establish downstream action rankings or event quality.
The reference uses dense float64 factorization and epsilon-1e-12 cross sums.

![Marginal quality across all fixed sample counts](convergence.png)

Thin curves are individual seeds; thick curves are medians, not confidence bands.
The contrast gate remains stricter than the probability gates in this experiment.

| Case | Seed | Contrast error at 8,192 | Probability RMS / max (pp) | Combined gate |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

All checkpoint estimates are retained in each `seed_*.npz`; the summarizer
recomputes their errors against the saved reference. Exact-law chi-square
interval diagnostics account for 20,000 queries and four fixed looks per seed.
They exclude model and numerical error and do not validate continuous geometry.

## Costs, including the stronger deterministic baseline

| Case | Common fit + mean queries (s) | Deterministic variance median (s) | Streamed 8,192 draws + queries + reduction (s) | Deterministic contrast error |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(costs)}

Add common fit/mean cost to either method for first-use comparison. Stochastic
query-plan setup is also recorded separately and must be added. The stochastic
elapsed column includes generation, triangular solves, source aggregation,
queries, transfers, reductions and checkpoint-statistics overhead. The baseline
includes cross-sum query construction and triangular factor application, in
three fresh evaluations. Exact reference construction is diagnostic work and
is excluded from both application costs but retained in the raw stage records.
The `load` timer opens the archive; array materialization and hashing appear
only in total instrumented wall time. Disk artifact writes are diagnostic.

Whole-run process isolation fails for these runs. Combined with failed quality
gates, this rules out a timing-ratio claim. The displayed costs are diagnostics,
not a comparison with historical-machine timings or upstream applications.

## What this changes

The earlier low total-variance error concealed a more demanding contrast
requirement. Faster sample generation alone does not solve that statistical
problem. Dense deterministic variance remains the stronger option here.
Probability-only tasks may need fewer draws, but require their own independent
end-to-end comparison, including mean error and task decisions.

[Issue #34](https://github.com/alecjacobson/fast-stochastic-psr/issues/34) remains
open for stronger sample-efficiency methods and application validation.
[Issue #35](https://github.com/alecjacobson/fast-stochastic-psr/issues/35) tracks
matrix-free inverse-root sampling to remove the dense factor. It is unimplemented;
even a faster inverse-root action would not eliminate Monte Carlo contrast noise.

## Reproduce

```bash
./reproduction/run_env.sh -m reproduction.benchmark_correction_contrast \\
  --case A --output /opt/geospsr-runtime/correction-contrast/A
./reproduction/run_env.sh -m reproduction.benchmark_correction_contrast \\
  --case B --output /opt/geospsr-runtime/correction-contrast/B
./reproduction/run_env.sh -m pytest -q tests
./reproduction/run_env.sh -m reproduction.summarize_correction_contrast
```

Copy benchmark outputs into the report's A/B folders before summarizing. The
runtime was restored from the pinned lock on this VM. Setup now recovers a
missing upstream worktree registration left by an ephemeral runtime disk.
Independent tests verify explicit Fourier sums, full correction covariance,
partial batches and bounded moment accumulation. {test_text}
''')
    print(json.dumps(result))


if __name__=='__main__':main()
