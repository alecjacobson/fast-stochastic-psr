"""Report inverse-root validation and costs without a marginal/event speedup claim."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results/inverse_root');a=p.parse_args()
    out=a.results;raw={};summary={'scope':'matrix-free correction preparation pilot','cases':{}}
    rows=[];bounds=[]
    for name in ['A','million','three_million']:
        r=json.loads((out/name/'results.json').read_text());v=json.loads((out/name/'verification.json').read_text())
        assert r['complete'] and v['complete']
        assert hashlib.sha256((out/name/'results.json').read_bytes()).hexdigest()==v['results_sha256']
        assert hashlib.sha256((out/name/'fields.npz').read_bytes()).hexdigest()==v['fields_sha256']
        raw[name]=r
        polynomial=[d['draw_through_query_seconds'] for d in r['polynomial_draws']]
        pathwise=[d['draw_through_query_seconds'] for d in r['pathwise_draws']]
        polynomial_valid=all(d['quality_pass'] for d in r['polynomial_draws'])
        native_valid=all(d['quality_pass'] for d in r['pathwise_draws'])
        valid=polynomial_valid and native_valid
        isolated=r['gpu_process_audit']['sampled_isolation_pass']
        error=max(max(d['roundtrip_relative_errors']) for d in r['polynomial_draws'])
        memory=max((o['own_memory_bytes'] or 0) for o in r['gpu_process_audit']['observations'])/2**30
        row={'N':r['N'],'degree':r['polynomial']['degree'],'trace_degree':r['trace_control']['degree'],
             'polynomial_draw_seconds':polynomial,'pathwise_draw_seconds':pathwise,
             'max_roundtrip_relative_error':error,'polynomial_quality_pass':polynomial_valid,
             'pathwise_native_pass':native_valid,'quality_pass':valid,'sampled_isolation':isolated,
             'sampled_process_peak_GiB':memory,'timing_ratio_eligible':valid and isolated}
        if valid and isolated:row['pathwise_over_polynomial_median_draw_ratio']=float(np.median(pathwise)/np.median(polynomial))
        summary['cases'][name]=row
        rows.append(f'| {name} | {r["N"]:,} | {r["polynomial"]["degree"]} | '
                    f'{min(polynomial):.3f}–{max(polynomial):.3f} | {min(pathwise):.3f}–{max(pathwise):.3f} | '
                    f'{100*error:.6f}% | {"pass" if polynomial_valid else "FAIL"} | {"pass" if native_valid else "FAIL"} |')
        bounds.append(f'| {name} | {r["trace_control"]["upper"]:.6g} | {r["spatial_bound"]["upper"]:.6g} | '
                      f'{r["trace_control"]["degree"]} → {r["polynomial"]["degree"]} | '
                      f'{r["seconds"]["polynomial_setup"]:.3f} | {r["seconds"]["shared_operator_plans"]:.3f} | {memory:.2f} |')
    followup_rows=[];followups={}
    for n,case in [(1000000,'million'),(3000000,'three_million')]:
        native=json.loads((out/f'residual-{n}'/'results.json').read_text())
        mixed=json.loads((out/f'mixed-{n}'/'results.json').read_text())
        assert native['complete'] and mixed['complete']
        assert native['source_sha256']==mixed['source_sha256']==raw[case]['source_sha256']
        paired=[]
        for d,m in zip(native['draws'],mixed['draws']):
            assert d['rhs_sha256']==m['rhs_sha256'] and d['draw']==m['draw']
            after_native=max(d['attempts'][-1]['double_relative_residual'])
            before=max(m['refinement']['relative_residual_history'][0])
            after=max(m['refinement']['relative_residual_history'][-1])
            assert (after<=2e-5)==m['quality_pass']
            paired.append({'draw':m['draw'],'rhs_sha256':m['rhs_sha256'],
                          'double_residual_after_native_restarts':after_native,
                          'double_residual_before_mixed':before,'double_residual_after_mixed':after,
                          'mixed_corrections':m['refinement']['corrections'],
                          'mixed_solve_and_verify_seconds':m['solve_and_verify_seconds'],'mixed_pass':m['quality_pass']})
            followup_rows.append(f'| {n:,} | {m["draw"]} | {before:.3g} | {after_native:.3g} | '
                                 f'{after:.3g} | {m["refinement"]["corrections"]} | {m["solve_and_verify_seconds"]:.3f} |')
        followups[n]=(native,mixed)
        summary.setdefault('residual_followup',{})[str(n)]=paired
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    names=list(raw);x=np.arange(3)
    for shift,key,label in [(-.17,'polynomial_draw_seconds','Fixed polynomial'),(.17,'pathwise_draw_seconds','Joint-prior + CG')]:
        values=[summary['cases'][n][key] for n in names]
        bars=axes[0].bar(x+shift,[np.median(v) for v in values],width=.32,label=label)
        if key=='pathwise_draw_seconds':
            for name,bar in zip(names,bars):
                if not summary['cases'][name]['pathwise_native_pass']:bar.set_hatch('///')
        for position,v in zip(x+shift,values):axes[0].plot([position]*len(v),v,'k.',ms=4)
    axes[0].set(yscale='log',ylabel='Correction draw + 128 queries (s)',xticks=x,xticklabels=['A (691)','1M prefix','3M'])
    axes[0].legend(fontsize=8);axes[0].grid(axis='y',alpha=.2)
    axes[1].bar(x-.17,[raw[n]['trace_control']['degree'] for n in names],width=.32,label='Trace interval')
    axes[1].bar(x+.17,[raw[n]['polynomial']['degree'] for n in names],width=.32,label='Spatial-box interval')
    axes[1].set(yscale='log',ylabel='Fixed polynomial degree',xticks=x,xticklabels=['A (691)','1M prefix','3M'])
    axes[1].legend(fontsize=8);axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Hatched controls fail the native residual gate; no application-speedup claim',fontsize=10)
    fig.savefig(out/'preparation.png',dpi=180);fig.savefig(out/'preparation.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for ax,(n,(native,mixed)) in zip(axes,followups.items()):
        for d,m,color in zip(native['draws'],mixed['draws'],['C0','C1']):
            ax.semilogy(range(len(d['attempts'])),[max(v['double_relative_residual']) for v in d['attempts']],
                        '--o',color=color,label=f'Draw {d["draw"]}: native restart')
            history=m['refinement']['relative_residual_history']
            ax.semilogy(range(len(history)),[max(v) for v in history],'-s',color=color,label=f'Draw {d["draw"]}: double defect')
        ax.axhline(2e-5,color='black',ls=':',label='Acceptance threshold')
        ax.set(xlabel='Correction rounds',ylabel='Max independently checked relative residual',title=f'{n:,} observations',xticks=[0,1,2])
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.savefig(out/'refinement.png',dpi=180);fig.savefig(out/'refinement.pdf');plt.close(fig)
    dense=raw['A']['polynomial_draws']
    worst_weight=max(d['dense_weights_relative_error'] for d in dense)
    worst_field=max(d['dense_fields_relative_error'] for d in dense)
    all_pass=all(s['polynomial_quality_pass'] for s in summary['cases'].values())
    all_isolated=all(s['sampled_isolation'] for s in summary['cases'].values())
    slower=[name for name,s in summary['cases'].items() if np.median(s['polynomial_draw_seconds'])>np.median(s['pathwise_draw_seconds'])]
    labels={'A':'released A','million':'the million-point prefix','three_million':'the three-million-point input'}
    cost_sentence=('Recorded polynomial draw costs exceed the initial joint-prior/CG control for '+', '.join(labels[n] for n in slower)+'.'
                   if slower else 'The numerical and cost tables below retain both measured preparations.')
    (out/'README.md').write_text(f'''# Matrix-free fixed-polynomial correction sampling

A fixed Chebyshev inverse-root correction sampler is implemented and exercised
through three million observations. The spatial-box spectral envelope keeps
the polynomial degree manageable. This is a control for future rational/Krylov
sampling, not a replacement for coherent posterior samples or a demonstrated
marginal-variance/application speedup.

{cost_sentence} Failed controls are not matched-quality timing baselines.
Two draws per method are insufficient for robust timing statistics.
The current evidence does not support replacing the existing
pathwise sampler with this polynomial control.

[Protocol](PROTOCOL.md) · [Summary](summary.json) ·
[Issue #35](https://github.com/alecjacobson/fast-stochastic-psr/issues/35).

## Measured correction preparation

Both methods target correction covariance C A^-1 Cᵀ for the **same finite
training cube 200**, scalar/cross cube 50 and noise variance .02. All selected
observations enter each covariance action. Each correction draw contains three
independent Gaussian normal components. There are two draws per method and
size, with batch1; the table retains their ranges, not confidence intervals.

| Input | N | Degree | Polynomial draw + query (s) | Joint-prior + CG draw + query (s) | Worst round-trip error | Polynomial checks | Native CG checks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

The million-point input is the **first million points of the published
three-million-point source**, not the earlier million-point scan. Source hashes
and prefix provenance are retained. A uses the released 691 observations.

![Preparation costs and polynomial degrees](preparation.png)

Dots on the timing bars show both measurements; hatching marks failed native
CG controls. Their costs are **not matched-quality baseline timings**.
Shared operator/plan setup,
polynomial setup and validation are excluded from these draw costs and reported
below or in raw records. The pathwise control includes full joint-prior
generation, noise, CG, source aggregation and the same query evaluation. Its
existing routine also creates an unused scalar prior; it is not claimed to be
a minimal vector-only implementation. Different Gaussian couplings are compared
as distributions, not as identical individual realizations.

## Spectral envelope and fixed polynomial

| Input | Trace upper bound | Spatial-box upper bound | Trace → box degree | Polynomial setup (s) | Shared operators/plans (s) | Sampled process peak (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(bounds)}

The lower bound is the noise variance. Partition the periodic domain into
128-cubed boxes and count observations. The positive analytic circle Matérn
decreases with shortest angular distance, so minimum box-to-box distances give
entrywise upper envelopes. A separable FFT convolution computes box row sums;
their maximum over occupied boxes bounds the analytic spectral radius in exact
arithmetic. The finite analytically normalized Fourier kernel is below the
analytic kernel in PSD order, so it inherits the upper bound. This is a
**spectral bound**, not an FMM implementation or a change to the covariance.

For the monotonicity used here, set a=sqrt(3)/(2*ell) and t=a*(1-d/pi).
The derivative with respect to t has the sign of
a*coth(a)*sinh(t)-t*cosh(t), which is nonnegative on 0<=t<=a because
t*coth(t) is increasing. Thus the kernel decreases with shortest distance d.

The convolution and kernel formula use float64 and a relative 1e-8 pad. This is
an analytic enclosure evaluated numerically, **not directed-rounding interval
certification**. Neither the envelope nor the polynomial theory includes NUFFT
operator error. The source-independent trace bound is also recorded.

For f(t)=1/sqrt(c+h*t), select a Bernstein ellipse whose mapped leftmost real
part is positive. If that part is a, then |f|<=1/sqrt(a) on the ellipse. The
Chebyshev-Lobatto interpolation bound is 4 M rho^-degree/(rho-1); multiplying
by sqrt(upper) bounds relative inverse-root error. Degree is selected for a
bound <=1e-4. See [Trefethen, Theorem 8.2](https://people.maths.ox.ac.uk/trefethen/trefethen_sample.pdf).
Coefficient rounding, recurrence roundoff and approximate matrix actions are
outside this exact-arithmetic bound.

The interval, coefficients and degree are fixed before Gaussian draws. In exact
arithmetic, p(A) xi is consequently Gaussian with covariance p(A)^2. If the
relative scalar error is <=delta throughout the spectrum, then
(1-delta)^2 A^-1 <= p(A)^2 <= (1+delta)^2 A^-1 in PSD order. Congruence by C
gives the corresponding correction-covariance bounds. This conditional
statement is not a proved numerical covariance guarantee for the GPU code.

## Validation and limitations

All polynomial numerical checks pass: **{all_pass}**. On A, the dense symmetric
inverse root of the SAME finite kernel gives worst relative weight error
**{100*worst_weight:.6f}%** and correction-field error **{100*worst_field:.6f}%**.
Every polynomial draw also checks A p(A)^2 xi against xi, requiring relative
L2 <=.001 separately for each of its three RHS. These six random-direction
checks per size are **not** an operator norm, covariance or Monte Carlo accuracy
certificate. Dense reference and round-trip validation costs are separate.

The [15 targeted tests](tests.log) cover dense full covariance, fixed-map
linearity, periodic seams, duplicated points, short length scales, low noise,
float32 high-condition roundoff, streamed partial batches and correction-field
agreement. The [seven GPU/stream tests](stream-tests.log) were repeated after
factoring out shared aggregation. GPU fields replay from saved coefficients;
artifact hashes and accounting are checked in each verification record. The
large round-trip numbers are from the original runs, not a separate replay.

All whole-run sampled GPU isolation audits pass: **{all_isolated}**. Per-case
ratio eligibility is retained in JSON; no complete-application ratio follows.
The sampled memory peak covers the whole instrumented process, including
validation/baseline workspace of the initial pilot, and can miss brief peaks.
The separate double-verification follow-up has higher memory use. Sources, full Gaussian
RHS, weights and coefficients remain outside Git; compact query fields are
published in [A](A), [million](million) and [three million](three_million).

This does **not** resolve the [contrast-sensitive variance convergence failure](../correction_contrast/README.md).
Even instantaneous correction draws would retain Monte Carlo noise. Approximate
inverse roots also require an error allowance before applying intervals for the
exact target covariance. Correction fields must not be used as posterior draws
for ray/collision events. Training truncation, scalar bandwidth and full
application quality remain separate, open checks. Rational multi-shift CIQ is
still unimplemented; fixed-polynomial validation is a baseline for that work.

## Residual follow-up and mixed-precision correction

The original large-N native CG flags remain **failed**. A [separate follow-up](RESIDUAL_FOLLOWUP.md)
checks a float64 action of the same finite cube at epsilon1e-10, not an analytic
kernel with a different truncation. The first batched double plan failed with
CUFFT_INTERNAL_ERROR ([retained failure](residual-batched-failed)); checking one
RHS column at a time avoids that plan failure. Its exact cause is unestablished.

Fresh seeded RHS are archived for each large input, then held fixed between
native-restart and mixed-precision studies. Native restarts reduce the native
residual but one three-million-point draw still misses the independent
double-precision threshold after two restarts. Mixed-precision correction instead
forms the defect using the double action, solves that smaller defect with
native CG at rtol .1, accumulates in double, and stores the result in float32.
Only the independent residual of those stored weights controls acceptance.

| N | Draw | Initial double residual | After capped native restarts | After double-defect correction | Mixed corrections | Initial solve + mixed correction + checks (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(followup_rows)}

![Independent residuals under two correction strategies](refinement.png)

Both strategies use identical archived RHS, with fresh initial solves. Native
restarts and mixed correction have a cap of two. The acceptance threshold stays
2e-5. Table times include initial solving and independent checks, exclude common
operator setup and RHS generation, and do not include field aggregation/querying.
These runs are separate from the initial preparation pilot; do not combine their
times into an end-to-end or isolated speedup. The independent double action is
stronger numerical evidence, not exact arithmetic or a covariance/event guarantee.
The mixed-precision runs have sampled process peaks of
{max(o['own_memory_bytes'] or 0 for o in followups[1000000][1]['gpu_process_audit']['observations'])/2**30:.2f} GiB
and {max(o['own_memory_bytes'] or 0 for o in followups[3000000][1]['gpu_process_audit']['observations'])/2**30:.2f} GiB,
respectively. Sampled GPU isolation passes for both mixed-precision runs:
**{all(mixed['gpu_process_audit']['sampled_isolation_pass'] for _,mixed in followups.values())}**.
The initial pilot's per-case audits are recorded separately in the summary;
the preparation and mixed-solve tables remain different measurement scopes.

`geospsr.refinement.refine_solution` is an explicit opt-in helper; existing solver
defaults are unchanged. [19 targeted tests pass](all-tests.log), including a
biased native action, an independently constructed dense target and GPU storage
precision. Adoption in full posterior applications, preconditioned solves and
mixed-band configurations remains open in [issue #37](https://github.com/alecjacobson/fast-stochastic-psr/issues/37).

## Reproduce

```bash
./reproduction/run_env.sh -m reproduction.benchmark_inverse_root \\
  --case A --output /opt/geospsr-runtime/inverse-root/A
# Repeat with --case million and --case three_million in fresh serial processes.
./reproduction/run_env.sh -m reproduction.verify_inverse_root \\
  --run /opt/geospsr-runtime/inverse-root/A
# Verify each large run too; publish results/verification JSON and fields.npz.
./reproduction/run_env.sh -m reproduction.diagnose_inverse_root_residuals \\
  --count 1000000 --output /opt/geospsr-runtime/inverse-root/residual-1000000
./reproduction/run_env.sh -m reproduction.refine_inverse_root_residuals \\
  --source /opt/geospsr-runtime/inverse-root/residual-1000000 \\
  --output /opt/geospsr-runtime/inverse-root/mixed-1000000
# Repeat both for 3000000; publish the follow-up JSON before summarizing.
./reproduction/run_env.sh -m reproduction.summarize_inverse_root
```

Large cases require the regenerated source from `../large_inputs`. Normals are
not used for zero-mean correction sampling. This is preparation work, not a
mean fitting or posterior-conditioning benchmark.
''')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
