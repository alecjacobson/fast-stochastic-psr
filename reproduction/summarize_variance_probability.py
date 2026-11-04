"""Summarize measured probability workloads; never extrapolate censored SDD."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

ROOT=Path(__file__).resolve().parent/'results'/'variance_probability'
CANDIDATE=('cross_spectrum','analytic_matrix','cholesky','dense_mean_solve',
           'mean_aggregate','mean_center','variance_series_prepare','moment_grid_prepare')
DIRECT=('cholesky_float32','mean_solve_float32','mean_aggregate_float32','mean_center_float32')
COLORS={'cholesky_nufft_moment_grid':'tab:blue','cholesky_direct_float32':'tab:green',
        'cholesky_spatial_variance_grid':'tab:orange','sdd_direct':'tab:red'}
NAMES={'cholesky_nufft_moment_grid':'NUFFT variance field',
       'cholesky_direct_float32':'Direct Cholesky',
       'cholesky_spatial_variance_grid':'Quintic variance grid',
       'sdd_direct':'Direct SDD (censored)'}


def summarize(r):
    s=r['seconds'];prep={
        'cholesky_nufft_moment_grid':sum(s[k] for k in CANDIDATE),
        'cholesky_direct_float32':sum(s[k] for k in DIRECT)+(s['prepared_mean_float32'] if r['args'].get('shared_mean') else 0.)}
    if 'cholesky_spatial_variance_prepare' in s:
        prep['cholesky_spatial_variance_grid']=sum(s[k] for k in DIRECT)+s['prepared_mean_float32']+s['cholesky_spatial_variance_prepare']
    out={'n':r['n'],'complete':r['complete'],'sampled_gpu_isolation_pass':r['gpu_audit']['sampled_isolation_pass'],
         'preparation_seconds':prep,'model_only_table_seconds':s['reference_table'],
         'moment_cache_bytes':r['moment_cache_bytes'],'variance_series':r['variance_series'],
         'variance_grid':r.get('spatial_variance_grid')}
    out['measurement_quality_pass']={name:all(row[group][name]['pass'] for row in r['measurements']
        for group in ('quality','heldout_exact_quality') if name in row[group]) for name in {name for row in r['measurements'] for name in row['seconds']}}
    if 'replay' in r:
        out['replay']={}
        for name,m in r['replay']['methods'].items():
            out['replay'][name]={k:v for k,v in m.items() if k!='batches'}
            if name in prep:out['replay'][name]['total_seconds']=prep[name]+m['query_seconds']
            out['replay'][name]['max_heldout_errors']={key:max(b['heldout_exact_quality'][key] for b in m['batches'])
                for key in ('mean_relative_l2','variance_relative_l2','contrast_relative_l2','probability_rms','probability_max')}
            out['replay'][name]['max_full_probability_rms']=max(b['probability_rms_error'] for b in m['batches'])
            out['replay'][name]['max_full_probability_error']=max(b['probability_max_error'] for b in m['batches'])
        candidate=r['replay']['methods']['cholesky_nufft_moment_grid']
        direct=r['replay']['methods']['cholesky_direct_float32']
        candidate_total=prep['cholesky_nufft_moment_grid']+np.cumsum([b['seconds'] for b in candidate['batches']])
        direct_total=prep['cholesky_direct_float32']+np.cumsum([b['seconds'] for b in direct['batches']])
        crossing=np.flatnonzero(candidate_total<direct_total)
        out['first_measured_winning_batch']=int(crossing[0]+1) if len(crossing) else None
        out['variance_ablation']={label:{key:max(b[label][key] for b in r['replay']['variance_ablation'])
            for key in ('probability_rms_error','probability_max_error')}
            for label in ('prior','constant_posterior')}
    return out


def main():
    records={p.stem:json.loads(p.read_text()) for p in sorted((ROOT/'raw').glob('*.json'))}
    records={k:v for k,v in records.items() if v['complete']}
    result={name:summarize(r) for name,r in records.items()}
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    scale=[(name,r) for name,r in records.items() if name.startswith('quintic-B') or name=='shared-B16384']
    scale.sort(key=lambda x:x[1]['n'])
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for key,label in [('variance_series_prepare','Fourier correction construction'),('cholesky_spatial_variance_prepare','Cholesky quintic variance grid')]:
        xs=np.array([r['n'] for _,r in scale]);ys=np.array([r['seconds'][key] for _,r in scale])
        axes[0].loglog(xs,ys,'o-',label=label)
        if len(xs)>1:
            slope=float(np.polyfit(np.log(xs),np.log(ys),1)[0])
            axes[0].text(xs[-1],ys[-1],f'  slope {slope:.2f}',fontsize=8)
    axes[0].set(xlabel='Observations N',ylabel='Measured construction seconds',title='Fixed bandwidth; accuracy-selected grids')
    axes[0].set_xticks(xs,labels=[f'{v:,}' for v in xs]);axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].legend(fontsize=8);axes[0].grid(True,which='both',alpha=.25)
    if 'shared-B16384' in records:
        r=records['shared-B16384'];summary=result['shared-B16384'];m=r['replay']['queries_per_batch']
        for name,item in r['replay']['methods'].items():
            if name not in summary['preparation_seconds']:continue
            y=summary['preparation_seconds'][name]+np.cumsum([b['seconds'] for b in item['batches']])
            x=np.cumsum([b['queries'] for b in item['batches']])
            axes[1].loglog(x,y,'.-',label=NAMES[name],color=COLORS[name])
        sdd=r['replay']['methods'].get('sdd_direct')
        if sdd:
            axes[1].scatter(sdd['completed_queries'],sdd['query_seconds'],marker='^',color=COLORS['sdd_direct'],label='SDD completed prefix; query time only')
    axes[1].set(xlabel='Fresh query locations processed',ylabel='Preparation + actual query seconds',title='N=16,384; marginal probability and ambiguity')
    axes[1].legend(fontsize=8);axes[1].grid(True,which='both',alpha=.25)
    fig.savefig(ROOT/'variance-probability-scaling.png',dpi=180)
    plt.close(fig)
    write_report(records,result,scale)


def write_report(records,result,scale):
    if 'shared-B16384' not in records:return
    for name,_ in scale:
        assert result[name]['sampled_gpu_isolation_pass'] and all(result[name]['measurement_quality_pass'].values())
    r=records['shared-B16384'];s=result['shared-B16384'];methods=s['replay']
    candidate=methods['cholesky_nufft_moment_grid']
    assert s['sampled_gpu_isolation_pass'] and all(s['measurement_quality_pass'].values())
    assert all(v['quality_pass'] for v in methods.values())
    assert candidate['total_seconds']<min(methods[k]['total_seconds'] for k in ('cholesky_direct_float32','cholesky_spatial_variance_grid'))
    rows=[]
    for name in ('cholesky_nufft_moment_grid','cholesky_direct_float32','cholesky_spatial_variance_grid'):
        item=methods[name]
        rows.append(f"| {NAMES[name]} | {s['preparation_seconds'][name]:.3f} | {item['query_seconds']:.3f} | {item['total_seconds']:.3f} |")
    table='\n'.join(rows)
    scale_rows='\n'.join(f"| {v['n']:,} | {v['seconds']['variance_series_prepare']:.3f} | {v['seconds']['cholesky_spatial_variance_prepare']:.3f} | {v['spatial_variance_grid']['size']}³ |" for _,v in scale)
    xs=np.array([v['n'] for _,v in scale])
    slopes={k:float(np.polyfit(np.log(xs),np.log([v['seconds'][k] for _,v in scale]),1)[0]) for k in ('variance_series_prepare','cholesky_spatial_variance_prepare')}
    sdd=methods['sdd_direct'];ranking=candidate['ambiguity_ranking'];errors=candidate['max_heldout_errors']
    text=f"""# A measured variance-dependent reuse win

**A fixed map with many fresh moving-query batches is a winning scenario under
the declared moment-accuracy contract.** This is a synthetic planning/inspection
proxy on the released scorpion, not a ContactFusion replay. Each batch computes
mean, marginal variance, positive-field probability and an ambiguity score. Query
positions change; the prepared posterior field is reused. The probability is
Phi(mu/sqrt(v)) = P[f>0]; ambiguity is invariant to sign convention. A physical
inside/outside interpretation requires the chosen normal/field orientation.

The main workload has **16,384 observations and 32 × 65,536 = 2,097,152 queries**.
All completed methods pass both full-batch and independent exact-double
held-out checks. GPU work ran sequentially on one 1g.24gb Blackwell MIG instance;
its sampled process-isolation audit passed.

| Method | Charged preparation (s) | Actual 32-batch query work (s) | Total (s) |
|---|---:|---:|---:|
{table}

The NUFFT hybrid is **{methods['cholesky_direct_float32']['total_seconds']/candidate['total_seconds']:.2f}× faster than direct Cholesky** and
**{methods['cholesky_spatial_variance_grid']['total_seconds']/candidate['total_seconds']:.2f}× faster than the tested quintic spatial cache** for the complete workload.
The first measured direct-Cholesky crossover is batch **{s['first_measured_winning_batch']}**.
These totals include variance and the probability/ambiguity calculation. Baselines
receive the analytic observation matrix and reusable cross-kernel table free;
the candidate pays its own complete numerical preparation. See
[accounting and protocol](PROTOCOL.md) for the exact stage sums. Pilot tuning,
validation and logging are excluded from pipeline times, retained in raw records.

Direct SDD, given the same accurate prepared mean, used **{sdd['query_seconds']:.3f} s**
for **{sdd['completed_queries']:,} queries**, with passing accuracy. It hit the
180-second query budget before completing even the first 65,536-query batch.
This is a censored lower bound, not an extrapolated full SDD runtime. The selected
configuration was batch {r['sdd_selected']['batch']}, step multiplier {r['sdd_selected']['eta']},
{r['sdd_selected']['iterations']} iterations from four recorded configurations.
At fixed minibatch 128 and step multiplier 5, the N=2048 and N=16384 pilots
need 768 and 1280 iterations respectively. The selected configurations both
use 768 iterations, increasing minibatch from 128 to 512. This is limited
evidence of iteration growth at a fixed batch, not a general scaling law.

## Variance and application output

Maximum independent-panel errors for the NUFFT result across the replay:
variance relative L2 **{100*errors['variance_relative_l2']:.5f}%**;
centered variance contrast relative L2 **{100*errors['contrast_relative_l2']:.4f}%**;
probability RMS **{100*errors['probability_rms']:.5f} percentage points**;
probability maximum **{100*errors['probability_max']:.5f} percentage points**.
The limits are 0.02%, 1%, 0.05 pp and 0.5 pp respectively; mean relative L2 must
also be <=0.2%, and all variances must be positive. Full-query comparisons also
pass; see [summary.json](summary.json) for every maximum and reference check.

NUFFT selects candidate **{ranking['selected_batch']}** (zero-based), versus reference
candidate **{ranking['reference_best_batch']}**. Its range-normalized ambiguity-score
regret is **{ranking['range_normalized_regret']:.6g}**. Replacing variance by the prior
causes probability errors up to **{100*s['variance_ablation']['prior']['probability_max_error']:.2f} pp**;
even using the batch's exact average posterior variance causes errors up to
**{100*s['variance_ablation']['constant_posterior']['probability_max_error']:.2f} pp**.
This workload actually uses spatially varying variance. These are marginal
probabilities/scores, not joint collision or ray probabilities.

## Scaling and the methodological change

| Observations | Fourier variance construction (s) | Cholesky variance-grid construction (s) | Selected quintic grid |
|---:|---:|---:|---:|
{scale_rows}

The log-log fitted construction slopes over these measurements are
**{slopes['variance_series_prepare']:.2f}** for Fourier and **{slopes['cholesky_spatial_variance_prepare']:.2f}** for the spatial grid.
These are local empirical slopes, with an accuracy-selected grid at each N;
not asymptotic proofs or confidence intervals. Each is a single cold construction;
query timing/quality uses multiple changing geometries. The independent N=8192
paired-source pilot also measured 46.115 s, close to the 46.045 s repeated result.

![Measured construction and cumulative workload scaling](variance-probability-scaling.png)

The full correction factor is collapsed into one doubled-bandwidth Fourier
variance field. The gradient-product identity, packed real transforms and
batched inverse-factor solves reduce construction overhead. No Monte Carlo
sampling or rank truncation is used; mixed-precision transform error is measured.
In the original paper's N/M/L notation, the extra construction work is
O(N² + N L log L), versus O(H N²) for H direct spatial-grid sites; prepared
queries are O(M) for both. **Dense O(N³) conditioning remains.** This is a
Cholesky–NUFFT hybrid, not a matrix-free variance solver or an asymptotic M-only
win over an already-prepared spatial cache. [Derivation and costs](METHOD.md).

## Limits and retained failures

The result depends on the specified variance-accuracy contract and broad query
domain. Smaller spatial grids pass the probability gates alone while failing
the stricter variance gates. Adaptive/local grids and a probability-only accuracy
frontier remain [issue #66](https://github.com/alecjacobson/fast-stochastic-psr/issues/66).
The field caches must be rebuilt after covariance-changing map updates. Fixed
query coordinates can reuse values directly. The NUFFT retained mean/variance
cache is {s['moment_cache_bytes']/2**20:.1f} MiB, versus
{(s['variance_grid']['cache_bytes']+s['variance_grid']['mean_cache_bytes'])/2**20:.1f} MiB for the tested spatial/mean caches;
this is retained cache storage, not peak memory.

The small-A gradient pilot did not win against its spatial cache. An early N=8192 cubic
160³ grid failed held-out probability checks. The first N=16384 replay was stopped
because the interpolated reference mean failed the full-batch maximum probability
limit despite passing smaller panels. The corrected comparison gives all direct
variance baselines the same accurate prepared mean. These unsuccessful records
are retained in [raw/](raw/), including failed-reference-B16384.json.

[30 targeted tests](tests.log) passed: independent Fourier derivative sums,
partial and paired source batches, double/single arithmetic against dense
variance, and cubic/quintic interpolation against SciPy. Reproduce with
[reproduce.sh](reproduce.sh); raw JSON stores exact arguments, hashes, timings,
accuracy panels and GPU audit. Regenerate this summary and plot with
`./reproduction/run_env.sh -m reproduction.summarize_variance_probability`.
"""
    (ROOT/'README.md').write_text(text)


if __name__=='__main__':main()
