"""Report measured larger-N outcomes without inferring upstream infeasibility."""
import json
import hashlib
import re
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent/'results/large_fit'


def main():
    names=['half200','half250','sdd'];raw={n:json.loads((OUT/n/'results.json').read_text()) for n in names}
    source=json.loads((OUT.parent/'large_inputs/scorpion_3000000/manifest.json').read_text())
    fields={};rows=[];result={'scope':'three-million-point mean fitting only','runs':{}}
    for name,d in raw.items():
        if not d['complete']:
            result['runs'][name]={'complete':False,'failure':d.get('failure')}
            rows.append(f'| {name} | failed | — | — | — |');continue
        assert d['N']==3000000 and d['source_manifest']['arrays']==raw['half200']['source_manifest']['arrays']
        assert d['source_manifest']['arrays']==source['arrays']
        assert d['holdout_rows']==raw['half200']['holdout_rows'] and not set(d['holdout_rows'])&set(d['stopping_rows'])
        f=np.load(OUT/name/'fields.npz');fields[name]=f
        assert hashlib.sha256(f['query'].tobytes()).hexdigest()==d['query_sha256']
        error2=(f['residual'].astype('d')**2).sum(1);target2=(f['target'].astype('d')**2).sum(1)
        ids=np.random.default_rng(18).integers(0,len(error2),(2000,len(error2)))
        interval=np.quantile(np.sqrt(error2[ids].sum(1)/target2[ids].sum(1)),[.025,.975])
        np.testing.assert_allclose(interval,d['holdout_residual']['bootstrap_95_percent_interval'],rtol=1e-12)
        passed=bool(interval[1]<.001);assert passed==d['quality_pass']
        peaks=[v['own_memory_bytes'] for v in d['gpu_process_audit']['observations'] if v['own_memory_bytes'] is not None]
        peak=max(peaks)/2**30 if peaks else None
        info={'complete':True,'quality_pass':passed,'fit_through_mean_seconds':d['fit_through_mean_seconds'],
              'analytic_residual_interval':interval.tolist(),'sampled_process_peak_GiB':peak,
              'sampled_isolation_pass':d['gpu_process_audit']['sampled_isolation_pass'],'status':d['status']}
        result['runs'][name]=info
        rows.append(f'| {name} | {d["fit_through_mean_seconds"]:.3f} | '
                    f'{100*interval[0]:.4f}–{100*interval[1]:.4f}% | {"pass" if passed else "FAIL"} | {peak:.2f} |')
    paired={}
    if 'half250' in fields:
        reference=fields['half250']
        for name in ['half200','sdd']:
            if name in fields:
                np.testing.assert_array_equal(fields[name]['query'],reference['query'])
                paired[name]=float(np.linalg.norm(fields[name]['mean'].astype('d')-reference['mean'])/
                                   np.linalg.norm(reference['mean'].astype('d')))
    result['mean_relative_l2_to_half250']=paired
    result['paired_mean_gates']={name:v<=.002 for name,v in paired.items()}
    boundary_rows=[]
    if 'half250' in fields:
        # Same stationary scalar kernel (cutoff50, ell .01, variance .1).
        # This is a scale diagnostic, not the unknown posterior uncertainty.
        prior_std=float(np.sqrt(np.load(OUT.parent/'complement_variance/A/reference.npz')['prior_variance']))
        reference=fields['half250']['mean'].astype('d');near=np.abs(reference)<=2*prior_std
        result['posthoc_boundary_diagnostics']={'prior_std':prior_std,'query_count':int(near.sum()),'methods':{}}
        for name in ['half200','sdd']:
            if name not in fields:continue
            value=fields[name]['mean'].astype('d');error=value-reference
            diagnostic={'mean_contrast_relative_l2':float(np.linalg.norm(error)/np.linalg.norm(reference-reference.mean())),
                        'sign_disagreements':int(np.sum((value<=0)!=(reference<=0))),
                        'near_zero_rms_over_prior_std':float(np.sqrt(np.mean(error[near]**2))/prior_std),
                        'near_zero_max_over_prior_std':float(np.max(np.abs(error[near]))/prior_std)}
            result['posthoc_boundary_diagnostics']['methods'][name]=diagnostic
            boundary_rows.append(f'| {name} | {100*diagnostic["mean_contrast_relative_l2"]:.6f}% | '
                f'{diagnostic["sign_disagreements"]} | {diagnostic["near_zero_rms_over_prior_std"]:.6f} | '
                f'{diagnostic["near_zero_max_over_prior_std"]:.6f} |')
    eligible=(all(d.get('complete') and d.get('quality_pass') and d.get('sampled_isolation_pass') for d in result['runs'].values())
              and len(paired)==2 and all(v<=.002 for v in paired.values()))
    result['timing_ratio_eligible']=eligible
    if eligible:result['sdd_to_half200_fit_ratio']=result['runs']['sdd']['fit_through_mean_seconds']/result['runs']['half200']['fit_through_mean_seconds']
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    sdd=raw['sdd'];history=sdd.get('history',[])
    sdd_text=(f'SDD stopped with status `{sdd["status"]}` after {sdd.get("iterations",0):,} iterations and '
              f'{sdd["seconds"].get("sdd_updates",0):.3f}s of updates. '
              'Compilation, stopping-row setup/checks, aggregation, centering and querying are included in its table cost; '
              'the update budget excludes those costs and can overshoot by one 128-step chunk.')
    if history:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(6,3.3),layout='constrained')
        ax.semilogy([v['update_seconds'] for v in history],[100*v['analytic_residual']['relative_rms'] for v in history],label='SDD stopping rows')
        ax.axhline(.1,color='black',ls='--',label='0.1% gate')
        ax.set(xlabel='SDD update time (s)',ylabel='Sampled analytic residual (%)');ax.grid(alpha=.2);ax.legend()
        fig.savefig(OUT/'sdd-convergence.png',dpi=180);fig.savefig(OUT/'sdd-convergence.pdf');plt.close(fig)
        if len(fields)==3:
            reference=fields['half250']['mean'].reshape(2,100,100)
            arrays=[reference,fields['half200']['mean'].reshape(2,100,100)-reference,
                    fields['sdd']['mean'].reshape(2,100,100)-reference]
            titles=['Cutoff-250 mean','Cutoff 200 minus cutoff 250','Budgeted SDD minus cutoff 250']
            heights=fields['half250']['query'].reshape(2,100,100,3)[:,0,0,1]
            fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained',sharex=True,sharey=True)
            for j,(values,title) in enumerate(zip(arrays,titles)):
                scale=float(np.max(np.abs(values)))
                for i in range(2):
                    artist=axes[i,j].imshow(values[i],origin='lower',cmap='RdBu_r',vmin=-scale,vmax=scale,
                                            extent=[-np.pi/2,np.pi/2,-np.pi/2,np.pi/2])
                    axes[i,j].set_title(f'{title}\ny={heights[i]:.2f}',fontsize=9)
                    axes[i,j].set_xlabel('x')
                    if j==0:axes[i,j].set_ylabel('z')
                fig.colorbar(artist,ax=axes[:,j].tolist(),shrink=.8,pad=.02)
            fig.savefig(OUT/'mean-panels.png',dpi=180);fig.savefig(OUT/'mean-panels.pdf');plt.close(fig)
    pair_text='\n'.join(f'- {name} versus cutoff 250: **{100*v:.6f}%** relative L2 mean difference.' for name,v in paired.items())
    sdd_mean_text=('SDD **already passes the displayed-mean gate** in this run despite failing the\n'
        'training-residual gate. Its budget outcome must not be described as failure to\n'
        'reconstruct the mean at the declared query accuracy.' if paired.get('sdd',float('inf'))<=.002
        else 'SDD does not establish the displayed-mean gate in this run.')
    test_file=OUT/'tests.log'
    counts=re.findall(r'(\d+) passed',test_file.read_text()) if test_file.exists() else []
    test_text=(f'The **{counts[-1]} targeted tests pass** ([log](tests.log)), including compiled SDD equivalence to the pinned upstream loop and folded-spectrum covariance checks.'
               if counts else 'Targeted test result has not been recorded.')
    (OUT/'README.md').write_text(f'''# Three-million-point full mean fitting

The restored input now has **full mean fits**, beyond the earlier preconditioner-action
profile. Both complete finite training cubes (cutoffs 200 and 250) pass the
independent sampled analytic residual gate. This is mean-fit evidence, not
covariance, coherent-sampling or complete-application validation.

[Protocol](PROTOCOL.md) · [Summary](summary.json) ·
[Source provenance](../large_inputs/README.md).

## Measured quality and cost

All three million distinct observations enter each solve. The methods share
noise variance .02, length scale .01, scalar/cross cutoff 50, centering on all
observations and 20,000 fixed slice queries. NUFFT conjugacy folding preserves
every mode of the stated finite training cube. SDD uses the full analytic kernel.

| Method | Fit through mean queries (s) | Holdout analytic residual 95% bootstrap interval | Residual gate | Sampled process GPU peak (GiB) |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

The gate is an upper interval endpoint below .1%. The same 1,024 holdout rows
are independent of, and disjoint from, the 128 SDD stopping rows. Double
analytic covariance rows and matrix products are streamed in batches of 16.
The summarizer recomputes the bootstrap intervals from saved residuals/targets.
This empirical row-sampling interval is not a full-system norm bound.

{pair_text}

{sdd_mean_text} A fair mean/application
comparison should investigate earlier stopping based on the relevant output,
with independent confirmation; covariance and event tasks need their own checks.
[Issue #36](https://github.com/alecjacobson/fast-stochastic-psr/issues/36) tracks this distinction.

![Mean and signed difference panels](mean-panels.png)

The difference columns use separate color scales, shared between their two slices.
These show direct mean fields only and do not replace the matched mean/variance
teaser. The scalar bandwidth and query geometry remain fixed.

The prospective paired-mean gate is .2%. Mean agreement alone does not replace
the analytic residual gate or establish covariance/application agreement.

### Post-hoc near-surface diagnostics

The global mean gate can also conceal local errors near the zero level set.
After inspecting the fit, we checked the {int(near.sum())} queries where the
cutoff-250 mean lies within two scalar-prior standard deviations of zero.
That prior standard deviation is {prior_std:.8f} for the same stationary
cutoff-50 kernel; it is **not** the posterior uncertainty. These diagnostics
were not used to change the prospective gates or select a timing comparison.

| Method | Error / centered mean contrast | Sign differences over 20,000 queries | Near-zero RMS error / prior std | Near-zero max error / prior std |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(boundary_rows)}

The six SDD sign changes and larger near-zero errors motivate independent
surface/probability checks before treating its global mean agreement as
application equivalence. Pointwise signs on this grid are not a continuous
surface or topology certificate; these numbers do not measure probability error.

NUFFT uses a requested CG tolerance of 1e-5; the actual recomputed finite-system
residuals are retained in each raw record and can exceed that requested value
within the existing solver's floating-point acceptance criterion of twice the
requested tolerance. The actual finite residuals here are about 1.38–1.79e-5.

## The bounded SDD comparison

{sdd_text}

The configuration transfers the best selected million-point setting: batch 512,
eta 5, learning rate `5*512/N`, seed 0, momentum .9 and Polyak .01. It compiles
the released update algebra, with existing upstream-equivalence tests; it is
not the untouched Python loop or a new three-million-point tuning sweep.
Its noise parameter is interpreted consistently as variance .02.

![SDD residual at every declared checkpoint](sdd-convergence.png)

No timing ratio is eligible unless all three residual checks, both paired mean
checks and all whole-run isolation audits pass. For these records that combined
eligibility is **{eligible}**. GPU contention is retained in the process logs.
The measured budget outcome does **not** show that SDD is intractable: another
configuration or more update time could converge. It also does not establish
an upstream resource limit for posterior sampling or any downstream application.

## Timing and memory scope

Each method runs in a fresh process. Table costs include model/plan preparation,
solver work, centering and mean queries. Source loading/hash verification and
independent final validation are timed separately. SDD stopping machinery is
included in its fitting cost. Process imports/CUDA initialization are excluded;
on-disk compiler caches may already be warm. No historical-machine timing ratio
is used. There is only one fit per declared configuration, not repeated statistics.

GPU memory is sampled per process through nvidia-smi, including native allocations;
it can miss short-lived peaks and is not an exact peak-memory measurement.
Full source arrays and final weights live under `/opt/geospsr-runtime`, outside
Git. Their hashes are recorded; compact mean/residual panels are published.

## Reproduce

```bash
./reproduction/run_env.sh -m reproduction.prepare_large_cloud \\
  --points 3000000 --output /opt/geospsr-runtime/large-cloud-3000000
./reproduction/run_env.sh -m reproduction.benchmark_large_fit \\
  --source /opt/geospsr-runtime/large-cloud-3000000 --method half --cutoff 200 \\
  --output /opt/geospsr-runtime/large-fit/half200
```

Preparation refuses to overwrite an existing input. Run cutoff 250 in a separate
output/process, then `--method sdd --time-budget 600` in another. Copy results.json
and fields.npz from all three outputs into the corresponding report folders and
run `reproduction.summarize_large_fit`. Do not substitute the much larger saved
weights for the compact published field panels.

Next: larger-scale coherent sampling, covariance/tail checks, downstream
application quality, and a stronger or extended SDD confirmation if needed.
[Issue #8](https://github.com/alecjacobson/fast-stochastic-psr/issues/8) remains open.
{test_text}
''')
    print(json.dumps(result))


if __name__=='__main__':main()
