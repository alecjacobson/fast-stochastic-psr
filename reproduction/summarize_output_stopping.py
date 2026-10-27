"""Select calibration endpoints without choosing iterations on held-out quality."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def selected_checkpoint(record, target):
    for checkpoint in record['checkpoints']:
        if checkpoint['calibration_quality'][target]:
            return {'iteration': checkpoint['iteration'],
                    'holdout_confirmed': checkpoint['holdout_quality'][target],
                    'calibration_path_seconds': checkpoint['calibration_path_seconds'],
                    'with_selected_confirmation_seconds': checkpoint['with_selected_confirmation_seconds'],
                    'with_saved_oracle_preparation_seconds': checkpoint['with_saved_oracle_preparation_seconds']}
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, default=Path(__file__).resolve().parent/'results/output_stopping')
    a = p.parse_args(); out = a.results
    names = ['half200', 'sdd0', 'sdd1']; raw = {}
    for name in names:
        directory = out/name
        r = json.loads((directory/'results.json').read_text())
        v = json.loads((directory/'verification.json').read_text())
        assert r['complete'] and v['complete']
        assert v['results_sha256'] == hashlib.sha256((directory/'results.json').read_bytes()).hexdigest()
        assert v['contours_sha256'] == hashlib.sha256((directory/'contours.npz').read_bytes()).hexdigest()
        raw[name] = r
    assert all(r['source_hashes'] == raw['half200']['source_hashes'] and
               r['reference_weights_sha256'] == raw['half200']['reference_weights_sha256'] and
               r['grids'] == raw['half200']['grids'] and r['heights'] == raw['half200']['heights'] and
               r['boundary_targets'] == raw['half200']['boundary_targets'] and
               r['mean_gate'] == raw['half200']['mean_gate'] for r in raw.values())
    # Deterministic reference aggregation may differ by roundoff across processes;
    # raw field hashes are retained per run instead of assuming bitwise equality.
    result = {'scope': 'oracle checkpoint accuracy only', 'targets': {},
              'sampled_isolation': {n:r['gpu_process_audit']['sampled_isolation_pass'] for n,r in raw.items()}}
    table = []
    for target in ['coarse', 'strict']:
        selected = {n: selected_checkpoint(r, target) for n,r in raw.items()}
        eligible = all(d is not None and d['holdout_confirmed'] for d in selected.values()) and all(result['sampled_isolation'].values())
        result['targets'][target] = {'selected': selected, 'timing_ratio_eligible': eligible}
        if eligible:
            result['targets'][target]['calibration_path_ratios'] = {
                n:selected[n]['calibration_path_seconds']/selected['half200']['calibration_path_seconds'] for n in ['sdd0','sdd1']}
        for name, d in selected.items():
            if d is None:
                table.append(f'| {target} | {name} | none | — | — | — |')
            else:
                table.append(f'| {target} | {name} | {d["iteration"]:,} | '
                             f'{d["calibration_path_seconds"]:.3f} | {d["with_selected_confirmation_seconds"]:.3f} | '
                             f'{"pass" if d["holdout_confirmed"] else "FAIL"} |')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for name,r in raw.items():
        checkpoints=r['checkpoints']; x=[c['calibration_path_seconds'] for c in checkpoints]
        for column,ids in enumerate([[0,1],[2,3]]):
            error=[max(c['planes'][str(h)]['distance']['hausdorff_upper'] if c['planes'][str(h)]['distance'] else float('inf') for h in ids) for c in checkpoints]
            axes[column].loglog(x,100*np.array(error)/np.pi,marker='o',ms=3,label=name)
    for column,ax in enumerate(axes):
        ax.axhline(.1,color='.4',ls='--',label='coarse boundary target')
        ax.axhline(.01,color='.4',ls=':',label='strict boundary target')
        ax.set(xlabel='Cumulative calibration-path time (s)',ylabel='Max contour-distance upper / domain width (%)',
               title=['Calibration planes','Held-out planes'][column]);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(out/'accuracy.png',dpi=180);fig.savefig(out/'accuracy.pdf');plt.close(fig)
    runrows=[]
    for name,r in raw.items():
        last=r['checkpoints'][-1]
        runrows.append(f'| {name} | {r["seconds"]["method_setup"]:.3f} | {r["seconds"]["updates"]:.3f} | '
                       f'{last["calibration_path_seconds"]:.3f} | {r["seconds"]["reference_prepare_verify_and_save"]:.3f} | '
                       f'{r["seconds"]["instrumented_work_wall"]:.3f} | {r["gpu_process_audit"]["sampled_isolation_pass"]} |')
    (out/'README.md').write_text(f'''# Three-million-point output-aware checkpoint study

This measures when fixed solver checkpoints meet numerical mean and 2D boundary
targets on new slice geometry. Checkpoint selection uses calibration planes;
held-out planes confirm the same selected endpoint. It is an **oracle study**
using a saved higher-bandwidth mean, not a deployable stopping rule or full
application-speedup demonstration.

[Protocol](PROTOCOL.md) · [Selections and eligibility](summary.json) ·
[Tracking issue #36](https://github.com/alecjacobson/fast-stochastic-psr/issues/36).

## First calibration pass, then held-out confirmation

Calibration planes are y=-.4 and -.7; held-out planes are y=-.5 and -.8. All
methods use the same three-million-point scan. The reference uses training
cutoff 250, while NUFFT-CG uses cutoff 200 and SDD uses the full analytic kernel.
Scalar/cross cutoff remains 50. Source and saved-reference hashes are verified.

Both targets require <=0.2% relative mean error on each 2049-square grid,
matching component/open-component counts, and stable counts between the 1025
and 2049 grids. Coarse/strict contour Hausdorff upper limits are 0.1%/0.01% of
domain width; both candidate and reference must also meet that respective
limit in their grid-refinement comparison. These are numerical benchmark
targets, not physically justified tolerances for a particular application.

| Target | Method | First calibration iteration | Calibration path (s) | With selected held-out check (s) | Held-out confirmation |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

`none` means no scheduled checkpoint passed both calibration planes. A failed
held-out result is retained at the selected endpoint; a later passing result
does not replace it. Every checkpoint remains in the raw records.

![Boundary accuracy at every scheduled checkpoint](accuracy.png)

The plot shows only the boundary-distance criterion; mean, component and
refinement checks also affect the table. Distances are between **extracted
polylines**, with sample-cover brackets, not bounds on continuous zero sets.
These new planes are held out from checkpoint selection, but share the same
scene; this is not independent-scene validation or a 3D surface guarantee.

## Costs and timing eligibility

| Method | Setup (s) | All solver updates (s) | Final calibration path (s) | Saved-reference preparation (s) | Instrumented wall (s) | Sampled isolated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(runrows)}

Calibration-path time includes setup, solver updates, aggregation, all-source
centering, and every calibration field/contour/metric check up to that endpoint.
Held-out checks, artifacts and saved-reference preparation are excluded and
reported separately. The selected held-out check is added in the first table.
GPU work is synchronized at timing boundaries. CPU contour/metric work is
included in calibration/confirmation costs. Process imports/CUDA initialization
and source loading/hash verification are excluded from these path costs.
Reference preparation precedes the measured solver path in each process, so
shared scalar-transform kernels and caches can already be warm. These are not
cold-start pipeline measurements. CPU verification of a completed run may
overlap later GPU runs on the shared host; timings remain instrumented diagnostics.

Saved-reference preparation includes verifying and aggregating its weights,
evaluating all four planes and storing its arrays. It **does not include fitting
the reference**: that fit is the earlier contended cutoff-250 solve, identified
in each record. Thus even the `with_saved_oracle_preparation_seconds` field
does not describe an end-to-end deployable stopping algorithm. A practical
criterion must avoid needing this fitted oracle, or count its full cost.

Timing ratio eligibility (both SDD seeds and NUFFT confirm; every whole-run
isolation audit passes): coarse **{result['targets']['coarse']['timing_ratio_eligible']}**;
strict **{result['targets']['strict']['timing_ratio_eligible']}**. Any eligible
ratios in JSON apply only to the stated calibration path, not total pipeline
cost or deployment. One run per configuration limits timing statistics.
No historical-machine ratio or claim of upstream intractability is made.

## Validation and remaining work

The production CG observer is checked against the same solve without an
observer, including intermediate/final iterates and convergence metadata.
Compiled SDD has pinned-upstream equivalence tests. Saved field hashes, mean
errors, contour metrics, all quality decisions and calibration-time accounting
are recomputed by `verify_output_stopping`; each run's verification record
binds its exact JSON and contour archive. Large fields/coefficients remain
outside Git; compact contours and metrics are published.

This does not establish posterior variance, correlated ray/collision events,
next-view decisions, ContactFusion semantics, or full analytic/scalar-bandwidth
convergence. Mean-only stopping cannot be transferred to those tasks without
their own accuracy measurements. Subsequent work should test an implementable
stopping rule and its failure modes, and repeat on independent scenes.

## Reproduce

Recreate the three-million-point source and saved cutoff-250 fit as described in
`../large_fit`. Run fresh processes sequentially:

```bash
./reproduction/run_env.sh -m reproduction.benchmark_output_stopping \\
  --source /opt/geospsr-runtime/large-cloud-3000000 \\
  --reference /opt/geospsr-runtime/large-fit/half250 \\
  --method half200 --output /opt/geospsr-runtime/output-stopping/half200
# Repeat with --method sdd --seed 0 and --seed 1 in separate outputs.
./reproduction/run_env.sh -m reproduction.verify_output_stopping \\
  --run /opt/geospsr-runtime/output-stopping/half200
# Verify each SDD output, too. Then publish results/verification JSON and contours.
./reproduction/run_env.sh -m reproduction.summarize_output_stopping
```
''')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
