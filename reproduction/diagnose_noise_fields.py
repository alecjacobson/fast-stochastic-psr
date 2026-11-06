"""Full-query shape/variance diagnostics for independently audited unpaired fits.

These are single-field evaluations, not numerical accuracy comparisons. The
full-kernel probe audit remains the numerical evidence for an unpaired fit.
"""
import argparse
import json
from pathlib import Path

from reproduction.compare_hierarchical_application import compare
from reproduction.run_matched_lengthscales import digest


def run(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    for row in manifest['rows']:
        if row.get('full_query_check') or not row.get('sha256'):
            continue
        path=Path(row['path'])
        if digest(path)!=row['sha256']:raise ValueError('Changed production record')
        folder=root/'diagnostics';folder.mkdir(exist_ok=True)
        raw=folder/(path.stem+'.self-evaluation.json')
        if not raw.exists():compare(str(path),str(path),str(raw))
        d=json.loads(raw.read_text())
        if d['baseline_sha256']!=digest(path):raise ValueError('Stale field diagnostics')
        d['numerical_comparison_evidence']=False
        d['interpretation']='Single-field evaluation: self-difference metrics and the helper pass flag are not numerical accuracy evidence.'
        raw.write_text(json.dumps(d,indent=2)+'\n')
        result={key:d[key] for key in ['queries','shape_brier','shape_brier_scope',
            'max_error_if_prior_variance_is_used','max_error_if_constant_contact_variance_is_used',
            'max_constant_variance_error_over_prior']}
        result.update(source=str(path),source_sha256=digest(path),
            evaluation_kind='Single-field shape/variance diagnostics only; no opponent or numerical accuracy comparison',
            raw_evaluation=str(raw),raw_sha256=digest(raw))
        out=folder/(path.stem+'.json');out.write_text(json.dumps(result,indent=2)+'\n')
        row.update(full_query_diagnostics=str(out),full_query_diagnostics_sha256=digest(out),
                   full_query_comparison_available=False)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='reproduction/results/noise_selected')
    run(p.parse_args().root)
