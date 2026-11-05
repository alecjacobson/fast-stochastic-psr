"""Record the source audit supporting reuse of the full-Eagle endpoints."""
import hashlib
import json
from pathlib import Path

from reproduction.run_fixed_free_space import TEMPLATES


def main():
    rows=[]
    for method,path in TEMPLATES.items():
        d=json.loads(path.read_text());changes=[]
        allowed={'geospsr/hierarchical_gpu.py':'Only optional progress callbacks added to project; linear algebra and kernels unchanged.',
                 'geospsr/nonperiodic_local_gpu.py':'Unused by H2; locality endpoint already uses current warp-parallel fill.',
                 'reproduction/benchmark_hierarchical_variance.py':'Metadata module lookup fix and progress logging only; timing boundaries and numerical calls unchanged.'}
        for name,sha in d['source_sha256'].items():
            current=hashlib.sha256(Path(name).read_bytes()).hexdigest()
            if current==sha:continue
            if name not in allowed or (name.endswith('nonperiodic_local_gpu.py') and method!='h2-streamed'):
                raise ValueError('Unaudited source change '+name)
            changes.append(dict(path=name,recorded_sha256=sha,current_sha256=current,review=allowed[name]))
        rows.append(dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),changes=changes,
            reviewed_against_commits=['0eab6dfeb7a145aa5813abacaee9270fef2f70fd','014d6f6947910965ec21697ec33a13591b396bc1'],
            scope='Numerical implementation and settings match; earlier endpoint omits progress printing. No kernel speed optimization is mixed into the fitted series.'))
    Path('reproduction/results/asymptotic_regimes/reuse_audit.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
