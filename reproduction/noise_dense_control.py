"""Full-kernel dense-action control when tuned locality retains every point.

Use the same prefix-kd AFN and independent mean/residual-corrected variance
solver as H2/locality. Only the float64 matrix representation changes: cached
dense when it fits, otherwise bounded dense row blocks. No core source patch.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys

from reproduction.run_matched_crossover import invoke
from reproduction.run_matched_lengthscales import flags,digest


def worker():
    import geospsr.nonperiodic_dense_cg as dense
    from geospsr.hierarchical_gpu import GPUHierarchicalSystem
    original=dense.HierarchicalMaternSystem

    def prefix(*args,**kwargs):
        kwargs['preconditioner_backend']='prefix-kd'
        return original(*args,**kwargs)

    def project(self,basis,vectors,variance_iterations=8,mean_tolerance=1e-8,variance_refinements=0):
        if variance_refinements:
            raise ValueError('Float64 matched control uses the shared variance solver without mixed-precision correction')
        self.rhs_block_size=72
        return GPUHierarchicalSystem.project(self,basis,vectors,variance_iterations,mean_tolerance)

    dense.HierarchicalMaternSystem=prefix
    dense.GPUDenseCGSystem.project=project
    sys.argv.remove('--worker')
    out=Path(sys.argv[sys.argv.index('--output')+1])
    runpy.run_module('reproduction.benchmark_hierarchical_variance',run_name='__main__')
    d=json.loads(out.read_text());d['source_sha256'][__file__]=digest(__file__)
    d['control']='Float64 dense actions; prefix-kd AFN; shared independent mean and residual-corrected variance PCG; cache or stream dense rows'
    out.write_text(json.dumps(d,indent=2)+'\n')


def run(root):
    from reproduction.run_noise_comparison import mean_geometry_check,compare
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());rows=[]
    selection=json.loads((root/'selection.json').read_text())
    for n,sigma in [(8192,.2),(32768,.2),(131072,.05),(131072,.2)]:
        cell=next(c for c in selection if c['n']==n and c['normal_noise']==sigma)
        candidates=[r for r in manifest['rows'] if r['n']==n and r['normal_noise']==sigma and r['seed']==5101]
        local=next((r for r in candidates if r['family']=='locality'),None)
        baseline=next(r for r in candidates if r['family']=='h2')
        local_record=json.loads(Path((local or baseline)['path']).read_text())
        if local and local_record['sparse_diagnostics']['nnz']!=n*n:
            raise ValueError('Dense fallback control assumes the selected radius retained all pairs')
        out=root/f'dense_control/n{n}-s{sigma:g}-seed5101.json';out.parent.mkdir(exist_ok=True)
        if not out.with_suffix('.status.json').exists():
            a=local_record['arguments'].copy();a.update(method='cg-dense-float64',variance_refinements=0,output=str(out))
            status=invoke([sys.executable,'-m','reproduction.noise_dense_control','--worker',*flags(a)],out.with_suffix('.log'))
            status['control_source_sha256']=digest(__file__)
            out.with_suffix('.status.json').write_text(json.dumps(status,indent=2)+'\n')
            print('dense control',n,status,flush=True)
        status=json.loads(out.with_suffix('.status.json').read_text())
        if status['control_source_sha256']!=digest(__file__):raise ValueError('Changed dense control source')
        row=dict(n=n,normal_noise=sigma,seed=5101,path=str(out),passes=False)
        if out.exists():
            d=json.loads(out.read_text())
            for source,checksum in d['source_sha256'].items():
                if digest(source)!=checksum:raise ValueError('Changed dense control dependency '+source)
            if 'independent_audit' not in d:
                reference=local_record['independent_audit']['reference']
                bounds=['--reference-mean-bound','1e-9'] if n>16384 else []
                status=invoke([sys.executable,'-m','reproduction.audit_hierarchical_variance',str(out),
                    '--reference',reference,'--random-queries','8','--reference-preconditioner','prefix-kd',*bounds],out.with_suffix('.audit.log'))
                if status.get('returncode')!=0:raise RuntimeError('Dense control audit failed')
                d=json.loads(out.read_text());d['mean_geometry_check']=mean_geometry_check(d,cell)
                out.write_text(json.dumps(d,indent=2)+'\n')
            check,quality=compare(root,out,Path(baseline['path']),'dense_control')
            row.update(sha256=digest(out),full_query_check=str(check),
                       passes=bool(quality['passes'] and d['mean_geometry_check']['passes']),
                       total_seconds=d['total_seconds'],matrix_cached=d['matrix_cached'])
            if not local:
                baseline.update(full_query_check=str(check),full_query_passes=quality['passes'],comparison_family='dense_action')
                (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        rows.append(row);(root/'dense_control.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    if '--worker' in sys.argv:worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='reproduction/results/noise_selected')
        run(p.parse_args().root)
