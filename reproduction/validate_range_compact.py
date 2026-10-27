"""Replay compact/native actions and mean fits; optionally recheck residuals."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from reproduction.common_rhs_integrity import array_digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu-residuals',action='store_true');args=p.parse_args()
    root=Path(__file__).resolve().parents[1];base=root/'reproduction/results/range_compact';count=0
    for directory in [base,base/'native']:
        r=json.loads((directory/'results.json').read_text());assert r['complete']
        for f,h in r['source_sha256'].items():assert hashlib.sha256((root/f).read_bytes()).hexdigest()==h
        for case,v in r['cases'].items():
            old=root/f'reproduction/results/periodic_range/{case}/actions.npz'
            assert hashlib.sha256(old.read_bytes()).hexdigest()==v['reference_payload_sha256']
            control=np.load(old);saved=np.load(directory/f'{case}_actions.npz');w=control['weights'];ref=control['reference'];cov=ref-.02*w
            assert array_digest(w)==v['hashes']['weights']
            for method,m in v['methods'].items():
                y=saved[method];last=m['actions'][-1]
                e=[float(np.linalg.norm(y[:,j]-ref[:,j])/np.linalg.norm(ref[:,j])) for j in range(3)]
                ke=[float(np.linalg.norm(y[:,j]-.02*w[:,j]-cov[:,j])/np.linalg.norm(cov[:,j])) for j in range(3)]
                np.testing.assert_allclose(e,last['action_relative_l2_per_rhs'],rtol=1e-12,atol=1e-30)
                np.testing.assert_allclose(ke,last['covariance_relative_l2_per_rhs'],rtol=1e-12,atol=1e-30)
                assert max(e+ke)<=r['relative_l2_gate'] and last['quality_pass'];count+=1
    r=json.loads((base/'conditioning/results.json').read_text());assert r['complete']
    for f,h in r['source_sha256'].items():assert hashlib.sha256((root/f).read_bytes()).hexdigest()==h
    for label,v in r['cases'].items():
        saved=np.load(base/f'conditioning/{label}_fields.npz');ref=saved['reference_weights'];w=saved['range_weights']
        e=np.linalg.norm(w-ref,axis=0)/np.linalg.norm(ref,axis=0)
        last=v['methods']['range']['solves'][-1]
        np.testing.assert_allclose(e,last['weights_relative_l2_per_rhs'],rtol=1e-8)
        m=np.linalg.norm(saved['range_mean']-saved['reference_mean'])/np.linalg.norm(saved['reference_mean'])
        np.testing.assert_allclose(m,v['mean_relative_l2'],rtol=1e-8)
        assert max(e)<=r['gates']['weights_relative_l2'] and m<=r['gates']['mean_relative_l2'] and v['quality_pass']
        if args.gpu_residuals:
            import cupy as cp
            from geospsr.spectral_tail import analytic_covariance_rows64
            case=label[0];d=np.load(root/f'reproduction/results/matched/{case}/input.npz');x=d['points'];y=d['normals'].astype('d')
            for k,a in [('points',x),('normals',d['normals']),('query',d['query'])]:assert array_digest(a)==v['hashes'][k]
            wd=cp.asarray(w);rd=cp.asarray(ref);res=[];refres=[]
            for begin in range(0,len(x),1024):
                k=analytic_covariance_rows64(x[begin:begin+1024],x,v['lengthscale'],v['variance_per_axis'])
                res.append((k@wd+v['noise_variance']*wd[begin:begin+1024]).get()-y[begin:begin+1024])
                refres.append((k@rd+v['noise_variance']*rd[begin:begin+1024]).get()-y[begin:begin+1024])
            residual=np.linalg.norm(np.concatenate(res),axis=0)/np.linalg.norm(y,axis=0)
            assert max(residual)<=r['gates']['residual']
            assert max(np.linalg.norm(np.concatenate(refres),axis=0)/np.linalg.norm(y,axis=0))<=r['gates']['residual']
            print(label,'independent streamed residual maximum',max(residual))
            del k,wd,rd;cp.get_default_memory_pool().free_all_blocks()
    for scale in ['scale100k','scale1m']:
        folder=base/scale;r=json.loads((folder/'results.json').read_text());assert r['complete']
        for f,h in r['source_sha256'].items():assert hashlib.sha256((root/f).read_bytes()).hexdigest()==h
        a=np.load(folder/'control.npz');x=a['points'];rows=a['rows'];ref=a['reference'];actual=a['actual']
        w=np.random.default_rng(r['seed']).normal(size=(len(x),3))
        for key,value in [('points',x),('rows',rows),('weights',w)]:assert array_digest(value)==r['hashes'][key]
        np.testing.assert_array_equal(rows,np.sort(np.random.default_rng(r['query_seed']).choice(len(x),r['query_rows'],replace=False)))
        e=np.linalg.norm(actual-ref,axis=0)/np.linalg.norm(ref,axis=0)
        cov=ref-.02*w[rows];ke=np.linalg.norm(actual-.02*w[rows]-cov,axis=0)/np.linalg.norm(cov,axis=0)
        np.testing.assert_allclose(e,r['actions'][-1]['action_relative_l2_per_rhs'],rtol=1e-12,atol=1e-30)
        np.testing.assert_allclose(ke,r['actions'][-1]['covariance_relative_l2_per_rhs'],rtol=1e-12,atol=1e-30)
        assert max(e.max(),ke.max())<=r['relative_l2_gate']
        if args.gpu_residuals:
            import cupy as cp
            from geospsr.spectral_tail import analytic_covariance_rows64
            wd=cp.asarray(w);control=[]
            for b in range(0,len(rows),64):
                control.append((analytic_covariance_rows64(x[rows[b:b+64]],x)@wd+.02*wd[rows[b:b+64]]).get())
            control=np.concatenate(control)
            np.testing.assert_allclose(control,ref,rtol=1e-12,atol=1e-14)
            del wd;cp.get_default_memory_pool().free_all_blocks()
        print(scale,'saved sampled-row action and reference pass')
    print(f'Replayed {count} saved A/B actions, 3 mean fits and 2 sampled scale controls.')


if __name__=='__main__':main()
