"""Replay refinement fits; optionally regenerate direct and all-row audits."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from reproduction.common_rhs_integrity import array_digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu-audits',action='store_true');p.add_argument('--confirmation-archive',type=Path);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];base=root/'reproduction/results/analytic_refinement'
    r=json.loads((base/'results.json').read_text());assert r['complete'];count=0
    for path,h in r['source_sha256'].items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==h
    for label,v in r['cases'].items():
        data=np.load(root/f'reproduction/results/matched/{label[0]}/input.npz');x,y,q=[data[k] for k in ['points','normals','query']]
        for k,a in [('points',x),('normals',y),('query',q)]:assert array_digest(a)==v['hashes'][k]
        a=np.load(base/f'{label}_fields.npz');ref=a['reference_weights'];meanref=a['reference_mean']
        for name,m in v['methods'].items():
            w=a[name+'_weights'];mean=a[name+'_mean'];last=m['runs'][-1]
            e=np.linalg.norm(w.astype('d')-ref,axis=0)/np.linalg.norm(ref,axis=0)
            me=np.linalg.norm(mean-meanref)/np.linalg.norm(meanref)
            np.testing.assert_allclose(e,last['weights_relative_l2_per_rhs'],rtol=1e-8,atol=1e-30)
            np.testing.assert_allclose(me,m['mean_relative_l2'],rtol=1e-8)
            if args.gpu_audits:
                import cupy as cp
                from geospsr.spectral_tail import analytic_covariance_rows64
                wd=cp.asarray(w,dtype='d');res=[]
                for b in range(0,len(x),1024):
                    k=analytic_covariance_rows64(x[b:b+1024],x,v['lengthscale'],.1)
                    res.append((k@wd+v['noise_variance']*wd[b:b+1024]).get()-y[b:b+1024])
                residual=np.linalg.norm(np.concatenate(res),axis=0)/np.linalg.norm(y.astype('d'),axis=0)
                np.testing.assert_allclose(residual,last['independent_residual_per_rhs'],rtol=1e-5,atol=1e-14)
                del k,wd;cp.get_default_memory_pool().free_all_blocks()
            expected=all(s['fit_quality_pass'] for s in m['runs']) and me<=r['gates']['mean_relative_l2']
            assert expected==m['quality_pass'];assert m['quality_pass']==(name!='nufft');count+=1
    pilot=json.loads((base/'million/results.json').read_text());assert pilot['complete']
    preserved=np.load(base/'million/controls.npz')
    for name,m in pilot['methods'].items():
        me=np.linalg.norm(preserved[name+'_mean']-preserved['plain_mean'])/np.linalg.norm(preserved['plain_mean'])
        se=np.linalg.norm(preserved[name+'_sample_residual'],axis=0)/np.linalg.norm(preserved['normals'][preserved['rows']],axis=0)
        np.testing.assert_allclose(me,m['mean_relative_l2'],rtol=1e-7,atol=1e-30)
        np.testing.assert_allclose(se,m['independent_sampled_residual_per_rhs'],rtol=1e-12)
    print('Original million pilot: retained means/sample audits replay; full-weight archives lost in runtime reset.')
    folder=base/'confirmation';r=json.loads((folder/'results.json').read_text());assert r['complete']
    for path,h in r['source_sha256'].items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==h
    old=root/'reproduction/results/range_compact/scale1m/control.npz';assert hashlib.sha256(old.read_bytes()).hexdigest()==r['point_archive_sha256']
    x=np.load(old)['points'];data=np.load(folder/'controls.npz');y,q,rows=[data[k] for k in ['normals','query','rows']]
    for k,a in [('points',x),('normals',y),('query',q),('rows',rows)]:assert array_digest(a)==r['hashes'][k]
    assert hashlib.sha256((base/'million/results.json').read_bytes()).hexdigest()==r['pilot_results_sha256']
    assert hashlib.sha256((base/'million/controls.npz').read_bytes()).hexdigest()==r['pilot_controls_sha256']
    weights={name:np.load((args.confirmation_archive/f'{name}_weights.npy') if args.confirmation_archive else m['archive_path']) for name,m in r['methods'].items()}
    for name,m in r['methods'].items():
        w=weights[name];assert array_digest(w)==m['weights_sha256']
        np.testing.assert_array_equal(w[rows],data[name+'_sample_weights'])
        e=np.linalg.norm(w.astype('d')-weights['plain'],axis=0)/np.linalg.norm(weights['plain'],axis=0)
        me=np.linalg.norm(data[name+'_mean']-data['plain_mean'])/np.linalg.norm(data['plain_mean'])
        independent=np.linalg.norm(data[name+'_sample_residual'],axis=0)/np.linalg.norm(y[rows],axis=0)
        np.testing.assert_allclose(e,m['runs'][-1]['weights_relative_l2_per_rhs'],rtol=1e-7,atol=1e-30)
        np.testing.assert_allclose(me,m['mean_relative_l2'],rtol=1e-7,atol=1e-30)
        np.testing.assert_allclose(independent,m['independent_sampled_residual_per_rhs'],rtol=1e-12)
        expected=all(s['fit_quality_pass'] for s in m['runs']) and me<=r['gates']['mean_relative_l2'] and independent.max()<=r['gates']['residual']
        assert expected==m['quality_pass']
    if args.gpu_audits:
        import cupy as cp
        from geospsr.spectral_tail import analytic_covariance_rows64
        from geospsr.periodic_leaf_cache import LeafCachedRangeKernel
        model=LeafCachedRangeKernel(x,leaf_size=128,max_entries=200_000_000,max_cache_bytes=32*2**30)
        diagnostics={}
        for name,w in weights.items():
            wd=cp.asarray(w,dtype='d');res=model(wd).get()-y
            relative=np.linalg.norm(res,axis=0)/np.linalg.norm(y.astype('d'),axis=0)
            np.testing.assert_allclose(relative,r['methods'][name]['runs'][-1]['all_row_range_residual_per_rhs'],rtol=1e-5,atol=1e-14)
            direct=[];double_direct=[]
            for b in range(0,len(rows),64):
                rr=rows[b:b+64];k=analytic_covariance_rows64(x[rr],x)
                kv=k@wd
                # Replay original expression exactly, including finite-only float32 noise multiplication.
                direct.append((kv+.02*cp.asarray(w[rr])).get()-y[rr])
                double_direct.append((kv+.02*wd[rr]).get()-y[rr])
            np.testing.assert_allclose(np.concatenate(direct),data[name+'_sample_residual'],rtol=1e-3,atol=1e-11)
            full_double=np.concatenate(double_direct)
            diagnostics[name]={'full_double_sampled_residual_per_rhs':(np.linalg.norm(full_double,axis=0)/np.linalg.norm(y[rows].astype('d'),axis=0)).tolist(),
                               'maximum_noise_precision_difference':float(np.max(np.abs(full_double-np.concatenate(direct))))}
            if name!='nufft':assert max(diagnostics[name]['full_double_sampled_residual_per_rhs'])<=r['gates']['residual']
            else:assert max(diagnostics[name]['full_double_sampled_residual_per_rhs'])>r['gates']['residual']
            print(name,'million full range and sampled independent audits pass',flush=True)
            del wd,k;cp.get_default_memory_pool().free_all_blocks()
        (base/'validation_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print(f'Replayed {count} A/B/stress fits,4 original-pilot retained mean/sample controls and4 full confirmation fits; finite failures retained.')


if __name__=='__main__':main()
