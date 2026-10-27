"""Fresh-runtime confirmation of analytic refinement of a one-million-point finite NUFFT fit."""
import hashlib,json,time
from pathlib import Path
import cupy as cp
import numpy as np
from geospsr.periodic_leaf_cache import LeafCachedRangeKernel
from geospsr.spectral import FourierPoisson
from geospsr.packed_spectral import PackedFourierKernel
from geospsr.analytic_refinement import refine_analytic
from geospsr.spectral_tail import analytic_covariance_rows64
from reproduction.benchmark_leaf_conditioning import SolveAdapter
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit


def main():
    root=Path(__file__).resolve().parents[1];out=root/'reproduction/results/analytic_refinement/confirmation';out.mkdir(exist_ok=True)
    archive=Path('/opt/geospsr-runtime/analytic-refinement/confirmation');archive.mkdir(parents=True,exist_ok=True)
    r={'complete':False,'n':1000000,'scope':'mean fit, all-row range residual and1024 independent direct rows; no posterior/SDD/application speedup',
       'gates':{'residual':1e-8,'weights_relative_l2':1e-7,'mean_relative_l2':1e-7},'methods':{k:{'runs':[]} for k in ['plain','nufft','warm_cg','defect']},
       'source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['geospsr/analytic_refinement.py','geospsr/periodic_leaf_cache.py','geospsr/spectral.py','geospsr/packed_spectral.py','reproduction/confirm_million_refinement.py']}}
    def timed(fn):
        cp.cuda.get_current_stream().synchronize();start=time.perf_counter();v=fn();cp.cuda.get_current_stream().synchronize();return v,time.perf_counter()-start
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    try:
        free,_=cp.cuda.runtime.memGetInfo();r['initial_gpu_free_bytes']=free
        if free<40*2**30:raise MemoryError('need40GiB initially free GPU memory')
        old=root/'reproduction/results/range_compact/scale1m/control.npz';control=np.load(old);x=control['points'];rows=control['rows']
        pilot=root/'reproduction/results/analytic_refinement/million'
        preserved=np.load(pilot/'controls.npz');y=preserved['normals']
        oldmeta=json.loads((pilot/'results.json').read_text())
        assert array_digest(x)==oldmeta['hashes']['points'] and array_digest(y)==oldmeta['hashes']['normals']
        r['pilot_results_sha256']=hashlib.sha256((pilot/'results.json').read_bytes()).hexdigest()
        r['pilot_controls_sha256']=hashlib.sha256((pilot/'controls.npz').read_bytes()).hexdigest()
        r['phase']='single-pass confirmation after runtime loss; settings unchanged'
        q=np.load(root/'reproduction/results/matched/B/input.npz')['query']
        r['hashes']={k:array_digest(v) for k,v in [('points',x),('normals',y),('query',q),('rows',rows)]}
        r['point_archive_sha256']=hashlib.sha256(old.read_bytes()).hexdigest();yd=cp.asarray(y,dtype='d')
        kernel,r['analytic_setup_seconds']=timed(lambda:LeafCachedRangeKernel(x,leaf_size=128,max_entries=200_000_000,max_cache_bytes=32*2**30))
        r['analytic_plan']=kernel.record;analytic=SolveAdapter(kernel)
        spectral,r['finite_spectrum_setup_seconds']=timed(lambda:FourierPoisson(200,upsampfac=1.25))
        finite,r['finite_plan_setup_seconds']=timed(lambda:PackedFourierKernel(x,spectral,.02,rhs_count=3))
        print('setups',r['analytic_setup_seconds'],r['finite_spectrum_setup_seconds']+r['finite_plan_setup_seconds'],flush=True)
        def approx(rhs):return finite.solve(rhs,rtol=1e-5,maxiter=500)
        def progress(it,x):
            if it%16==0:print('analytic CG iteration',it,flush=True)
        last={};reference=None
        def assess(name,weights,info,sec,repeat,initial=0.):
            nonlocal reference
            if name=='plain':reference=weights
            last[name]=weights
            def allrow():return (cp.linalg.norm(kernel(weights)-yd,axis=0)/cp.linalg.norm(yd,axis=0)).get().tolist()
            residual,auditsec=timed(allrow)
            error=(cp.linalg.norm(weights.astype('d')-reference,axis=0)/cp.linalg.norm(reference,axis=0)).get().tolist()
            value={'seconds':sec,'initial_nufft_seconds':initial,'solve_total_seconds':sec+initial,'info':info,
                   'validation_only_range_residual_seconds':auditsec,'all_row_range_residual_per_rhs':residual,
                   'weights_relative_l2_per_rhs':error,'fit_quality_pass':max(residual)<=1e-8 and max(error)<=1e-7}
            r['methods'][name]['runs'].append(value)
            print(name,repeat,sec+initial,info.get('iterations',info.get('corrections')),max(residual),flush=True)
        for repeat in range(1):
            (a,info),sec=timed(lambda:analytic.solve(yd,rtol=5e-9,maxiter=1000,callback=progress));assess('plain',a,info,sec,repeat)
            (initial,iinfo),isec=timed(lambda:approx(yd));assess('nufft',initial,iinfo,isec,repeat)
            (a,info),sec=timed(lambda:analytic.solve(yd,rtol=5e-9,maxiter=1000,x0=initial,callback=progress));assess('warm_cg',a,info,sec,repeat,isec)
            (a,info),sec=timed(lambda:refine_analytic(kernel,approx,yd,initial,rtol=1e-8,max_corrections=8));assess('defect',a,info,sec,repeat,isec)
        # Release finite workspaces before shared double scalar queries/audits.
        del finite,spectral;cp.get_default_memory_pool().free_all_blocks()
        scalar,r['shared_scalar_setup_seconds']=timed(lambda:FourierPoisson(50,dtype='float64',eps=1e-12))
        def mean(weights):
            c=scalar.aggregate(x,weights);return scalar.evaluate(q,c)[0]-scalar.evaluate(x,c).mean()
        arrays={'normals':y,'query':q,'rows':rows};means={}
        for name,weights in last.items():
            value,sec=timed(lambda:mean(weights));means[name]=value;r['methods'][name]['mean_seconds']=sec
            host=weights.get();np.save(archive/f'{name}_weights.npy',host)
            r['methods'][name]['weights_sha256']=array_digest(host);r['methods'][name]['archive_path']=str(archive/f'{name}_weights.npy')
            arrays[name+'_mean']=value.get();arrays[name+'_sample_weights']=host[rows]
        for name,value in means.items():r['methods'][name]['mean_relative_l2']=float(cp.linalg.norm(value-means['plain'])/cp.linalg.norm(means['plain']))
        # Independent direct row sums, every source, all methods, shared blocks.
        def direct_audit():
            result={k:[] for k in last}
            for begin in range(0,len(rows),64):
                rr=rows[begin:begin+64];k=analytic_covariance_rows64(x[rr],x)
                for name,weights in last.items():result[name].append((k@weights.astype('d')+.02*weights[rr]).get()-y[rr])
            return {k:np.concatenate(v) for k,v in result.items()}
        direct,r['direct_audit_seconds']=timed(direct_audit)
        for name,error in direct.items():
            m=r['methods'][name];m['independent_sampled_residual_per_rhs']=(np.linalg.norm(error,axis=0)/np.linalg.norm(y[rows],axis=0)).tolist()
            m['quality_pass']=all(a['fit_quality_pass'] for a in m['runs']) and m['mean_relative_l2']<=1e-7 and max(m['independent_sampled_residual_per_rhs'])<=1e-8
            arrays[name+'_sample_residual']=error
        np.savez_compressed(out/'controls.npz',**arrays);r['complete']=True
    except Exception as exc:r['failure']=str(exc);raise
    finally:
        r['gpu_process_audit']=audit.stop();(out/'results.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
