"""Fixed inverse-root correction preparation at overlap and massive sizes."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from geospsr.inverse_root import FixedChebyshevInverseRoot,spatial_box_spectral_bound
from reproduction.common_rhs_integrity import array_digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',choices=['A','million','three_million'],required=True)
    p.add_argument('--source',type=Path,default=Path('/opt/geospsr-runtime/large-cloud-3000000'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import cupy as cp
    from geospsr.half_spectral import HalfCubeKernel,HalfCubeSpectrum
    from geospsr.spectral import FourierPoisson,spectral_axis
    from reproduction.correction_sampling import PolynomialCorrectionSampler
    from reproduction.gpu_audit import GpuProcessAudit
    from reproduction.posterior import sync
    root=Path(__file__).resolve().parent/'results'
    a.output.mkdir(parents=True,exist_ok=True)
    record={'complete':False,'case':a.case,'seconds':{},'polynomial_draws':[], 'pathwise_draws':[],
            'model':{'training_cutoff':200,'scalar_cutoff':50,'noise_variance':.02,'lengthscale':.01,'variance':.1,'eps':1e-6},
            'seeds':{'polynomial':401,'pathwise':509},'draws':2,'batch':1,
            'scope':'correction preparation only; not posterior event samples or marginal Monte Carlo convergence',
            'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start();wall=time.perf_counter()
    arrays={}
    def save():
        (a.output/'results.json').write_text(json.dumps(record,indent=2)+'\n')
    def timed(label,fn):
        sync();start=time.perf_counter();value=fn();sync()
        seconds=time.perf_counter()-start
        print(label,seconds,flush=True)
        return value,seconds
    def progress(stage):
        def report(i,total):
            if i%64==0 or i==total:
                sync();print(stage,i,total,flush=True)
        return report
    try:
        def source():
            data=np.load(root/'matched/A/input.npz')
            if a.case=='A':
                x=data['points']
                provenance={'kind':'released A','source_file':'reproduction/results/matched/A/input.npz'}
            else:
                full=np.load(a.source/'points.npy',mmap_mode='r')
                manifest=json.loads((root/'large_inputs/scorpion_3000000/manifest.json').read_text())
                digest=array_digest(full)
                if digest!=manifest['arrays']['points']['payload_sha256']:
                    raise ValueError('source hash mismatch')
                x=np.array(full[:1000000] if a.case=='million' else full)
                provenance={'kind':'prefix of published three-million-point source' if a.case=='million' else 'published three-million-point source',
                            'parent_source_sha256':digest,'prefix_count':len(x)}
            ids=np.random.default_rng(404).choice(len(data['query']),128,replace=False)
            return x,data['query'][ids],provenance
        (x,q,provenance),record['seconds']['source']=timed('source',source)
        record.update(N=len(x),source=provenance,source_sha256=array_digest(x),query_sha256=array_digest(q))
        arrays['query']=q
        def polynomial_setup():
            bound=spatial_box_spectral_bound(x)
            polynomial=FixedChebyshevInverseRoot(bound['lower'],bound['upper'],1e-4)
            trace=FixedChebyshevInverseRoot(.02,len(x)*.001+.02,1e-4)
            return bound,polynomial,trace.record
        (bound,polynomial,trace),record['seconds']['polynomial_setup']=timed('polynomial_setup',polynomial_setup)
        record.update(spatial_bound=bound,polynomial=polynomial.record,trace_control=trace)
        def operators():
            op=FourierPoisson(50)
            spectrum=HalfCubeSpectrum(200)
            kernel=HalfCubeKernel(x,spectrum,.02)
            sampler=PolynomialCorrectionSampler(x,op,polynomial,lambda r:kernel if r==3 else HalfCubeKernel(x,spectrum,.02,r))
            sampler.prepare(1)
            query=op.plan(2,q,1)
            return op,kernel,sampler,query
        (op,kernel,sampler,query_plan),record['seconds']['shared_operator_plans']=timed('shared_operator_plans',operators)
        exact_root=None
        if a.case=='A':
            def dense_reference():
                xd=x.astype('d');modes,rho=spectral_axis(200)
                matrix=np.empty((len(x),len(x)))
                for start in range(0,len(x),16):
                    value=np.ones((min(16,len(x)-start),len(x)))
                    for d in range(3):
                        delta=xd[start:start+16,None,d]-xd[None,:,d]
                        value*=np.cos(delta[...,None]*modes)@rho
                    matrix[start:start+16]=value
                matrix+=.02*np.eye(len(x))
                eig,vec=np.linalg.eigh(matrix)
                if eig.min()<bound['lower']*(1-1e-10) or eig.max()>bound['upper']:
                    raise ValueError('dense spectrum outside interval')
                record['dense_eigenvalue_range']=[float(eig.min()),float(eig.max())]
                return (vec/np.sqrt(eig))@vec.T
            exact_root,record['seconds']['dense_reference']=timed('dense_reference',dense_reference)
            ref_op=FourierPoisson(50,dtype='float64',eps=1e-12)
        rng=cp.random.RandomState(401)
        for draw in range(2):
            entry={'draw':draw}
            xi,entry['gaussian_seconds']=timed(f'polynomial_{draw}_gaussian',lambda:rng.standard_normal((len(x),3),dtype='f'))
            weights,entry['inverse_root_seconds']=timed(f'polynomial_{draw}_root',lambda:polynomial.apply(kernel,xi,xp=cp,progress=progress('root')))
            coeff,entry['aggregation_seconds']=timed(f'polynomial_{draw}_aggregate',lambda:sampler.aggregate(weights))
            values,entry['query_seconds']=timed(f'polynomial_{draw}_query',lambda:cp.asnumpy(query_plan.execute(coeff).real[0]))
            entry['draw_through_query_seconds']=sum(entry[k] for k in ['gaussian_seconds','inverse_root_seconds','aggregation_seconds','query_seconds'])
            def validation():
                twice=polynomial.apply(kernel,weights,xp=cp,progress=progress('roundtrip'))
                errors=cp.asnumpy(cp.linalg.norm(kernel(twice)-xi,axis=0)/cp.linalg.norm(xi,axis=0))
                return errors
            errors,entry['validation_seconds']=timed(f'polynomial_{draw}_roundtrip',validation)
            entry['roundtrip_relative_errors']=errors.tolist()
            entry['quality_pass']=bool(np.max(errors)<=.001)
            xi_host=cp.asnumpy(xi);weights_host=cp.asnumpy(weights)
            if exact_root is not None:
                def overlap():
                    reference=exact_root@xi_host.astype('d')
                    reference_values=cp.asnumpy(ref_op.evaluate(q,ref_op.aggregate(x,reference))[0])
                    arrays[f'dense_{draw}']=reference_values
                    return float(np.linalg.norm(weights_host-reference)/np.linalg.norm(reference)),float(np.linalg.norm(values-reference_values)/np.linalg.norm(reference_values))
                (we,fe),entry['dense_validation_seconds']=timed(f'polynomial_{draw}_dense_overlap',overlap)
                entry.update(dense_weights_relative_error=we,dense_fields_relative_error=fe)
                entry['quality_pass'] &= we<=.001 and fe<=.001
            arrays[f'polynomial_{draw}']=values
            np.save(a.output/f'xi_{draw}.npy',xi_host);np.save(a.output/f'weights_{draw}.npy',weights_host)
            np.save(a.output/f'coefficients_{draw}.npy',cp.asnumpy(coeff))
            entry.update(xi_sha256=array_digest(xi_host),weights_sha256=array_digest(weights_host),coefficients_sha256=array_digest(cp.asnumpy(coeff)))
            record['polynomial_draws'].append(entry);save()
            del xi,weights,coeff,xi_host,weights_host
        random=cp.random.RandomState(509)
        for draw in range(2):
            entry={'draw':draw}
            def prior_rhs():
                vector,scalar=kernel.joint_prior(random,op)
                return vector+np.float32(np.sqrt(.02))*random.standard_normal(vector.shape,dtype='f')
            rhs,entry['joint_prior_noise_seconds']=timed(f'pathwise_{draw}_rhs',prior_rhs)
            (weights,info),entry['solve_seconds']=timed(f'pathwise_{draw}_solve',lambda:kernel.solve(rhs,rtol=1e-5,maxiter=1000))
            coeff,entry['aggregation_seconds']=timed(f'pathwise_{draw}_aggregate',lambda:sampler.aggregate(weights))
            values,entry['query_seconds']=timed(f'pathwise_{draw}_query',lambda:cp.asnumpy(query_plan.execute(coeff).real[0]))
            entry.update(solve=info,quality_pass=bool(info['converged']))
            entry['draw_through_query_seconds']=sum(entry[k] for k in ['joint_prior_noise_seconds','solve_seconds','aggregation_seconds','query_seconds'])
            arrays[f'pathwise_{draw}']=values
            record['pathwise_draws'].append(entry);save()
            del rhs,weights,coeff
        np.savez_compressed(a.output/'fields.npz',**arrays)
        record['complete']=True
    except Exception as exc:
        record['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:
        record['seconds']['work_wall']=time.perf_counter()-wall
        record['gpu_process_audit']=audit.stop();save()


if __name__=='__main__':main()
