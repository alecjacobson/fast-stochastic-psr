"""Full-slice factor-space deflation with independent confirmation."""
import argparse,json,time,hashlib
from pathlib import Path
import cupy as cp
import cupyx.scipy.linalg as cla
import numpy as np
from scipy.special import ndtr
from geospsr.spectral import FourierPoisson
from geospsr.spectral_tail import analytic_covariance_rows64
from reproduction.posterior import Posterior,sync
from reproduction.correction_sampling import DenseCorrectionSampler
from reproduction.factor_deflation import FourierCorrectionFactor,FactorDeflatedMoments
from reproduction.benchmark_correction_contrast import deterministic_variance,metrics
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['A','B'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--factor-batch',type=int,choices=[4,16],default=4)
    p.add_argument('--confirm',type=Path,help='pilot results.json; use its selected rank/count and new seeds')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);a.archive.mkdir(parents=True,exist_ok=True)
    ranks,looks,seeds=[64,128,256,512],[32,128,512,2048],[(2401,401),(2503,503),(2607,607)]
    if a.confirm:
        pilot=json.loads(a.confirm.read_text())
        if not pilot['complete'] or pilot['case']!=a.case or pilot['selected'] is None:raise ValueError('no eligible pilot setting')
        ranks=[pilot['selected']['rank']];looks=[pilot['selected']['count']];seeds=[(2709,709),(2801,801),(2907,907)]
    r={'complete':False,'case':a.case,'phase':'confirmation' if a.confirm else 'pilot','ranks':ranks,'looks':looks,
       'seeds':seeds,'estimator':'factor-space','factor_batch':a.factor_batch,'seconds':{},'runs':[],'selected':None,
       'gates':{'contrast_relative_l2':.01,'probability_rms':.0005,'probability_max':.005},
       'control_precision_gate':.001,'model':'full analytic vector covariance; scalar/cross cutoff50; noise .02; direct Fourier',
       'scope':'first fixed-query marginal variance estimate; sketch/factor-action costs included; not posterior event samples',
       'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}
    def save():(a.output/'results.json').write_text(json.dumps(r,indent=2)+'\n')
    def timed(fn):
        sync();start=time.perf_counter();value=fn();sync();return value,time.perf_counter()-start
    def archive(name,value):
        if isinstance(value,cp.ndarray):value=cp.asnumpy(value)
        np.save(a.archive/(name+'.npy'),value);return array_digest(value)
    def good(m):return all(m[k]<=v for k,v in r['gates'].items()) and m['nonpositive_count']==0
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start();wall=time.perf_counter()
    try:
        data=np.load(ROOT/f'reproduction/results/matched/{a.case}/input.npz')
        x,y,q=[data[k] for k in ['points','normals','query']]
        old=ROOT/f'reproduction/results/correction_contrast/{a.case}'
        original=json.loads((old/'results.json').read_text());ref=np.load(old/'reference.npz')
        r['hashes']={k:array_digest(v) for k,v in [('points',x),('normals',y),('queries',q)]}
        if r['hashes']!=original['hashes'] or not np.array_equal(q,ref['query']):raise ValueError('reference/input mismatch')
        if a.confirm and r['hashes']!=pilot['hashes']:raise ValueError('pilot input mismatch')
        r['reference_payload_sha256']=hashlib.sha256((old/'reference.npz').read_bytes()).hexdigest()
        r.update(N=len(x),queries=len(q),batch=8)
        reference=ref['variance'];prior=float(ref['prior_variance'])
        model,r['seconds']['fit']=timed(lambda:Posterior(x,y,upsampfac=1.25))
        mean,r['seconds']['mean_queries']=timed(lambda:cp.asnumpy(model.mean(q)))
        np.savez_compressed(a.output/'reference.npz',mean=mean,variance=reference,prior_variance=prior,query=q)
        r['deterministic_seconds']=[]
        for repeat in range(3):
            direct,seconds=timed(lambda:deterministic_variance(model.op,x,model.L,q,prior))
            r['deterministic_seconds'].append(seconds)
        r['deterministic_quality']=metrics(direct,reference,mean,prior);r['deterministic_quality_pass']=good(r['deterministic_quality'])
        r['deterministic_median_seconds']=float(np.median(r['deterministic_seconds']))
        def reference_setup():
            op=FourierPoisson(50,dtype='float64',eps=1e-12)
            matrix=analytic_covariance_rows64(x,x)+.02*cp.eye(len(x),dtype='d')
            return op,cp.linalg.cholesky(matrix)
        (double,chol),r['seconds']['reference_factor_setup']=timed(reference_setup)
        def make_factor(op,chol):
            factor=FourierCorrectionFactor(x,q,op,chol,batch=a.factor_batch);factor.prepare(a.factor_batch);return factor
        factor,r['seconds']['action_setup']=timed(lambda:make_factor(model.op,model.L))
        reference_factor,r['seconds']['reference_action_setup']=timed(lambda:make_factor(double,chol))
        sampler=DenseCorrectionSampler(x,model.L,model.op)
        query_plan,r['seconds']['query_plan']=timed(lambda:model.op.plan(2,q,8))
        contrast_norm=float(np.linalg.norm(reference-reference.mean()))
        for pilot_seed,seed in seeds:
            run={'pilot_seed':pilot_seed,'seed':seed,'settings':{},'checkpoints':[],'cumulative':{'generation':0.,'queries':0.}}
            states={0:FactorDeflatedMoments(cp.empty((3*len(x),0)),cp.empty((len(q),0)),xp=cp)}
            reduction={str(rank):0. for rank in [0]+ranks}
            for rank in ranks:
                def make_sketch():
                    return cp.random.RandomState(pilot_seed).standard_normal((len(q),rank),dtype=model.op.dtype)
                omega,sketch_time=timed(make_sketch)
                raw,adjoint_time=timed(lambda:factor.adjoint(omega));del omega
                basis,qr_time=timed(lambda:cp.linalg.qr(raw.astype('d'),mode='reduced')[0]);del raw
                u,forward_time=timed(lambda:factor.forward(basis))
                state,control_time=timed(lambda:FactorDeflatedMoments(basis,u,xp=cp))
                u64,reference_time=timed(lambda:reference_factor.forward(basis))
                control64=cp.sum(u64*u64,axis=1)
                discrepancy=float(cp.linalg.norm(state.deterministic-control64))/contrast_norm
                hashes={'basis':archive(f'{seed}_{rank}_basis',basis),'factor_basis':archive(f'{seed}_{rank}_factor_basis',u),
                        'control':archive(f'{seed}_{rank}_control',state.deterministic),'reference_control':archive(f'{seed}_{rank}_reference_control',control64)}
                run['settings'][str(rank)]={'sketch_seconds':sketch_time,'adjoint_seconds':adjoint_time,'qr_seconds':qr_time,
                    'forward_seconds':forward_time,'control_seconds':control_time,'reference_action_seconds':reference_time,
                    'precision_discrepancy_over_contrast':discrepancy,'precision_pass':discrepancy<=r['control_precision_gate'],
                    'record':state.record,'hashes':hashes}
                states[rank]=state
                print('sketch',a.case,seed,rank,'precision',discrepancy,flush=True)
                del u,u64,basis,control64
            arrays={};stream=iter(sampler.draw_batches(max(looks),seed=seed,batch=8,retain_standard_normal=True))
            for done in range(8,max(looks)+1,8):
                block,seconds=timed(lambda:next(stream));run['cumulative']['generation']+=seconds
                values,seconds=timed(lambda:query_plan.execute(block['coefficients']).real);run['cumulative']['queries']+=seconds
                for rank,state in states.items():
                    _,seconds=timed(lambda:state.update(values,block['standard_normal']));reduction[str(rank)]+=seconds
                if done in looks:
                    for rank,state in states.items():
                        def output():
                            variance=cp.asnumpy(prior-state.diagonal())
                            probability=ndtr(-mean/np.sqrt(np.maximum(variance,np.finfo(float).tiny)))
                            return variance,probability
                        (estimate,probability),seconds=timed(output)
                        m=metrics(estimate,reference,mean,prior)
                        setup=(sum(run['settings'][str(rank)][k] for k in ['sketch_seconds','adjoint_seconds','qr_seconds','forward_seconds','control_seconds'])+
                               r['seconds']['action_setup']) if rank else 0.
                        total=setup+r['seconds']['query_plan']+sum(run['cumulative'].values())+reduction[str(rank)]+seconds
                        check={'rank':rank,'count':done,'metrics':m,'quality_pass':good(m) and (not rank or run['settings'][str(rank)]['precision_pass']),
                               'first_estimate_seconds':total,'setup_seconds':setup,'shared_generation_query_seconds':sum(run['cumulative'].values()),
                               'reduction_seconds':reduction[str(rank)],'output_seconds':seconds}
                        run['checkpoints'].append(check);arrays[f'variance_{rank}_{done}']=estimate
                        arrays[f'probability_{rank}_{done}']=probability
                        arrays[f'residual_second_moment_{rank}_{done}']=cp.asnumpy(state.sum_squares/done)
                        print('checkpoint',a.case,seed,rank,done,m['contrast_relative_l2'],total,flush=True)
                del values,block
            stream.close()
            np.savez_compressed(a.output/f'seed_{seed}.npz',**arrays)
            r['runs'].append(run);save()
        candidates=[]
        for rank in ranks:
            for count in looks:
                checks=[next(c for c in run['checkpoints'] if c['rank']==rank and c['count']==count) for run in r['runs']]
                if all(c['quality_pass'] for c in checks):
                    candidates.append({'rank':rank,'count':count,'median_first_estimate_seconds':float(np.median([c['first_estimate_seconds'] for c in checks]))})
        r['eligible_settings']=candidates
        r['selected']=min(candidates,key=lambda c:c['median_first_estimate_seconds']) if candidates else None
        r['complete']=True
    except Exception as exc:r['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:
        r['seconds']['work_wall']=time.perf_counter()-wall;r['gpu_process_audit']=audit.stop();save()


if __name__=='__main__':main()
