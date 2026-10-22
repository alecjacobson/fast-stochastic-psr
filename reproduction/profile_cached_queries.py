import cupy as cp,numpy as np,cufinufft,time,json
from geospsr.cached_spectral import CachedFourierEvaluator
coeff=cp.zeros((128,101,101,101),dtype='F');coeff[:,50,50,50]=cp.arange(128)
c=CachedFourierEvaluator(coeff,batch=1)
q=cp.asarray(np.random.default_rng(3).uniform(-3,3,(6144,3)),dtype='f')
coords=[cp.ascontiguousarray(q[:,i]) for i in range(3)]
def timed(fn):
 cp.cuda.runtime.deviceSynchronize();t=time.perf_counter();v=fn();cp.cuda.runtime.deviceSynchronize();return v,time.perf_counter()-t
for batch in [4,8,16,32,64]:
 p,setup=timed(lambda:cufinufft.Plan(2,c.shape,n_trans=batch,eps=c.eps,dtype='complex64',gpu_spreadinterponly=1,upsampfac=c.oversampling,isign=1))
 _,pts=timed(lambda:p.setpts(*coords))
 def run():return cp.concatenate([p.execute(c.grid[i:i+batch]) for i in range(0,c.packed_count,batch)])
 _,first=timed(run); ts=[timed(run)[1] for i in range(3)]
 print(json.dumps({'batch':batch,'makeplan':setup,'setpts':pts,'execute_first':first,'execute_median':np.median(ts)}),flush=True)
