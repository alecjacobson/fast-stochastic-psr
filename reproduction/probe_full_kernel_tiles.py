import json,time
import cupy as cp
from geospsr.nonperiodic_dense_gpu import _TILED_KERNEL
n=796825
x=cp.random.RandomState(12).random_sample((n,3))
b=cp.ones((n,54))
k=cp.RawKernel(_TILED_KERNEL,'covariance_tile')
rows=[]
for nr,nc in [(84,n),(2048,32768)]:
 a=cp.empty((nr,nc)); elapsed=[]
 for repeat in range(4):
  cp.cuda.get_current_stream().synchronize(); start=time.perf_counter()
  k(((nr*nc+255)//256,), (256,), (x,a,0,0,nr,nc,57.735,.0025,.000025))
  y=a@b[:nc]
  cp.cuda.get_current_stream().synchronize()
  elapsed.append(time.perf_counter()-start)
 rows.append(dict(rows=nr,columns=nc,entries=nr*nc,seconds=elapsed))
print(json.dumps(rows,indent=2))
