"""Bounded-memory sample moments and fixed-look Bernoulli error bounds.

No independent-spatial-query assumption is made. The Hoeffding union bound
covers every reported event and the declared number of looks, but concerns
Monte Carlo error only, not geometry, model, or numerical error.
"""
import numpy as np


class OnlineMoments:
    def __init__(self,covariance=False):
        self.count=0
        self.mean=self.m2=None
        self.covariance=covariance

    def update(self,values):
        values=np.asarray(values,dtype=np.float64)
        if values.ndim!=2 or not values.size or not np.isfinite(values).all():
            raise ValueError('expected nonempty finite (samples,queries) array')
        n=len(values);mean=values.mean(0);centered=values-mean
        m2=centered.T@centered if self.covariance else (centered*centered).sum(0)
        if not self.count:
            self.count=n;self.mean=mean;self.m2=m2
            return
        if self.mean.shape!=mean.shape:
            raise ValueError('query shape changed')
        delta=mean-self.mean;total=self.count+n
        correction=np.outer(delta,delta) if self.covariance else delta*delta
        self.m2+=m2+correction*(self.count*n/total)
        self.mean+=delta*(n/total)
        self.count=total

    def summary(self):
        if self.count<2:
            raise ValueError('at least two samples required')
        return {'count':self.count,'mean':self.mean.copy(),
                'covariance' if self.covariance else 'variance':self.m2/(self.count-1)}


class OnlineEvents:
    def __init__(self):
        self.count=0
        self.successes=None

    def update(self,events):
        events=np.asarray(events)
        if events.ndim<2 or not events.size or not np.all((events==0)|(events==1)):
            raise ValueError('expected nonempty binary sample events')
        successes=events.sum(0,dtype=np.int64)
        if self.successes is None:
            self.successes=successes
        else:
            if successes.shape!=self.successes.shape:
                raise ValueError('event geometry changed')
            self.successes+=successes
        self.count+=len(events)

    def summary(self,alpha=.05,looks=1,family_size=None):
        if not self.count or not 0<alpha<1 or looks!=int(looks) or looks<1:
            raise ValueError('require samples, 0<alpha<1 and positive integer looks')
        family=self.successes.size if family_size is None else family_size
        if family!=int(family) or family<self.successes.size:
            raise ValueError('family_size must cover every event')
        radius=float(np.sqrt(np.log(2*family*looks/alpha)/(2*self.count)))
        probability=self.successes/self.count
        return {'count':self.count,'probability':probability,'successes':self.successes.copy(),
                'lower':np.maximum(0,probability-radius),'upper':np.minimum(1,probability+radius),
                'simultaneous_radius':radius,'alpha':alpha,'looks':looks,'family_size':int(family),
                'qualification':'Hoeffding union bound for independent draws and predeclared looks; Monte Carlo error only'}


def view_scores(ray_summary,distance):
    probability=ray_summary['probability'];lower=ray_summary['lower'];upper=ray_summary['upper']
    distance=np.asarray(distance)
    if probability.ndim!=2 or probability.shape[1]!=len(distance) or not np.all(np.diff(distance)>0):
        raise ValueError('expected rays by increasing ray-distance samples')
    score=np.trapezoid(((probability>=.05)&(probability<=.95)).astype(float),distance,axis=1)
    low=np.trapezoid(((lower>=.05)&(upper<=.95)).astype(float),distance,axis=1)
    high=np.trapezoid(((upper>=.05)&(lower<=.95)).astype(float),distance,axis=1)
    selected=int(score.argmax())
    return {'scores':score,'lower':low,'upper':high,'selected':selected,
            'regret_upper_bound':float(max(0,high.max()-low[selected]))}
