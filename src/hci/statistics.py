"""Finite-candidate, episode-level outcome-risk selection. No new theory claimed."""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import beta, binom

def _counts(k,n):
    if not isinstance(k,(int,np.integer)) or not isinstance(n,(int,np.integer)) or n<0 or not 0<=k<=n:
        raise ValueError('Require integer 0 <= k <= n.')

def cp_upper(k:int,n:int,delta:float=.05):
    _counts(k,n)
    if not 0<delta<1: raise ValueError('delta must be in (0,1)')
    if n==0 or k==n: return 1.0
    return float(beta.ppf(1-delta,k+1,n-k))

def binomial_safe_p(k:int,n:int,epsilon:float):
    _counts(k,n)
    if not 0<epsilon<1: raise ValueError('epsilon must be in (0,1)')
    if n==0: return 1.0
    return float(binom.cdf(k,n,epsilon))

def holm_reject(pvalues,delta=.05):
    if not 0<delta<1: raise ValueError('delta must be in (0,1)')
    ps=np.asarray(pvalues,dtype=float)
    if ps.ndim!=1 or not np.isfinite(ps).all() or np.any((ps<0)|(ps>1)):
        raise ValueError('Invalid p-values')
    out=np.zeros(len(ps),dtype=bool)
    for j,i in enumerate(np.argsort(ps,kind='stable')):
        if ps[i]<=delta/(len(ps)-j): out[i]=True
        else: break
    return out.tolist()

def paired_metrics(y0,yg):
    a=np.asarray(y0);b=np.asarray(yg)
    if a.ndim!=1 or a.shape!=b.shape or not len(a): raise ValueError('Nonempty paired 1-D arrays required')
    if not np.isin(a,[0,1]).all() or not np.isin(b,[0,1]).all(): raise ValueError('Binary outcomes only')
    n=len(a);np_=int(a.sum());nf=n-np_
    harm=int(((a==1)&(b==0)).sum());rescue=int(((a==0)&(b==1)).sum())
    return {'n':n,'baseline_successes':np_,'baseline_failures':nf,'harm_count':harm,'rescue_count':rescue,
            'both_success':int(((a==1)&(b==1)).sum()),'both_fail':int(((a==0)&(b==0)).sum()),
            'baseline_success_rate':float(a.mean()),'success_rate':float(b.mean()),
            'gain_pp':float((b.mean()-a.mean())*100),
            'conditional_harm':harm/np_ if np_ else None,
            'conditional_rescue':rescue/nf if nf else None,
            'harm_upper_95_pointwise':cp_upper(harm,np_),
            'risk_denominator_is_baseline_successes':True}

def bootstrap_gain(y0,yg,seed=42,draws=2000):
    a=np.asarray(yg,dtype=float)-np.asarray(y0,dtype=float)
    if a.ndim!=1 or not len(a): raise ValueError('Nonempty episode pairs required')
    rng=np.random.default_rng(seed)
    values=np.array([a[rng.integers(0,len(a),len(a))].mean() for _ in range(draws)])
    return [float(x*100) for x in np.quantile(values,[.025,.975])]

def zero_event_requirement(epsilon,delta=.05,m=1):
    if not 0<epsilon<1 or not 0<delta<1 or m<1: raise ValueError('Invalid design')
    return int(math.ceil(math.log(delta/m)/math.log1p(-epsilon)))
