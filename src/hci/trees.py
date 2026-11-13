import copy
import numpy as np

def clone(x):
    if hasattr(x,'detach'): return x.detach().clone()
    if isinstance(x,np.ndarray): return x.copy()
    if isinstance(x,dict): return {k:clone(v) for k,v in x.items()}
    if isinstance(x,list): return [clone(v) for v in x]
    if isinstance(x,tuple): return tuple(clone(v) for v in x)
    return copy.deepcopy(x)

def as_numpy(x):
    if hasattr(x,'detach'): return x.detach().cpu().numpy()
    return np.asarray(x)

def maxdiff(a,b):
    if isinstance(a,dict):
        if set(a)!=set(b): raise ValueError('State keys changed')
        return max([maxdiff(a[k],b[k]) for k in a]+[0.])
    if isinstance(a,(tuple,list)):
        if len(a)!=len(b): raise ValueError('State length changed')
        return max([maxdiff(x,y) for x,y in zip(a,b)]+[0.])
    aa,bb=as_numpy(a),as_numpy(b)
    if aa.shape!=bb.shape: raise ValueError('State shape changed')
    if not aa.size:return 0.
    if aa.dtype.kind not in 'bifuc':return 0. if np.array_equal(aa,bb) else float('inf')
    if not np.isfinite(aa).all() or not np.isfinite(bb).all(): return float('inf')
    return float(np.abs(aa.astype(float)-bb.astype(float)).max())
