from __future__ import annotations
import numpy as np
from .io import seed_for

ACTIONS=('continue','refresh','retreat_retry')

def default_registry():
    candidates=[{'id':'never','family':'never','kind':'never'}]
    for action in (1,2):
        for t in (28,60):candidates.append({'id':f'timer_a{action}_t{t}','family':'timer','kind':'timer','time':t,'action':action})
        for prob in (.03,.1):candidates.append({'id':f'random_a{action}_p{prob}','family':'random','kind':'random','probability':prob,'action':action})
    for th in (.5,.8):candidates.append({'id':f'risk_q_{th}','family':'risk_q','kind':'risk_q','threshold':th})
    for th in (0.,.05,.15):candidates.append({'id':f'q_{th}','family':'q','kind':'q','threshold':th})
    for penalty in (0.,1.,3.):
        for th in (0.,.05):candidates.append({'id':f'joint_l{penalty}_t{th}','family':'joint','kind':'joint','penalty':penalty,'threshold':th})
    return {'version':1,'candidates':candidates,'fixed_before_calibration':True,
            'epsilon_primary':.05,'delta_familywise':.05,
            'note':'All families can use both non-continue actions. Q and joint share the same episode-level risk selection.'}

def choose_action(c,*,eligible,episode,t,q=None,joint=None):
    eligible=np.asarray(eligible,bool)
    if eligible.shape!=(3,) or not eligible[0]:raise ValueError('Action mask must have continue available')
    k=c['kind']
    if k=='never':return 0
    if k in ('timer','random'):
        a=c['action']
        if not eligible[a]:return 0
        if k=='timer':return a if t>=c['time'] else 0
        z=np.random.default_rng(seed_for(episode,t,0,'trigger')).random()
        return a if z<c['probability'] else 0
    if q is None or np.asarray(q).shape!=(3,):raise ValueError('Three Q estimates required')
    if k=='risk_q':
        if 1-q[0]<c['threshold']:return 0
        scores=np.asarray(q[1:])-q[0]
    elif k=='q':scores=np.asarray(q[1:])-q[0]-c['threshold']
    elif k=='joint':
        if joint is None or np.asarray(joint).shape!=(2,4):raise ValueError('Two four-class outcome estimates required')
        # Classes: both fail, rescue, harm, both succeed. Rescue-harm = marginal success gain.
        scores=joint[:,1]-joint[:,2]-c['penalty']*joint[:,2]-c['threshold']
    else:raise ValueError(f'Unknown decision rule {k}')
    scores=np.array(scores,copy=True);scores[~eligible[1:]]=-np.inf
    if not np.isfinite(scores).any() or np.max(scores)<=0:return 0
    return int(np.argmax(scores)+1)
