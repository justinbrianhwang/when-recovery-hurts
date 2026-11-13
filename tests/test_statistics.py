import numpy as np
import pytest
from hci.statistics import *

def test_four_outcomes_and_denominators():
    m=paired_metrics([0,0,1,1],[0,1,0,1])
    assert m['harm_count']==m['rescue_count']==m['both_success']==m['both_fail']==1
    assert m['conditional_harm']==m['conditional_rescue']==.5
    assert m['gain_pp']==0

def test_zero_denominator_is_not_safe():
    m=paired_metrics([0,0],[1,0])
    assert m['conditional_harm'] is None and m['harm_upper_95_pointwise']==1.
    assert binomial_safe_p(0,0,.05)==1.

def test_all_baseline_success_rescue_undefined():
    assert paired_metrics([1,1],[1,1])['conditional_rescue'] is None

def test_zero_event_bound_58_59():
    assert cp_upper(0,58)>.05 and cp_upper(0,59)<=.05
    assert binomial_safe_p(0,58,.05)>.05 and binomial_safe_p(0,59,.05)<=.05
    assert zero_event_requirement(.05)==59

def test_familywise_more_samples():
    assert zero_event_requirement(.05,.05,19)>59

def test_holm_stepdown():
    assert holm_reject([.001,.02,.04],.05)==[True,True,True]
    assert holm_reject([.001,.03,.04],.05)==[True,False,False]
    assert holm_reject([])==[]

def test_success_not_rescue_is_objective():
    # b=.8: rho=.20,h=.01 beats rho=.25,h=.05.
    assert .8*(1-.01)+.2*.20 > .8*(1-.05)+.2*.25

def test_gain_decomposition():
    a=np.array([1]*80+[0]*20);b=a.copy();b[:4]=0;b[80:90]=1
    m=paired_metrics(a,b)
    gain=(1-m['baseline_success_rate'])*m['conditional_rescue']-m['baseline_success_rate']*m['conditional_harm']
    assert np.isclose(gain*100,m['gain_pp'])

def test_bootstrap_reproducible():
    assert bootstrap_gain([0,1,1],[1,1,0],draws=30)==bootstrap_gain([0,1,1],[1,1,0],draws=30)

@pytest.mark.parametrize('a,b',[([],[]),([2],[1]),([1],[1,0]),([[1]],[[1]])])
def test_invalid_pairs(a,b):
    with pytest.raises(ValueError):paired_metrics(a,b)

@pytest.mark.parametrize('k,n',[(-1,10),(11,10),(0,-1),(1.2,10)])
def test_invalid_counts(k,n):
    with pytest.raises(ValueError):cp_upper(k,n)

def test_harm_upper_bounded_by_false_alarm_under_noop_coupling():
    y0=np.array([1,1,1,1,0]);trigger=np.array([1,0,1,0,1]);yg=np.array([0,1,1,1,1])
    assert paired_metrics(y0,yg)['conditional_harm']<=trigger[y0==1].mean()
