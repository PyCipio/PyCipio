import numpy as np


# det_dot function
def det_dot(a, b):
    """
    The theano dot product and NUTS sampler don't work with large matrices?
    
    :param a: (np matrix)
    :param b: (theano vector)
    """
    return (a * b[None, :]).sum(axis=-1)


### test this 
a = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]) # (5, 3)
b = np.array([1, 2, 3]) # (3, 2)
det_dot(a, b)
b[None, :]
# data
time = np.arange(0, 20, 0.1);

# Amplitude of the sine wave is sine of a variable like time
sines = np.sin(time) + np.random.normal(0, 0.2, 200)
coses = np.cos(time) + np.random.normal(0, 0.2, 200)
y_true = sines + coses


def fourier_series(t, p=365.25, n=10):
    # 2 pi n / p
    x = 2 * np.pi * np.arange(1, n + 1) / p
    # 2 pi n / p * t
    x = x * t[:, None]
    x = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    return x

## testing it 
n = 4
t = np.arange(1000)
beta = np.random.normal(size=2 * n)
#plt.figure(figsize=(16, 6))
#plt.plot(fourier_series(t, 365.25, n) @ beta)
x[0]

test = np.dot(x, beta)
test.shape

x = fourier_series(t, 365.25, n)
x.shape
def seasonality_model(m, df, period='yearly', seasonality_prior_scale=10):
    
    if period == 'yearly':
        n = 10
        # rescale the period, as t is also scaled
        p = 365.25 / (df['ds'].max() - df['ds'].min()).days
    else:  # weekly
        n = 3
        # rescale the period, as t is also scaled
        p = 7 / (df['ds'].max() - df['ds'].min()).days
    x = fourier_series(df['t'], p, n)
    with m:
        beta = pm.Normal(f'beta_{period}', mu=0, sd=seasonality_prior_scale, shape=2 * n)
    return x, beta

with m:
    # changepoints_prior_scale is None, so the exponential distribution
    # will be used as prior on \tau.
    y, A, s = trend_model(m, df['t'], changepoints_prior_scale=None)
    x_yearly, beta_yearly = seasonality_model(m, df, 'yearly')
    x_weekly, beta_weekly = seasonality_model(m, df, 'weekly')
    
    y += det_dot(x_yearly, beta_yearly) + det_dot(x_weekly, beta_weekly)
    
    sigma = pm.HalfCauchy('sigma', 0.5, testval=1)
    obs = pm.Normal('obs', 
                 mu=y, 
                 sd=sigma,
                 observed=df['y_scaled'])
    

####  https://discourse.pymc.io/t/multiple-time-series/5868
from scipy import stats
N, s = 100, 10 # number of points per series and number of series

x = np.zeros((N,s))

true_thetas = np.repeat(np.array([[0.7]]),s).reshape(1,s) + np.random.normal(0, 0.2, s)
true_taus = stats.halfnorm.rvs(0.1, size=s)
true_center = 0
x[0]=0.0

for i in range(1, N):
    x[i] = true_thetas[0] * (x[i-1]) + stats.norm.rvs(true_center, np.sqrt(1/true_taus), s)


with pm.Model() as m_ar:
    
    theta = pm.Normal('theta', 0, 1, shape=s) # The shape here is stating that there is only 1 theta for each series
    tau = pm.HalfNormal('tau', sd=1, shape=s) # The shape here is stating that there is only 1 tau for each series
    
    # AR Process
    η_t = pm.AR1("η_t", k=theta, tau_e=tau, observed=x) # Here we are using the observed data, so it will evaluate for the 100 parameters of each series
    
    trace = pm.sample()

## importing stuff
import numpy as np 
import pandas as pd 
from covid19dh import covid19
from datetime import date
from Get_covid_data import get_data
import pymc3 as pm
import pandas as pd 
import numpy as np 
import seaborn as sns
import arviz as az
import matplotlib.pyplot as plt
import os 
import theano
import theano.tensor as tt 
import random
import fns as f

## simulating
time = np.arange(0, 30, 1) 
time_true = np.append(time, time)

# Amplitude of the sine wave is sine of a variable like time
sines1 = np.sin(time) + np.random.normal(0, 0.2, 30)
sines2 = np.sin(time) + np.random.normal(0, 0.3, 30)
coses1 = np.cos(time) + np.random.normal(0, 0.2, 30)
coses2 = np.cos(time) + np.random.normal(0, 0.4, 30)
line1 = 1 + 0.5 * time + np.random.normal(0, 0.2, 30) 
line2 = 0.5 + 0.3 * time + np.random.normal(0, 0.2, 30)

# set up y
y1 = np.array(sines1 + coses1 + line1)
y2 = np.array(sines2 + coses2 + line2)

y_true = np.stack([y1, y2], axis = 1)
y_true.shape # n of points, and n of series. 

# set up idx
idx1 = np.zeros(30, dtype = int)
idx2 = np.ones(30, dtype = int)

idx_true = np.stack([idx1, idx2], axis = 1)
idx_true.shape # n of points and n of series. 

## parameters
n = 2
N = 2
p = 6.5
c = (2 * np.pi * np.arange(1, n + 1) / p)[:, None]
seasonality_prior_scale=2

## model 
with pm.Model() as m1: 
    
    # shared 
    t_shared = pm.Data('t_shared', time)
    idx_shared = pm.Data('idx_shared', idx_true)
    
    # estimate it? 
    p_1 = pm.Normal("p_1", mu = p, sd = 1)
    week = (2*np.pi*np.arange(1, n+1) / p_1)[:, None]
    
    # creating fourier
    x_week = week * t_shared
    #x_week = c_week * t_shared
    # x_month = c_month * t_shared
    
    x_week_waves = tt.concatenate((tt.cos(x_week), tt.sin(x_week)), axis=0)
    # x_month_waves = tt.concatenate((tt.cos(x_month), tt.sin(x_month)), axis = 0)
    
    # beta
    beta_week_waves = pm.Normal('beta_week_waves', mu = 0, sd = seasonality_prior_scale, shape = 2*n) 
    # beta_month_waves = pm.Normal('beta_month_waves', mu = 0, sd = seasonality_prior_scale, shape = (2*n, N))
    beta_line = pm.Normal('beta_line', mu = 0, sd = 0.5)
    alpha = pm.Normal('alpha', mu = 0, sd = 0.5)
    
    
    ## new try 
    mu = alpha + beta_line * t_shared + pm.math.dot(x_week_waves.T, beta_week_waves)
    
    # sigma 
    sigma = pm.HalfCauchy('sigma', 0.05)
    
    # likelihood 
    y_pred = pm.Normal('y_pred', 
                       mu = mu,
                       sd = sigma,
                       observed = y_true)

###### for loop ######
time = np.arange(0, 30, 1) 
time_true = np.append(time, time)

# Amplitude of the sine wave is sine of a variable like time
sines1 = np.sin(time) + np.random.normal(0, 0.2, 30)
sines2 = np.sin(time) + np.random.normal(0, 0.3, 30)
coses1 = np.cos(time) + np.random.normal(0, 0.2, 30)
coses2 = np.cos(time) + np.random.normal(0, 0.4, 30)
line1 = 1 + 0.5 * time + np.random.normal(0, 0.2, 30) 
line2 = 0.5 + 0.3 * time + np.random.normal(0, 0.2, 30)

y1 = sines1 + coses1 + line1
y2 = sines2 + coses2 + line2

y_true = np.append(y1, y2)
idx_true = np.append(np.zeros(30, dtype = int), np.ones(30, dtype = int))

# plot it
plt.plot(time_true, y_true)

## set up variables
n = 1
p_week = 7
seasonality_prior_scale=1
N = 2

with pm.Model() as m1: 
    # shared 
    t_shared = pm.Data('t_shared', time_true)
    idx_shared = pm.Data('idx_shared', idx_true)
    
    # estimate it? 
    p_1 = pm.Normal("p_1", mu = p, sd = 1)
    week = (2*np.pi*np.arange(1, n+1) / p_1)[:, None]
    
    # creating fourier
    x_week = week * t_shared
    #x_week = c_week * t_shared
    # x_month = c_month * t_shared
    
    x_week_waves = tt.concatenate((tt.cos(x_week), tt.sin(x_week)), axis=0)
    # x_month_waves = tt.concatenate((tt.cos(x_month), tt.sin(x_month)), axis = 0)
    
    # beta one way
    beta_week_waves = []
    for i in range(2): 
        beta_week_waves.append(pm.Normal(f'beta_week_waves_{i}', mu = 0, sd = seasonality_prior_scale, shape = 2*n))
    beta_week_waves=tt.stack(beta_week_waves)
    
    # beta another way
    beta_weeks = pm.Normal("beta_weeks", mu = 0, sd = seasonality_prior_scale, shape = (2*n, N))
    
    pm.math.dot(beta_weeks, t_shared)
    
    
    
    beta_line = pm.Normal('beta_line', mu = 0, sd = 0.5)
    alpha = pm.Normal('alpha', mu = 0, sd = 0.5)
    
    
    ## new try 
    mu = alpha + beta_line * t_shared + pm.math.dot(x_week_waves.T, beta_week_waves)
    
    # sigma 
    sigma = pm.HalfCauchy('sigma', 0.05)
    
    # likelihood 
    y_pred = pm.Normal('y_pred', 
                       mu = mu,
                       sd = sigma,
                       observed = y_true)
beta_week = []