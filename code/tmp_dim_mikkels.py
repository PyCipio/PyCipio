## import stuff
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
import random
# import functions from fns file 
import fns as f

#Get data
df = get_data(level = 2, start = date(2020,1,1)) #can get more or less data here.

df["new_infected"] = df.groupby(["administrative_area_level_2"])["confirmed"].diff()
df = df[df["administrative_area_level_2"].isin(["Colorado"])]
df = df[df["new_infected"].notna()]

def det_dot(a, b):
    """
    The theano dot product and NUTS sampler don't work with large matrices?
    
    :param a: (np matrix)
    :param b: (theano vector)
    """
    return (a * b[None, :]).sum(axis=-1)

df.reset_index(inplace = True)
df['date'] = pd.to_datetime(df['date'])

## train/test
import fns as f
df["date_idx"] = list(range(len(df.date)))
train, test = f.train_test(df, "date_idx", train_size = .75)

# Scale the data
def scalar(df, df_ref): 
    df['y_scaled'] = df['new_infected'] / df_ref['new_infected'].max()
    df['t'] = (df['date'] - df_ref['date'].min()) / (df_ref['date'].max() - df_ref['date'].min())
    df.reset_index()
    return(df)

# scale both train & test
train = scalar(train, train)
test = scalar(test, train)

t = train.t.values
y_scaled = train.y_scaled.values
n = 3
p = 10

## Fourier
np.random.seed(6)
def fourier_series(t, p=365.25, n=10):
    # 2 pi n / p
    x = 2 * np.pi * np.arange(1, n + 1) / p
    # 2 pi n / p * t
    x = x * t[:, None]
    x = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    return x

## Seasonality
def seasonality_model(m, df, period='yearly', seasonality_prior_scale=10):
    
    if period == 'yearly':
        n = 10
        # rescale the period, as t is also scaled
        p = 365.25 / (df['date'].max() - df['date'].min()).days
        
    if period == "monthly":
        n = 10
        # rescale the period, as t is also scaled
        p = 30.5 / (df['date'].max() - df['date'].min()).days
    else:  # weekly
        n = 3
        # rescale the period, as t is also scaled
        p = 7 / (df['date'].max() - df['date'].min()).days
    x = fourier_series(df['t'], p, n)
    with m:
        beta = pm.Normal(f'beta_{period}', mu=0, sd=seasonality_prior_scale, shape=2 * n)
    return x, beta


t = train.t.values
y_scaled = train.y_scaled.values
n = 3
p = 10


with pm.Model() as m:
    # changepoints_prior_scale is None, so the exponential distribution
    # will be used as prior on \tau.
    #y, A, s = trend_model(m, df['t'], changepoints_prior_scale=None, n_changepoints = 40)
    #x_yearly, beta_yearly = seasonality_model(m, df, 'yearly')
    #t_shared = pm.Data('t_shared', t)
    
    '''
    # 2 pi n / p
    x = 2 * np.pi * np.arange(1, n + 1) / p
    # 2 pi n / p * t
    x = x * t[:, None]
    x = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    '''
    x = fourier_series()
    
    
    sigma = pm.HalfCauchy('sigma', 0.5, testval=1)
    y_pred = pm.Normal('y_pred', 
                       mu = x,
                       sd = sigma,
                       observed = y_scaled)
    #x_monthly, beta_monthly = seasonality_model(m, df, "monthly")
    #x_weekly, beta_weekly = seasonality_model(m, df, 'weekly')
    '''
    x_monthly, beta_monthly = seasonality_model(m, df, "monthly")
    x_weekly, beta_weekly = seasonality_model(m, df, 'weekly')
    
    y += det_dot(x_weekly, beta_weekly) + det_dot(x_monthly, beta_monthly) # + det_dot(x_yearly, beta_yearly)
    
    sigma = pm.HalfCauchy('sigma', 0.5, testval=1)
    obs = pm.Normal('obs', 
                 mu=y, 
                 sd=sigma,
                 observed=df['y_scaled'])
    '''

with m: 
    m_idata = pm.sample(return_inferencedata=True)
    
m_idata

##### Testing stuff ##### 
arr1 = np.array([[5, 6, 7, 8]])
arr2 = np.array([[1, 2, 3, 4]])

import theano
import theano.tensor as tt  
import numpy as np
a = tt.matrix('a')
b = theano.shared(np.array([[3, 4, 5, 6]]))
c = a + b
f = theano.function(inputs = [a], outputs = [c])
output = f(np.array([[3, 4, 5, 6]]))
output

##### testing Mikkel ##### 

def fourier_series(t, p=365.25, n=10):
    # 2 pi n / p
    x = 2 * np.pi * np.arange(1, n + 1) / p
    # 2 pi n / p * t
    x = x * t[:, None]
    x = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    return x

## different versions of t. 
t = train.t.values
t_format = t[:, None]
t_shared = theano.shared(t)
t_format_shared = theano.shared(t_format)

## testing it out. 
p = theano.tensor.scalar('p')
n = theano.tensor.scalar('n')
c = pm.math.dot((2 * np.pi * tt.arange(1, n + 1) / p), t_format_shared)
f = theano.function(inputs = [p, n], outputs = [c])
out = f(365, 10)


## pyMC3
c = (2 * np.pi * np.arange(1, n + 1) / p)[:, None]
t = train.t.values
y_scaled = train.y_scaled.values
n = 3
p = 10
seasonality_prior_scale=10

'''
def det_dot(a, b):
    """
    The theano dot product and NUTS sampler don't work with large matrices?
    
    :param a: (np matrix)
    :param b: (theano vector)
    """
    return (a * b[None, :]).sum(axis=-1)
'''

import theano.tensor as tt
with pm.Model() as m2: 
    
    # shared 
    t_shared = pm.Data('t_shared', t)
    
    # creating fourier
    x_tmp = c * t_shared
    x = tt.concatenate((tt.cos(x_tmp), tt.sin(x_tmp)), axis=0)
    
    # beta
    beta = pm.Normal(f'beta', mu=0, sd = seasonality_prior_scale, shape = (6, 1)) #2*n
    
    # mu
    #print(x.shape.eval())
    #print(beta.shape)
    mu = pm.math.dot(x.T, beta)
    
    # sigma 
    sigma = pm.HalfCauchy('sigma', 0.5, testval=1)
    
    # likelihood 
    y_pred = pm.Normal('y_pred', 
                       mu = mu,
                       sd = sigma,
                       observed = y_scaled)


with m2: 
    m2_idata = pm.sample(return_inferencedata = True)


az.plot_trace(m2_idata)

# plot y. 
with m2: 
    m2_pred = pm.sample_posterior_predictive(m2_idata)

# check shape 
m2_pred["y_pred"].shape

# 

#### in pymc3 #### 
a_train = np.array([1, 2, 3, 4, 5, 6])
b_train = np.array([4, 6, 8, 9, 11, 15])
n = 3
p = 10
c = 2 * np.pi * np.arange(1, n + 1)/p
c.shape
c_shape = c[:, None]
a_train.shape

a_train
c_shape
with pm.Model() as m1: 
    a_shared = pm.Data('a_shared', a_train)
    
    #c = 2 * np.pi * tt.arange(1, n + 1)/p
    
    x = pm.Deterministic("x", c_shape * a_shared)
    
    sigma = pm.HalfCauchy('sigma', 0.5, testval=1)
    
    y_pred = pm.Normal('y_pred', 
                       mu = x,
                       sd = sigma,
                       observed = b_train)

