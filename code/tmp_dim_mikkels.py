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
import theano.tensor as tt 
import random
# import functions from fns file 
import fns as f

#Get data
df = get_data(level = 2, start = date(2020,1,1)) #can get more or less data here.

df["new_infected"] = df.groupby(["administrative_area_level_2"])["confirmed"].diff()
df = df[df["administrative_area_level_2"].isin(["Colorado"])]
df = df[df["new_infected"].notna()]

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


## pyMC3
n = 3
p = 10
c = (2 * np.pi * np.arange(1, n + 1) / p)[:, None]
t = train.t.values
y_scaled = train.y_scaled.values
seasonality_prior_scale=10


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

