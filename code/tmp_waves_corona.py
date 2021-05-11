##### import stuff ###### 
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

###### Emil Roenn = Get data ###### 
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

## take out variables ##
time_train = train.t.values
y_train = train.y_scaled.values
time_test = test.t.values
y_test = test.y_scaled.values

###### Run on own data ######
n = 2
p = 7
c = (2 * np.pi * np.arange(1, n + 1) / p)[:, None]
seasonality_prior_scale=2

with pm.Model() as m: 
    
    # shared 
    t_shared = pm.Data('t_shared', time_train)
    
    # creating fourier
    x_tmp = c * t_shared
    x_waves = tt.concatenate((tt.cos(x_tmp), tt.sin(x_tmp)), axis=0)
    
    # beta
    beta_waves = pm.Normal('beta_waves', mu=0, sd = seasonality_prior_scale, shape = 2*n) #2*n
    beta_line = pm.Normal('beta_line', mu = 0, sd = 1)
    
    # mu temp
    mu_waves = pm.math.dot(x_waves.T, beta_waves) 
    mu_line = beta_line * t_shared
    
    # mu = tt.sum(mu_tmp)
    mu = mu_waves + mu_line
    
    # sigma 
    sigma = pm.HalfCauchy('sigma', 0.5, testval=1)
    
    # likelihood 
    y_pred = pm.Normal('y_pred', 
                       mu = mu,
                       sd = sigma,
                       observed = y_train)
    
# idata
with m: 
    m_idata = pm.sample(return_inferencedata = True,
                         draws = 500)
    
