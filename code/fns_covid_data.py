
#%%
#victor
# import packages
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
data = get_data(level = 2, start = date(2020,12,12)) #can get more or less data here.
#group_by introduce lag (new infected from commulative)
data["new_infected"] = data.groupby(["administrative_area_level_2"])["confirmed"].diff()
data = data[data["new_infected"].notna()]
#Create subset from 5 states
subset = data[data['administrative_area_level_2'].isin(["Minnesota", "Florida", "Alabama","Vermont","Washington"])]
# check data
sns.lineplot(data = subset, x = "date", y = "new_infected", hue = "administrative_area_level_2")

#%%
# specify x, y & grouping variable in orig. data &
# the name that will be used going forward (can be the same). 
y = "new_infected"
x_old = "date"
idx_old = "administrative_area_level_2"
x_new = "date_2"
idx_new = "idx"

# get idx for each group (e.g. country) and zero-index for time variable (e.g. year).
d = f.get_idx(subset, idx_old, x_old, idx_new, x_new)

# use test/train split function
train, test = f.train_test(d, x_new) # check doc-string.

# we need N here as well for shape.
N = len(np.unique(train[idx_new]))

#%%
# pooled model
with pm.Model() as m1: 
    
    # hyper-priors
    α_μ = pm.Normal('α_μ', mu=5000, sd=1500)
    α_σ = pm.HalfNormal('α_σ', 2000)
    β_μ = pm.Normal('β_μ', mu=0, sd=100)
    β_σ = pm.HalfNormal('β_σ', sd=50)
    
    # priors
    α = pm.Normal('α', mu=α_μ, sd=α_σ, shape=N)
    β = pm.Normal('β', mu=β_μ, sd=β_σ, shape=N)
    ε = pm.HalfCauchy('ε', 50)
    
    # containers 
    x_ = pm.Data('m1_x', train[x_new].values) #Year_ as data container (can be changed). 
    idx_ = pm.Data('m1_idx', train[idx_new].values)
    
    y_pred = pm.Normal('y_pred',
                       mu=α[idx_] + β[idx_] * x_,
                       sd=ε, observed=train[y].values)
    
    # trace 
    m1_trace = pm.sample(2000, return_inferencedata = True,
                        target_accept = .99)
    
    ''' should be implemented at some point
    m1_prior = pm.sample_prior_predictive(samples=50)
    m1_ppc = pm.sample_posterior_predictive(m1_trace, 
                                            var_names = ["α", "β", "y_pred"]) 
    '''
#%%

# check trace
az.plot_trace(m1_trace)

# posterior predictive on new data. 
predictions = f.pp_test(
    m1, 
    m1_trace, 
    {'m1_x': test[x_new].values, 'm1_idx': test[idx_new].values},
    params = ["α", "β", "y_pred"]
    )

# plot posterior predictive on new data. 
f.plot_pp(predictions, train, test, d, x_new, y, idx_old)


#%%
# non-pooled model
with pm.Model() as m2: 
    
    # set priors
    α = pm.Normal('α', mu=5000, sd=1500, shape = N)
    β = pm.Normal('β', mu=0, sd=100, shape = N)
    ϵ = pm.HalfCauchy('ϵ', 50)
    
    # containers 
    x_ = pm.Data('m2_x', train[x_new].values) #Year_ as data container (can be changed). 
    idx_ = pm.Data('m2_idx', train[idx_new].values)
    
    y_pred = pm.Normal('y_pred',
                       mu=α[idx_] + β[idx_] * x_,
                       sd=ε, observed=train[y].values)
    
    # trace 
    m2_trace = pm.sample(2000, return_inferencedata = True,
                        target_accept = .99)
    
    ''' should be implemented at some point
    m1_prior = pm.sample_prior_predictive(samples=50)
    m1_ppc = pm.sample_posterior_predictive(m1_trace, 
                                            var_names = ["α", "β", "y_pred"]) 
    '''


# %%
az.plot_trace(m2_trace)

# %%


# posterior predictive on new data. 
predictions = f.pp_test(
    m2, 
    m2_trace, 
    {'m2_x': test[x_new].values, 'm2_idx': test[idx_new].values},
    params = ["α", "β", "y_pred"]
    )

# plot posterior predictive on new data. 
f.plot_pp(predictions, train, test, d, x_new, y, idx_old)

