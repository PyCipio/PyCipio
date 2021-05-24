# import packages
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

# load data 
d = pd.read_csv("../data/hh_budget.csv")

# subset (optionally). 
d = d[d["Country"] == "Australia"]

# check data
sns.lineplot(data = d, x = "Year", y = "Wealth")

# specify x, y & grouping variable in orig. data &
# the name that will be used going forward (can be the same). 
y = "Wealth"
x_old = "Year"
idx_old = "Country"
x_new = "Year_idx"
idx_new = "idx"

# get idx for each group (e.g. country) and zero-index for time variable (e.g. year).
d = f.get_idx(d, idx_old, x_old, idx_new, x_new)

# use test/train split function
train, test = f.train_test(d, x_new) # check doc-string.

# run simple model (only one country)
with pm.Model() as m0: 
    
    # set priors
    α = pm.Normal('α', mu = 400, sd = 20)
    β = pm.Normal('β', mu = 20, sd = 10)
    ϵ = pm.HalfCauchy('ϵ', 5)
    
    x_ = pm.Data('m0_x', train[x_new].values) #Year_ as data container.
    
    # y pred 
    y_pred = pm.Normal('y_pred', 
                       mu = α + β * x_, 
                       sd = ϵ, 
                       observed = train[y].values)
    # trace 
    m0_trace = pm.sample(2000, 
                         return_inferencedata = True,
                         target_accept = .99)
    ''' could be relevant at some point.
    m0_prior = pm.sample_prior_predictive(samples=50) # just for model check. 
    m0_ppc = pm.sample_posterior_predictive(m0_trace, 
                                            var_names = ["α", "β", "y_pred"])
    '''

# check trace
az.plot_trace(m0_trace)

'''
Should have both prior predictive checks and posterior predictive checks
before making predictions, but that is left out for now.  
'''

# posterior predictive on new data. 
predictions = f.pp_test(
    m0, 
    m0_trace, 
    {'m0_x': test[x_new].values},
    params = ["α", "β", "y_pred"]
    )

# plot posterior predictive on new data.
f.plot_pp(predictions, train, test, d, x_new, y, idx_old)

# MSE 
mse = f.MSE()