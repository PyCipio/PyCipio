'''
VMP 30-4-21: 
Testing the functions (from fns.py) on multiple
groups (not pooled). 
'''

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

# check data
sns.lineplot(data = d, x = "Year", y = "Wealth", hue = "Country")

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

# we need N here as well for shape.
N = len(np.unique(train[idx_new]))

# unpooled model
with pm.Model() as m: 
    
    # set priors
    α = pm.Normal('α', mu = 400, sd = 20, shape = N)
    β = pm.Normal('β', mu = 20, sd = 10, shape = N)
    ϵ = pm.HalfCauchy('ϵ', 5)
    
    # data that we change later (data containers). 
    x_ = pm.Data('m_x', train[x_new].values) 
    idx_ = pm.Data('m_idx', train[idx_new].values)
    
    # y pred 
    y_pred = pm.Normal('y_pred', 
                       mu = α[idx_] + β[idx_] * x_, 
                       sd = ϵ, 
                       observed = train[y].values)
    # trace 
    m_trace = pm.sample(2000, return_inferencedata = True)
    
    ''' has to be done at some point
    m2_prior = pm.sample_prior_predictive(samples=50) # just for model check. 
    m2_ppc = pm.sample_posterior_predictive(m2_trace, 
                                            var_names = ["α", "β", "y_pred"]) 
    ''' 
    
# check trace
az.plot_trace(m_trace)

# posterior predictive on new data. 
predictions = f.pp_test(
    m, 
    m_trace, 
    {'m2_x': test[x_new].values, 'm2_idx': test[idx_new].values},
    params = ["α", "β", "y_pred"]
    )

# plot posterior predictive on new data. 
f.plot_pp(predictions, train, test, d, x_new, y, idx_old)

# MSE 
