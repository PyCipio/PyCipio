'''
VMP: 
1. two components for one of the n's.
'''

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

##### generate data #####
###### .... ###### 
length = 120
time = np.arange(0, length, 1) 
time_true = np.append(time, time)

# Amplitude of the sine wave is sine of a variable like time
sines1 = np.sin(0.9*time) + 2*np.sin(0.22*time) + np.random.normal(0, 0.2, length)
sines2 = np.sin(2*time) + np.sin(time) + 2*np.sin(0.25*time) + np.random.normal(0, 0.3, length)
coses1 = np.cos(0.9 * time) + 2*np.cos(0.22*time) + np.random.normal(0, 0.2, length)
coses2 = np.sin(2*time) + np.cos(time) + 2*np.cos(0.25*time) + np.random.normal(0, 0.4, length)
line1 = 1 + 0.5 * time + np.random.normal(0, 0.2, length) 
line2 = 0.5 + 0.3 * time + np.random.normal(0, 0.2, length)

y1 = sines1 + coses1 + line1
y2 = sines2 + coses2 + line2

y_true = np.append(y1, y2)
idx_true = np.append(
    np.zeros(length, dtype = int), 
    np.ones(length, dtype = int))

plt.plot(time_true, y_true)

#### parameters ####
n_idx = len(np.unique(idx_true))

## scale stuff
time_true_scaled = (time_true - time_true.min()) / (time_true.max() - time_true.min())
time_scaled = (time - time.min()) / (time.max() - time.min())

## week 
n_week_components = 2
p_week_mu = (7 - time_true.min()) / (time_true.max() - time_true.min())
p_week_sd = (2 - time_true.min()) / (time_true.max() - time_true.min())
beta_week_sd = 0.3

## month
n_month_components = 1
p_month_mu = (30 - time_true.min()) / (time_true.max() - time_true.min())
p_month_sd = (4 - time_true.min()) / (time_true.max() - time_true.min())
beta_month_sd = 0.3

# each individually?
y_true_scaled = (y_true - y_true.min()) / (y_true.max() - y_true.min())
plt.plot(time_true_scaled, y_true_scaled)

#### model #### 
with pm.Model() as m0: 
    
    # shared 
    t_shared = pm.Data('t_shared', time_true_scaled)
    idx_shared = pm.Data('idx_shared', idx_true)
    
    # prepare fourier week
    p_week = pm.Beta("p_week", mu = p_week_mu, sd = p_week_sd, shape = n_idx)
    period_x_week = 2*np.pi*np.arange(1, n_week_components+1)
    period_stack_week = np.stack([period_x_week, period_x_week])
    period_scaled_week = period_stack_week.T / p_week
    x_week = tt.reshape(period_scaled_week[:, :, None] * time_scaled, (n_week_components, n_idx*len(time_scaled)))
    x_week_waves = tt.concatenate((tt.cos(x_week), tt.sin(x_week)), axis = 0)
    
    # prepare fourier month
    p_month = pm.Beta("p_month", mu = p_month_mu, sd = p_month_sd, shape = n_idx)
    period_x_month = 2*np.pi*np.arange(1, n_month_components+1)
    period_stack_month = np.stack([period_x_month, period_x_month])
    period_scaled_month = period_stack_month.T / p_month
    x_month = tt.reshape(period_scaled_month[:, :, None] * time_scaled, (n_month_components, n_idx*len(time_scaled)))
    x_month_waves = tt.concatenate((tt.cos(x_month), tt.sin(x_month)), axis = 0)
    
    # beta week 
    beta_week_waves = pm.Normal('beta_week_waves', mu = 0, 
                                sd = beta_week_sd, shape = (2*n_week_components, n_idx)) 
    beta_month_waves = pm.Normal('beta_month_waves', mu = 0,
                                sd = beta_month_sd, shape = (2*n_month_components, n_idx))
    
    # other priors
    beta_line = pm.Normal('beta_line', mu = 0, sd = 0.3, shape = n_idx)
    alpha = pm.Normal('alpha', mu = 0.5, sd = 0.3, shape = n_idx)
    
    # loop for week waves. 
    week_lst = []
    index_first = 0
    index_second = len(time_scaled)
    for i in range(n_idx): 
        week_tmp = pm.math.dot(x_week_waves.T[index_first:index_second, :], beta_week_waves[:, i])
        week_lst.append(week_tmp)
        index_first += len(time_scaled)
        index_second += len(time_scaled) 
    week_stacked = tt.stack(week_lst)
    week_flat = tt.flatten(week_stacked)
    
    # loop for month waves.
    month_lst = []
    index_first = 0
    index_second = len(time_scaled)
    for i in range(n_idx): 
        month_tmp = pm.math.dot(x_month_waves.T[index_first:index_second, :], beta_month_waves[:, i])
        month_lst.append(month_tmp)
        index_first += len(time_scaled)
        index_second += len(time_scaled)
    month_stacked = tt.stack(month_lst)
    month_flat = tt.flatten(month_stacked)

    mu = alpha[idx_shared] + beta_line[idx_shared] * t_shared + week_flat + month_flat 
    
    # sigma 
    sigma = pm.Exponential('sigma', 1)
    
    # likelihood 
    y_pred = pm.Normal('y_pred', 
                       mu = mu,
                       sd = sigma,
                       observed = y_true_scaled)

## prior sampling 
with m0:
    prior_pred = pm.sample_prior_predictive(100) # like setting this low. 
    m0_idata = az.from_pymc3(prior=prior_pred)

az.plot_ppc(m0_idata, group="prior")

## sample everything 
# convenience function 
def sample_mod(
    model, 
    posterior_draws = 1000, 
    post_pred_draws = 1000,
    prior_pred_draws = 500):
    
    with model: 
        trace = pm.sample(
            return_inferencedata = False, 
            draws = posterior_draws,
            target_accept = .95) # tuning!
        post_pred = pm.sample_posterior_predictive(trace, samples = post_pred_draws)
        prior_pred = pm.sample_prior_predictive(samples = prior_pred_draws)
        m_idata = az.from_pymc3(trace = trace, posterior_predictive=post_pred, prior=prior_pred)
    
    return m_idata

m0_idata = sample_mod(m0)

## plot checks 
az.plot_ppc(m0_idata, num_pp_samples = 100, group = "prior")
az.plot_ppc(m0_idata, num_pp_samples = 100)

## plot trace
az.plot_trace(m0_idata)
az.summary(m0_idata)

## fit to data
m_pred = m0_idata.posterior_predictive.mean(axis = 1)
m_pred = m_pred.mean(axis = 0)

# try to plot it 
plt.plot(time_true_scaled, m_pred["y_pred"])
plt.plot(time_true_scaled, y_true_scaled)

# save it
m0_idata.to_netcdf("../models/VMP3_components.nc")

'''
We have an issue with pm.Beta() as outcome.
The issue is that it requires 0 < beta < 1 -
which we cannot ensure at present with our 
alpha and beta. Even if we force alpha to be
a pm.Beta() (which could make sense), we would
still have the issue that this constraint does
not make sense for beta (the line). 
########### beta outcome? ##########
with pm.Model() as m1: 
    
    # shared 
    t_shared = pm.Data('t_shared', time_true_scaled)
    idx_shared = pm.Data('idx_shared', idx_true)
    
    # prepare fourier week
    p_week = pm.Beta("p_week", mu = p_week_mu, sd = p_week_sd, shape = n_idx)
    period_x_week = 2*np.pi*np.arange(1, n_week_components+1)
    period_stack_week = np.stack([period_x_week, period_x_week])
    period_scaled_week = period_stack_week.T / p_week
    x_week = tt.reshape(period_scaled_week[:, :, None] * time_scaled, (n_week_components, n_idx*len(time_scaled)))
    x_week_waves = tt.concatenate((tt.cos(x_week), tt.sin(x_week)), axis = 0)
    
    # prepare fourier month
    p_month = pm.Beta("p_month", mu = p_month_mu, sd = p_month_sd, shape = n_idx)
    period_x_month = 2*np.pi*np.arange(1, n_month_components+1)
    period_stack_month = np.stack([period_x_month, period_x_month])
    period_scaled_month = period_stack_month.T / p_month
    x_month = tt.reshape(period_scaled_month[:, :, None] * time_scaled, (n_month_components, n_idx*len(time_scaled)))
    x_month_waves = tt.concatenate((tt.cos(x_month), tt.sin(x_month)), axis = 0)
    
    # beta week 
    beta_week_waves = pm.Normal('beta_week_waves', mu = 0, 
                                sd = beta_week_sd, shape = (2*n_week_components, n_idx)) 
    beta_month_waves = pm.Normal('beta_month_waves', mu = 0,
                                sd = beta_month_sd, shape = (2*n_month_components, n_idx))
    
    # other priors
    beta_line = pm.Normal('beta_line', mu = 0, sd = 0.3, shape = n_idx)
    alpha = pm.Normal('alpha', mu = 0.5, sd = 0.3, shape = n_idx)
    
    # loop for week waves. 
    week_lst = []
    index_first = 0
    index_second = len(time_scaled)
    for i in range(n_idx): 
        week_tmp = pm.math.dot(x_week_waves.T[index_first:index_second, :], beta_week_waves[:, i])
        week_lst.append(week_tmp)
        index_first += len(time_scaled)
        index_second += len(time_scaled) 
    week_stacked = tt.stack(week_lst)
    week_flat = tt.flatten(week_stacked)
    
    # loop for month waves.
    month_lst = []
    index_first = 0
    index_second = len(time_scaled)
    for i in range(n_idx): 
        month_tmp = pm.math.dot(x_month_waves.T[index_first:index_second, :], beta_month_waves[:, i])
        month_lst.append(month_tmp)
        index_first += len(time_scaled)
        index_second += len(time_scaled)
    month_stacked = tt.stack(month_lst)
    month_flat = tt.flatten(month_stacked)

    mu = alpha[idx_shared] + beta_line[idx_shared] * t_shared + week_flat + month_flat 
    
    # sigma 
    sigma = pm.Exponential('sigma', 1)
    
    # likelihood 
    y_pred = pm.Beta('y_pred', 
                       mu = mu,
                       sd = sigma,
                       observed = y_true_scaled)
    
with m1:
    prior_pred = pm.sample_prior_predictive(100) # like setting this low. 
    m_idata = az.from_pymc3(prior=prior_pred)

az.plot_ppc(m_idata, group="prior")
'''