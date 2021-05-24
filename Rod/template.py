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
time = np.arange(0, 30, 1) 
time_true = np.append([time, time], time)

# Amplitude of the sine wave is sine of a variable like time
sines1 = np.sin(0.3*time) + np.random.normal(0, 0.2, 30)
sines2 = np.sin(time) + np.random.normal(0, 0.3, 30)
sines3 = np.sin(1.5*time) + np.random.normal(0, 0.2, 30)
coses1 = np.cos(0.3 * time) + np.random.normal(0, 0.2, 30)
coses2 = np.cos(time) + np.random.normal(0, 0.4, 30)
coses3 = np.cos(1.5 * time) + np.random.normal(0, 0.2, 30)
line1 = 1 + 0.5 * time + np.random.normal(0, 0.2, 30) 
line2 = 0.5 + 0.3 * time + np.random.normal(0, 0.2, 30)
line3 = 3 + 1 * time + np.random.normal(0, 0.2, 30)

y1 = sines1 + coses1 + line1
y2 = sines2 + coses2 + line2
y3 = sines1 + sines2 + sines3 + coses1 + coses2 + coses3 + line3

y_true = np.append([y1, y2], y3)
idx_true = np.append(
    [np.zeros(30, dtype = int), 
    np.ones(30, dtype = int)],
    np.ones(30, dtype = int) + 1)

plt.plot(time_true, y_true)

#### parameters ####
n_idx = len(np.unique(idx_true))
n_components = 3
p_week_mu = 7
p_week_sd=2
beta_week_sd=1

#### model #### 
with pm.Model() as m0: 
    
    # shared 
    t_shared = pm.Data('t_shared', time_true)
    idx_shared = pm.Data('idx_shared', idx_true)
    
    # prepare fourier 
    p_1_week = pm.Normal("p_1", mu = p_week_mu, sd = p_week_sd, shape = n_idx)
    period_x_week = 2*np.pi*np.arange(1, n_components+1)
    period_stack_week = np.stack([period_x_week, period_x_week, period_x_week])
    period_scaled_week = period_stack_week.T / p_1_week
    x_week = tt.reshape(period_scaled_week[:, :, None] * time, (n_components, n_idx*len(time)))
    x_week_waves = tt.concatenate((tt.cos(x_week), tt.sin(x_week)), axis=0)
    

    # beta
    beta_week_waves = pm.Normal('beta_week_waves', mu = 0, 
                                sd = beta_week_sd, shape = (2*n_components, n_idx)) 
    beta_line = pm.Normal('beta_line', mu = 0, sd = 0.5, shape = n_components)
    alpha = pm.Normal('alpha', mu = 0, sd = 0.5, shape = n_components)
    
    # loop for waves. 
    week_lst = []
    index_first = 0
    index_second = len(time)
    for i in range(n_idx): 
        week_tmp = pm.math.dot(x_week_waves.T[index_first:index_second, :], beta_week_waves[:, i])
        week_lst.append(week_tmp)
        index_first += 30
        index_second += 30
        
    week_stacked = tt.stack(week_lst)
    week_flat = tt.flatten(week_stacked)
    
    #pm.math.dot(test2, alpha[idx_shared])
    #pm.math.dot(test, beta_line[idx_shared])
    mu = alpha[idx_shared] + beta_line[idx_shared] * t_shared + week_flat
    
    # sigma 
    sigma = pm.Exponential('sigma', 1)
    
    # likelihood 
    y_pred = pm.Normal('y_pred', 
                       mu = mu,
                       sd = sigma,
                       observed = y_true)


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