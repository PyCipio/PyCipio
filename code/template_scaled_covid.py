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

###### Part 1: the data we receive #####
df = get_data(level = 2, start = date(2020,3,6)) #can get more or less data here.
df = df[df["administrative_area_level_2"].isin(["Colorado", "Mississippi", "New York", "Texas"])]
df["t"] = df["date"]
df["y"] = df["new_infected_pr_capita"]
df["idx"] = df["administrative_area_level_2"]
df = df[["idx", "t", "y"]]

## user also gives us names 
time = "t"
values = "y"
index = "idx"
split = .20

###### Part 2: our preprocessing ######
index_codes = "idx_code"
time_codes = "t_code"

## handling idx if missing
if index == "None": 
    df[index] == np.zeros(len(df))
    df[index_codes] == np.zeros(len(df))

## create idx codes
df[index_codes] = pd.Categorical(df[index]).codes

## handling time 
if type(df[time]) != int: 
    df[time_codes] = df.groupby([index]).cumcount()+0

## train / test split 
train, test = f.train_test(df, time, train_size = split)

## variables for test 
t1_test = ((test[time_codes] - df[time_codes].min()) / (df[time_codes].max() - df[time_codes].min())).values
t2_test = np.unique(t1_test)
t3_test = len(t2_test)
y_test = ((test[values] - df[values].min()) / (df[values].max() - df[values].min())).values
idx_test = test[index_codes].values
n_test = len(np.unique(idx_test))

## variables for train
t1_train = ((train[time_codes] - train[time_codes].min()) / (train[time_codes].max() - train[time_codes].min())).values
t2_train = np.unique(t1_train) 
t3_train = len(t2_train)
y_train = ((train[values] - train[values].min()) / (train[values].max() - train[values].min())).values
idx_train = train[index_codes].values
n_train = len(np.unique(idx_train))

##### Part 3: set up periods #######

## NB: we assume that input is in days. 

## common across week & month (I guess)
## NB: deviation might as well just go in on each place then.
divisor = 7
deviation = 0.2

## week 
n_week_components = 4
week_mu = 7
week_sd = week_mu/divisor

## normalize week.
p_week_mu = (week_mu - train[time_codes].min()) / (train[time_codes].max() - train[time_codes].min())
p_week_sd = (week_sd - train[time_codes].min()) / (train[time_codes].max() - train[time_codes].min())
beta_week_sd = deviation
p_week_mu

## month
n_month_components = 4
month_mu = 30
month_sd = month_mu/divisor

## normalize month.
p_month_mu = (month_mu -  train[time_codes].min()) / (train[time_codes].max() - train[time_codes].min())
p_month_sd = (month_sd - train[time_codes].min()) / (train[time_codes].max() - train[time_codes].min())
beta_month_sd = deviation

##### Part 4: Seasonal component #####
def seasonal_component(
    name, 
    name_beta, 
    mu, 
    sd, 
    beta_sd, 
    n_components, 
    shape,
    t2,
    t3):
    
    p = pm.Beta(name, 
                mu = mu, 
                sd = sd, 
                shape = shape)

    period_x = 2*np.pi*np.arange(1, n_components+1)
    period_stack_x = np.stack([period_x for i in range(shape)])
    period_scaled_x = period_stack_x.T / p
    x = tt.reshape(period_scaled_x[:, :, None] * t2, (n_components, shape*t3))
    x_waves = tt.concatenate((tt.cos(x), tt.sin(x)), axis = 0)

    beta_waves = pm.Normal(
        name_beta, 
        mu = 0,
        sd = beta_sd, 
        shape = (2*n_components, shape))

    ### flatten waves
    lst = []
    index_first = 0
    index_second = t3
    for i in range(shape): 
        tmp = pm.math.dot(x_waves.T[index_first:index_second, :], beta_waves[:, i])
        lst.append(tmp)
        index_first += t3
        index_second += t3
    stacked = tt.stack(lst)
    x_flat = tt.flatten(stacked)
    
    return (beta_waves, x_waves, x_flat)


##### Part 5: run the model ######
with pm.Model() as m0: 
    
    # shared 
    t1_shared = pm.Data('t1_shared', t1_train)
    t2_shared = pm.Data('t2_shared', t2_train)
    t3_shared = pm.Data('t3_shared', np.array(t3_train))
    idx_shared = pm.Data('idx_shared', idx_train)
    
    # prepare fourier week
    #seasonal_component(name, name_beta, mu, sd, beta_sd, n_components, shape, time_scaled)
    beta_week_waves, x_week_waves, week_flat = seasonal_component(name = "p_week",
                                                       name_beta = "beta_week_waves",
                                                       mu = p_week_mu,
                                                       sd = p_week_sd,
                                                       beta_sd = beta_week_sd,
                                                       n_components = n_week_components,
                                                       shape = n_train,
                                                       t2 = t2_shared,
                                                       t3 = t3_shared)
    
    beta_month_waves, x_month_waves, month_flat = seasonal_component(name = "p_month",
                                                       name_beta = "beta_month_waves",
                                                       mu = p_month_mu,
                                                       sd = p_month_sd,
                                                       beta_sd = beta_month_sd,
                                                       n_components = n_month_components,
                                                       shape = n_train,
                                                       t2 = t2_shared,
                                                       t3 = t3_shared)
    
    # other priors
    beta_line = pm.Normal('beta_line', mu = 0, sd = 0.3, shape = n_train)
    alpha = pm.Normal('alpha', mu = 0.5, sd = 0.3, shape = n_train)

    mu = alpha[idx_shared] + beta_line[idx_shared] * t1_shared + week_flat * t1_shared + month_flat * t1_shared
    
    # sigma 
    sigma = pm.Exponential('sigma', 1)
    
    # likelihood 
    y_pred = pm.Normal(
        'y_pred', 
        mu = mu,
        sd = sigma,
        observed = y_train)
    
##### Part 6: Sampling ######

## sample prior
with m0:
    prior_pred = pm.sample_prior_predictive(100) # like setting this low. 
    m0_idata = az.from_pymc3(prior=prior_pred)

az.plot_ppc(m0_idata, group="prior")

## convenience function 
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

## sample posterior & predictive
m0_idata = sample_mod(m0)

## plot checks 
az.plot_ppc(m0_idata, num_pp_samples = 100, group = "prior")
az.plot_ppc(m0_idata, num_pp_samples = 100)

## plot trace
az.plot_trace(m0_idata)

###### Part 7: check fit to data #######
m_pred = m0_idata.posterior_predictive.mean(axis = 1)
m_pred_std = m_pred.std(axis = 0)
m_pred = m_pred.mean(axis = 0)