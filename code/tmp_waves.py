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

##### fitting sine waves (check) ###### 
# Get x values of the sine wave
time = np.arange(0, 20, 0.1);

# Amplitude of the sine wave is sine of a variable like time
sines = np.sin(time) + np.random.normal(0, 0.2, 200)
coses = np.cos(time) + np.random.normal(0, 0.2, 200)
y_true = sines + coses

# plot it
plt.plot(time, y_true)
plt.plot(time, sines)
plt.plot(time, coses)

# setup for model
n = 2
p = 6.5
c = (2 * np.pi * np.arange(1, n + 1) / p)[:, None]
seasonality_prior_scale=2

with pm.Model() as m3: 
    
    # shared 
    t_shared = pm.Data('t_shared', time)
    
    # creating fourier
    x_tmp = c * t_shared
    x = tt.concatenate((tt.cos(x_tmp), tt.sin(x_tmp)), axis=0)
    
    # beta
    beta = pm.Normal(f'beta', mu=0, sd = seasonality_prior_scale, shape = 2*n) #2*n
    
    # mu
    #print(x.shape.eval())
    #print(beta.shape)
    mu = pm.math.dot(x.T, beta) 
    # mu = tt.sum(mu_tmp)
    
    # sigma 
    sigma = pm.HalfCauchy('sigma', 0.5, testval=1)
    
    # likelihood 
    y_pred = pm.Normal('y_pred', 
                       mu = mu,
                       sd = sigma,
                       observed = y_true)

# idata
with m3: 
    m3_idata = pm.sample(return_inferencedata = True,
                         draws = 500)

az.plot_trace(m3_idata)

# predictive
with m3: 
    m3_pred = pm.sample_posterior_predictive(m3_idata,
                                             samples = 400,
                                             var_names = ["beta", "y_pred"])

# plot 
# mean from some axis?
m3_pred["y_pred"].shape
y_pred = m3_pred["y_pred"].mean(axis = 0) # mean over 400 draws

# plot them 
plt.plot(time, y_pred)
plt.plot(time, y_true)
plt.show();


####### adding a component ######
##### fitting sine waves (check) ###### 
# Get x values of the sine wave
time = np.arange(0, 30, 0.1);

# Amplitude of the sine wave is sine of a variable like time
sines = np.sin(time) + np.random.normal(0, 0.2, 300)
coses = np.cos(time) + np.random.normal(0, 0.2, 300)
line = 0.5 * time + np.random.normal(0, 0.2, 300)
y = sines + coses + line

# plot it
plt.plot(time, y_true)
plt.plot(time, sines)
plt.plot(time, coses)
plt.plot(time, line)

# setup for model
n = 2
p = 6.5
c = (2 * np.pi * np.arange(1, n + 1) / p)[:, None]
seasonality_prior_scale=2

# train/test
time_train = time[:200]
y_train = y[:200]
time_test = time[200:]
y_test = y[200:]

with pm.Model() as m4: 
    
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
with m4: 
    m4_idata = pm.sample(return_inferencedata = True,
                         draws = 500)

az.plot_trace(m4_idata)

# predictive
with m4: 
    m4_pred = pm.sample_posterior_predictive(m4_idata,
                                             samples = 400,
                                             var_names = ["y_pred"])

# plot 
# mean from some axis?
m4_pred["y_pred"].shape
y_pred = m4_pred["y_pred"].mean(axis = 0) # mean over 400 draws

# plot them 
plt.plot(time_train, y_pred)
plt.plot(time_train, y_train)
plt.show();

# predict 
with m4:
    pm.set_data({"t_shared": time_test})
    m4_new_pred = pm.fast_sample_posterior_predictive(
        m4_idata.posterior
    )

# mean from some axis?
y_mean = m4_new_pred["y_pred"].mean(axis = 0) # mean over 400 draws

# plot them 
plt.plot(time_test, y_mean)
plt.plot(time_test, y_test)
plt.show();