# colorado, mississippi, nevada, new jersey, new york
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

## 
time = np.arange(0, 30, 1) 
time_true = np.append(time, time)

# Amplitude of the sine wave is sine of a variable like time
sines1 = np.sin(0.5*time) + np.random.normal(0, 0.2, 30)
sines2 = np.sin(time) + np.random.normal(0, 0.3, 30)
coses1 = np.cos(0.5 * time) + np.random.normal(0, 0.2, 30)
coses2 = np.cos(time) + np.random.normal(0, 0.4, 30)
line1 = 1 + 0.5 * time + np.random.normal(0, 0.2, 30) 
line2 = 0.5 + 0.3 * time + np.random.normal(0, 0.2, 30)

y1 = sines1 + coses1 + line1
y2 = sines2 + coses2 + line2

y_true = np.append(y1, y2)
idx_true = np.append(np.zeros(30, dtype = int), np.ones(30, dtype = int))

# plot it
plt.plot(time_true, y_true)

## set up variables
n = 1
p_week = 7
seasonality_prior_scale=1
N = 2
## scale p. 
idx_unique = np.array([0, 1])


## 
with pm.Model() as m1: 
    
    # shared 
    t_shared = pm.Data('t_shared', time_true)
    idx_shared = pm.Data('idx_shared', idx_true)
    
    # estimate it? 
    p_1 = pm.Normal("p_1", mu = p_week, sd = 1)
    week = (2*np.pi*np.arange(1, n+1) / p_1)[:, None]
    
    # creating fourier
    x_week = week * t_shared
    #x_week = c_week * t_shared
    # x_month = c_month * t_shared
    
    x_week_waves = tt.concatenate((tt.cos(x_week), tt.sin(x_week)), axis=0)
    # x_month_waves = tt.concatenate((tt.cos(x_month), tt.sin(x_month)), axis = 0)
    
    # beta
    #beta_week_waves = pm.Normal('beta_week_waves', mu = 0, sd = seasonality_prior_scale, shape = (2*n, N)) 
    # beta_month_waves = pm.Normal('beta_month_waves', mu = 0, sd = seasonality_prior_scale, shape = (2*n, N))
    beta_line = pm.Normal('beta_line', mu = 0, sd = 0.5, shape = N)
    alpha = pm.Normal('alpha', mu = 0, sd = 0.5, shape = N)
    
    ## new try 
    mu = alpha[idx_shared] + beta_line[idx_shared] * t_shared # + pm.math.dot(x_week_waves.T, beta_week_waves[:, idx_shared])
    
    # sigma 
    sigma = pm.HalfCauchy('sigma', 0.05)
    
    # likelihood 
    y_pred = pm.Normal('y_pred', 
                       mu = mu,
                       sd = sigma,
                       observed = y_true)

# idata
with m1: 
    m_idata = pm.sample(return_inferencedata = True,
                        chains = 1,
                        #init = "adapt_diag",
                        draws = 500,
                        target_accept = .95)
    
# check the parameters. 
az.plot_trace(m_idata)

# predictive
with m1: 
    m_pred = pm.sample_posterior_predictive(m_idata,
                                            samples = 500,
                                            var_names = ["y_pred"])

m_pred["y_pred"].shape

#### 
idx_true
n = 2
with pm.Model() as m2: 
    
    # shared 
    t_shared = pm.Data('t_shared', time_true)
    idx_shared = pm.Data('idx_shared', idx_true)
    
    # estimate it? 
    p_1 = pm.Normal("p_1", mu = p_week, sd = 2.5)
    week = (2*np.pi*np.arange(1, n+1) / p_1)[:, None]
    
    # creating fourier
    x_week = week * t_shared
    #x_week = c_week * t_shared
    # x_month = c_month * t_shared
    
    x_week_waves = tt.concatenate((tt.cos(x_week), tt.sin(x_week)), axis=0)
    # x_month_waves = tt.concatenate((tt.cos(x_month), tt.sin(x_month)), axis = 0)
    
    # beta
    beta_week_waves = pm.Normal('beta_week_waves', mu = 0, sd = seasonality_prior_scale, shape = 2*n) 
    # beta_month_waves = pm.Normal('beta_month_waves', mu = 0, sd = seasonality_prior_scale, shape = (2*n, N))
    beta_line = pm.Normal('beta_line', mu = 0, sd = 0.5, shape = N)
    alpha = pm.Normal('alpha', mu = 0, sd = 0.5, shape = N)
    
    #weekly = beta_week_waves[:, idx_shared]
    #pm.math.dot(beta_week_waves, x_week_waves.T)
    #test = pm.math.dot(x_week_waves.T, beta_week_waves)
    #pm.math.dot(test, alpha)
    ## new try 
    #waves = pm.math.dot(x_week_waves.T, beta_week_waves)
    #test2 = test[:, idx_shared]
    
    #pm.math.dot(test, beta_line[idx_shared])
    mu = alpha[idx_shared] + beta_line[idx_shared] * t_shared + pm.math.dot(x_week_waves.T, beta_week_waves)
    
    # sigma 
    sigma = pm.HalfCauchy('sigma', 0.05)
    
    # likelihood 
    y_pred = pm.Normal('y_pred', 
                       mu = mu,
                       sd = sigma,
                       observed = y_true)
    
with m2: 
    m2_trace = pm.sample(return_inferencedata = True,
                        chains = 1,
                        draws = 500,
                        target_accept = .95)

az.plot_trace(m2_trace)

# predictive
with m2: 
    m_pred = pm.sample_posterior_predictive(m2_trace,
                                            samples = 500,
                                            var_names = ["y_pred"])
m_pred["y_pred"].shape

y_preds = m_pred["y_pred"].mean(axis = 0) # mean over 500 draws

# try to plot it 
plt.plot(time_true, y_preds)
plt.plot(time_true, y_true)

plt.plot(time_true[:30], y_preds[2, :][:30])
plt.plot(time_true[:30], y_preds[2, :][30:])
plt.plot(time_true[:30], y_preds[:, 15][:30])
plt.plot(time_true[:30], y_preds[:, 15][30:])
plt.plot(time_true[:30], y_true[:30])
plt.plot(time_true[30:], y_true[30:])


###### for loop ##### 
with pm.Model() as m2: 
    
    # shared 
    t_shared = pm.Data('t_shared', time_true)
    idx_shared = pm.Data('idx_shared', idx_true)
    
    # estimate it? 
    p_1 = pm.Normal("p_1", mu = p_week, sd = 2.5)
    week = (2*np.pi*np.arange(1, n+1) / p_1)[:, None]
    
    # creating fourier
    x_week = week * t_shared
    #x_week = c_week * t_shared
    # x_month = c_month * t_shared
    
    x_week_waves = tt.concatenate((tt.cos(x_week), tt.sin(x_week)), axis=0)
    # x_month_waves = tt.concatenate((tt.cos(x_month), tt.sin(x_month)), axis = 0)
    
    # beta
    beta_week_waves = pm.Normal('beta_week_waves', mu = 0, sd = seasonality_prior_scale, shape = (2*n, N)) 
    # beta_month_waves = pm.Normal('beta_month_waves', mu = 0, sd = seasonality_prior_scale, shape = (2*n, N))
    beta_line = pm.Normal('beta_line', mu = 0, sd = 0.5, shape = N)
    alpha = pm.Normal('alpha', mu = 0, sd = 0.5, shape = N)
    
    #weekly = beta_week_waves[:, idx_shared]
    #pm.math.dot(beta_week_waves, x_week_waves.T)
    #test = pm.math.dot(x_week_waves.T, beta_week_waves)
    #pm.math.dot(test, alpha)
    ## new try 
    #waves = pm.math.dot(x_week_waves.T, beta_week_waves)
    #test2 = test[:, idx_shared]
    
    placeholder = []
    nn = 0
    nnn = 30
    for i in range(N): 
        tmp = pm.math.dot(x_week_waves.T[nn:nnn, :], beta_week_waves[:, i])
        placeholder.append(tmp)
        nn += 30
        nnn += 30
        
    test = tt.stack(placeholder)
    test2 = tt.flatten(test)
    
    #pm.math.dot(test2, alpha[idx_shared])
    #pm.math.dot(test, beta_line[idx_shared])
    mu = alpha[idx_shared] + beta_line[idx_shared] * t_shared + test2
    
    # sigma 
    sigma = pm.HalfCauchy('sigma', 0.05)
    
    # likelihood 
    y_pred = pm.Normal('y_pred', 
                       mu = mu,
                       sd = sigma,
                       observed = y_true)
    
with m2: 
    m2_trace = pm.sample(return_inferencedata = True,
                        chains = 1,
                        draws = 500,
                        target_accept = .95)
    
az.plot_trace(m2_trace)

with m2: 
    m_pred = pm.sample_posterior_predictive(m2_trace,
                                            samples = 500,
                                            var_names = ["y_pred"])

y_preds = m_pred["y_pred"].mean(axis = 0)


plt.plot(time_true, y_preds)
plt.plot(time_true, y_true)



n = 2
p_week = 7
seasonality_prior_scale=1
N = 2

###### flexible p ######
with pm.Model() as m3: 
    
    # shared 
    t_shared = pm.Data('t_shared', time_true)
    idx_shared = pm.Data('idx_shared', idx_true)
    
    # estimate it? 
    p_1 = pm.Normal("p_1", mu = p_week, sd = 2.5, shape = N)
    week = (2*np.pi*np.arange(1, n+1) / p_1)[:, None]
    
    # creating fourier
    x_week = week * t_shared
    #x_week = c_week * t_shared
    # x_month = c_month * t_shared
    
    x_week_waves = tt.concatenate((tt.cos(x_week), tt.sin(x_week)), axis=0)
    # x_month_waves = tt.concatenate((tt.cos(x_month), tt.sin(x_month)), axis = 0)
    
    # beta
    beta_week_waves = pm.Normal('beta_week_waves', mu = 0, sd = seasonality_prior_scale, shape = (2*n, N)) 
    # beta_month_waves = pm.Normal('beta_month_waves', mu = 0, sd = seasonality_prior_scale, shape = (2*n, N))
    beta_line = pm.Normal('beta_line', mu = 0, sd = 0.5, shape = N)
    alpha = pm.Normal('alpha', mu = 0, sd = 0.5, shape = N)
    
    #weekly = beta_week_waves[:, idx_shared]
    #pm.math.dot(beta_week_waves, x_week_waves.T)
    #test = pm.math.dot(x_week_waves.T, beta_week_waves)
    #pm.math.dot(test, alpha)
    ## new try 
    #waves = pm.math.dot(x_week_waves.T, beta_week_waves)
    #test2 = test[:, idx_shared]
    
    placeholder = []
    nn = 0
    nnn = 30
    for i in range(N): 
        tmp = pm.math.dot(x_week_waves.T[nn:nnn, :], beta_week_waves[:, i])
        placeholder.append(tmp)
        nn += 30
        nnn += 30
        
    test = tt.stack(placeholder)
    test2 = tt.flatten(test)
    
    #pm.math.dot(test2, alpha[idx_shared])
    #pm.math.dot(test, beta_line[idx_shared])
    mu = alpha[idx_shared] + beta_line[idx_shared] * t_shared + test2
    
    # sigma 
    sigma = pm.HalfCauchy('sigma', 0.05)
    
    # likelihood 
    y_pred = pm.Normal('y_pred', 
                       mu = mu,
                       sd = sigma,
                       observed = y_true)
    
with m3: 
    m3_trace = pm.sample(return_inferencedata = True,
                        chains = 1,
                        draws = 500,
                        target_accept = .95)

az.plot_trace(m3_trace)

with m3: 
    m_pred = pm.sample_posterior_predictive(m3_trace,
                                            samples = 500,
                                            var_names = ["y_pred"])

m_pred["y_pred"].shape
preds = m_pred["y_pred"].mean(axis = 0)

plt.plot(time_true, y_true)
plt.plot(time_true, preds)

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

n = 3
p_week = 7
seasonality_prior_scale=1
N = 3
plt.plot(time_true, y_true)

###### flexible p ######
with pm.Model() as m4: 
    
    # shared 
    t_shared = pm.Data('t_shared', time_true)
    idx_shared = pm.Data('idx_shared', idx_true)
    
    # prepare fourier 
    p_1 = pm.Normal("p_1", mu = p_week, sd = 2.5, shape = N)
    test = 2*np.pi*np.arange(1, n+1)
    testy = np.stack([test, test, test])
    tasty = testy.T / p_1
    x_week = tt.reshape(tasty[:, :, None] * time, (n, N*len(time)))
    x_week_waves = tt.concatenate((tt.cos(x_week), tt.sin(x_week)), axis=0)
    

    # beta
    beta_week_waves = pm.Normal('beta_week_waves', mu = 0, sd = seasonality_prior_scale, shape = (2*n, N)) 
    beta_line = pm.Normal('beta_line', mu = 0, sd = 0.5, shape = N)
    alpha = pm.Normal('alpha', mu = 0, sd = 0.5, shape = N)
    
    # loop for waves. 
    placeholder = []
    nn = 0
    nnn = 30
    for i in range(N): 
        tmp = pm.math.dot(x_week_waves.T[nn:nnn, :], beta_week_waves[:, i])
        placeholder.append(tmp)
        nn += 30
        nnn += 30
        
    test = tt.stack(placeholder)
    test2 = tt.flatten(test)
    
    #pm.math.dot(test2, alpha[idx_shared])
    #pm.math.dot(test, beta_line[idx_shared])
    mu = alpha[idx_shared] + beta_line[idx_shared] * t_shared + test2
    
    # sigma 
    sigma = pm.HalfCauchy('sigma', 0.05)
    
    # likelihood 
    y_pred = pm.Normal('y_pred', 
                       mu = mu,
                       sd = sigma,
                       observed = y_true)
    
with m4: 
    m4_trace = pm.sample(return_inferencedata = True,
                        chains = 1,
                        draws = 500,
                        target_accept = .95)

az.plot_trace(m4_trace)

with m4: 
    m_pred = pm.sample_posterior_predictive(m4_trace,
                                            samples = 500,
                                            var_names = ["y_pred"])

m_pred["y_pred"].shape
preds = m_pred["y_pred"].mean(axis = 0)

plt.plot(time_true, y_true)
plt.plot(time_true, preds)


#### CLEAN DATA ####
###### .... ###### 
time_one = 100
time = np.arange(0, time_one, 1) 
time_true = np.append([time, time], time)

# Amplitude of the sine wave is sine of a variable like time
sines1 = 5 * np.sin(0.3*time) + np.random.normal(0, 0.1, time_one)
sines2 = 2 * np.sin(time) + np.random.normal(0, 0.1, time_one)
sines3 = 3 * np.sin(1.5*time) + np.random.normal(0, 0.1, time_one)
#coses1 = np.cos(0.3 * time) + np.random.normal(0, 0.1, 30)
#coses2 = np.cos(time) + np.random.normal(0, 0.1, 30)
#coses3 = np.cos(1.5 * time) + np.random.normal(0, 0.1, 30)
line1 = 1 + 0.5 * time + np.random.normal(0, 0.1, time_one) 
line2 = 0.5 + 0.3 * time + np.random.normal(0, 0.1, time_one)
line3 = 3 + 1 * time + np.random.normal(0, 0.1, time_one)

y1 = sines1 + line1
y2 = sines2 + line2
y3 = sines1 + sines3 + line3

y_true = np.append([y1, y2], y3)
idx_true = np.append(
    [np.zeros(time_one, dtype = int), 
    np.ones(time_one, dtype = int)],
    np.ones(time_one, dtype = int) + 1)

plt.plot(time_true, y_true)

n = 3
p_week = 7
seasonality_prior_scale=1
N = 3


###### flexible p ######
with pm.Model() as m4: 
    
    # shared 
    t_shared = pm.Data('t_shared', time_true)
    idx_shared = pm.Data('idx_shared', idx_true)
    
    # estimate it? 
    ## our p
    '''
    p_1 = pm.Normal("p_1", mu = p_week, sd = 2.5, shape = N)
    weeks = []
    for i in range(N): 
        week = (2*np.pi*np.arange(1, n+1) / p_1[i])
        weeks.append(week)

    print(weeks)
    test = np.array(weeks).T[:, :, None]
    x_week = np.reshape(test * time, (n, N*len(time)))
    '''
    p_1 = pm.Normal("p_1", mu = p_week, sd = 2, shape = N)
    test = 2*np.pi*np.arange(1, n+1)
    testy = np.stack([test, test, test])
    tasty = testy.T / p_1
    x_week = tt.reshape(tasty[:, :, None] * time, (n, N*time_one))
        
    # 
    x_week_waves = tt.concatenate((tt.cos(x_week), tt.sin(x_week)), axis=0)
    # x_month_waves = tt.concatenate((tt.cos(x_month), tt.sin(x_month)), axis = 0)
    
    # beta
    beta_week_waves = pm.Normal('beta_week_waves', mu = 0, sd = seasonality_prior_scale, shape = (2*n, N)) 
    # beta_month_waves = pm.Normal('beta_month_waves', mu = 0, sd = seasonality_prior_scale, shape = (2*n, N))
    beta_line = pm.Normal('beta_line', mu = 0, sd = 0.5, shape = N)
    alpha = pm.Normal('alpha', mu = 0, sd = 0.5, shape = N)
    
    # 
    placeholder = []
    nn = 0
    nnn = time_one
    for i in range(N): 
        tmp = pm.math.dot(x_week_waves.T[nn:nnn, :], beta_week_waves[:, i])
        placeholder.append(tmp)
        nn += time_one
        nnn += time_one
        
    test = tt.stack(placeholder)
    test2 = tt.flatten(test)
    
    #pm.math.dot(test2, alpha[idx_shared])
    #pm.math.dot(test, beta_line[idx_shared])
    mu = alpha[idx_shared] + beta_line[idx_shared] * t_shared + test2
    
    # sigma 
    sigma = pm.HalfCauchy('sigma', 0.5)
    
    # likelihood 
    y_pred = pm.Normal('y_pred', 
                       mu = mu,
                       sd = sigma,
                       observed = y_true)
    
with m4: 
    m4_trace = pm.sample(return_inferencedata = True,
                        chains = 1,
                        draws = 500,
                        target_accept = .95)

az.plot_trace(m4_trace)

with m4: 
    m_pred = pm.sample_posterior_predictive(m4_trace,
                                            samples = 500,
                                            var_names = ["y_pred"])

m_pred["y_pred"].shape
preds = m_pred["y_pred"].mean(axis = 0)

plt.plot(time_true, y_true)
plt.plot(time_true, preds)