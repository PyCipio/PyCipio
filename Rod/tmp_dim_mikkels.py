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
n = 1
p = 7
c = (2 * np.pi * np.arange(1, n + 1) / p)[:, None]
t = train.t.values
y_scaled = train.y_scaled.values
seasonality_prior_scale=2

with pm.Model() as m2: 
    
    # shared 
    t_shared = pm.Data('t_shared', t)
    
    # creating fourier
    x_tmp = c * t_shared
    x = tt.concatenate((tt.cos(x_tmp), tt.sin(x_tmp)), axis=0)
    
    # beta
    beta = pm.Normal(f'beta', mu=0, sd = seasonality_prior_scale, shape = (2*n, 1)) #2*n
    
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
    m2_idata = pm.sample(return_inferencedata = True,
                         draws = 400,
                         chains = 1)

az.plot_trace(m2_idata)
m2_idata

# plot y. 
with m2: 
    m2_pred = pm.sample_posterior_predictive(m2_idata,
                                             samples = 400,
                                             var_names = ["beta", "y_pred"])

# check shape 
m2_pred["y_pred"].shape

# mean from some axis?
test = m2_pred["y_pred"].mean(axis = 0) # mean over 400 draws
first = test[0]
second = test[1]

# plot them 
plt.plot(t, first)
plt.plot(t, second)
plt.plot(t, y_scaled)
plt.show();

# plot individual draw 
iter = 3
first = [m2_pred["y_pred"][n, :, 0] for n in range(iter)]
second = [m2_pred["y_pred"][n, 0, :] for n in range(iter)]

for i in range(iter): 
    plt.plot(t, first[i], color = "r", alpha = 0.1)
    plt.plot(t, second[i], color = "b", alpha = 0.1)
plt.plot(t, y_scaled)
plt.show();

##### dummy data ##### 
# Get x values of the sine wave
time = np.arange(0, 20, 0.1);

# Amplitude of the sine wave is sine of a variable like time
sines = np.sin(time) + np.random.normal(0, 0.2, 200)
coses = np.cos(time) + np.random.normal(0, 0.2, 200)

# plot it
plt.plot(t, y_true)
plt.plot(t, sines)
plt.plot(t, coses)

# setup for model
n = 2
p = 7
c = (2 * np.pi * np.arange(1, n + 1) / p)[:, None]
t = time
y_true = sines + coses
seasonality_prior_scale=2

with pm.Model() as m3: 
    
    # shared 
    t_shared = pm.Data('t_shared', t)
    
    # creating fourier
    x_tmp = c * t_shared
    x = tt.concatenate((tt.cos(x_tmp), tt.sin(x_tmp)), axis=0)
    
    # beta
    beta = pm.Normal(f'beta', mu=0, sd = seasonality_prior_scale, shape = (2*n, 1)) #2*n
    
    # mu
    #print(x.shape.eval())
    #print(beta.shape)
    mu_tmp = pm.math.dot(x.T, beta) 
    mu = tt.sum(mu_tmp)
    
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
test = m3_pred["y_pred"].mean(axis = 0) # mean over 400 draws
first = test[0]
second = test[1]

# plot them 
plt.plot(t, first)
plt.plot(t, second)
plt.plot(t, y_true)
plt.show();

# individual draws 
iter = 1
first = [m3_pred["y_pred"][n, :, 0] for n in range(iter)]
second = [m3_pred["y_pred"][n, 0, :] for n in range(iter)]

for i in range(iter): 
    plt.plot(t, first[i], color = "r", alpha = 0.1)
    plt.plot(t, second[i], color = "b", alpha = 0.1)
plt.plot(t, y_true)
plt.show();


##### dummy data 2 ######
# Get x values of the sine wave
time = np.arange(0, 30, 0.1);

# Amplitude of the sine wave is sine of a variable like time
sines = np.sin(time)*2 + np.random.normal(0, 0.2, 300)
coses = np.cos(time)*2 + np.random.normal(0, 0.2, 300)

# setup for model
n = 1
p = 3
c = (2 * np.pi * np.arange(1, n + 1) / p)[:, None]
t = time
y_true = sines + coses
seasonality_prior_scale=2

## train/test
time_train = time[:200]
y_train = y_true[:200]
time_test = time[200:]
y_test = y_true[200:]

with pm.Model() as m4: 
    
    # shared 
    t_shared = pm.Data('t_shared', time_train)
    
    # testing. 
    p = pm.Normal("p", mu = 7, sd = 1)
    c = (2*np.pi*np.arange(1, n + 1) / p)[:, None]
    
    # creating fourier
    x_tmp = c * t_shared
    
    # two x vals 
    x_sine = tt.sin(x_tmp)
    x_cos = tt.cos(x_tmp)
    
    # beta
    
    beta_sine = pm.Normal("beta_sine", mu=0, sd = seasonality_prior_scale) #2*n
    beta_cos = pm.Normal("beta_cos", mu = 0, sd = seasonality_prior_scale)
    
    # mu
    #print(x.shape.eval())
    #print(beta.shape)
    mu = pm.Deterministic("mu", x_sine * beta_sine + x_cos * beta_cos)
    
    # sigma 
    sigma = pm.HalfCauchy('sigma', 0.5, testval=1)
    
    # likelihood 
    y_pred = pm.Normal('y_pred', 
                       mu = mu,
                       sd = sigma,
                       observed = y_train)

# inferencedata.     
with m4: 
    m4_idata = pm.sample(return_inferencedata = True,
                         draws = 500)
    
# posterior predictive. 
az.plot_trace(m4_idata)

# posterior pred.
with m4: 
    m4_pred = pm.sample_posterior_predictive(
        m4_idata, 
        samples = 500, 
        var_names = ["y_pred"]
    )
    

# plot 
# mean from some axis?
y_mean = m4_pred["y_pred"].mean(axis = 0) # mean over 400 draws
y_more_mean = y_mean[0]

# plot them 
plt.plot(time_train, y_more_mean)
plt.plot(time_train, y_train)
plt.show();

## predictions on new data
with m4:
    pm.set_data({"t_shared": time_test})
    m4_new_pred = pm.fast_sample_posterior_predictive(
        m4_idata.posterior
    )

# 
# mean from some axis?
y_mean = m4_new_pred["y_pred"].mean(axis = 0) # mean over 400 draws
y_more_mean = y_mean[0]

# plot them 
plt.plot(time_test, y_more_mean)
plt.plot(time_test, y_test)
plt.show();

