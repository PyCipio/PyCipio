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

###### Emil Roenn = Get data ###### 
df = get_data(level = 2, start = date(2020,1,1)) #can get more or less data here.

np.unique(df.administrative_area_level_2.values)

df["new_infected"] = df.groupby(["administrative_area_level_2"])["confirmed"].diff()
df = df[df["administrative_area_level_2"].isin(["Colorado", "Mississippi"])]
df = df[df["new_infected"].notna()]
df.reset_index(inplace = True)


## train/test
df["date_idx"] = df.groupby(["administrative_area_level_2"]).cumcount()+0
train, test = f.train_test(df, "date_idx", train_size = .20)

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
time_train = train.date_idx.values
y_train = train.y_scaled.values
time_test = test.date_idx.values
y_test = test.y_scaled.values
train_idx = pd.Categorical(train["administrative_area_level_2"]).codes
test_idx = pd.Categorical(test["administrative_area_level_2"]).codes
N = len(np.unique(train_idx))

###### Run on own data ######
n = 2
p_week = 7
# p_month = 30
c_week = (2 * np.pi * np.arange(1, n + 1) / p_week)[:, None]
# c_month = (2 * np.pi * np.arange(1, n + 1) / p_month)[:, None]
seasonality_prior_scale=0.1
## unique for time and idx

#### the first way we can do it ####
y_train.mean()

# scale time?
time_train_sc = time_train / time_train.max() 
time_train_sc

## scale p. 

# model 
with pm.Model() as m: 
    
    # shared 
    t_shared = pm.Data('t_shared', time_train_sc)
    idx_shared = pm.Data('idx_shared', train_idx)
    
    # estimate it? 
    p_1 = pm.Normal("p_1", mu = 7/time_train.max(), sd = 0.1)
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
    beta_line = pm.Normal('beta_line', mu = 0, sd = 0.1, shape = N)
    alpha = pm.Normal('alpha', mu = 0.2, sd = 0.1, shape = N)
    
    '''
    # mu temp
    mu_week_waves = pm.math.dot(x_week_waves.T, beta_week_waves[:, idx_shared]) #[:, idx_shared]
    # mu_month_waves = pm.math.dot(x_month_waves.T, beta_month_waves[:, idx_shared]) #[:, idx_shared]
    mu_line = beta_line[idx_shared] * t_shared

    #pm.math.dot(mu_week_waves[idx_shared], mu_line[idx_shared])
    # mu = tt.sum(mu_tmp)
    mu = alpha[idx_shared] + mu_line + mu_week_waves #mu_month_waves + mu_line
    '''
    
    ## new try 
    mu = alpha[idx_shared] + beta_line[idx_shared] * t_shared + pm.math.dot(x_week_waves.T, beta_week_waves[:, idx_shared])
    
    # sigma 
    sigma = pm.HalfCauchy('sigma', 0.05)
    
    # likelihood 
    y_pred = pm.Normal('y_pred', 
                       mu = mu,
                       sd = sigma,
                       observed = y_train)
    
# idata
with m: 
    m_idata = pm.sample(return_inferencedata = True,
                        chains = 1,
                        #init = "adapt_diag",
                        draws = 500,
                        target_accept = .95)
    
# ooookay..
az.plot_trace(m_idata)

# predictive
with m: 
    m_pred = pm.sample_posterior_predictive(m_idata,
                                            samples = 500,
                                            var_names = ["y_pred"])


m_pred["y_pred"].shape
y_pred = m_pred["y_pred"].mean(axis = 0) # mean over 400 draws
m_pred["y_pred"].shape
# plot them 
plt.plot(time_train_sc, y_pred)
plt.plot(time_train_sc, y_train)
plt.show();
len(time_train_sc)
# predict 
with m:
    pm.set_data({"t_shared": time_test})
    m_new_pred = pm.fast_sample_posterior_predictive(
        m_idata.posterior
    )

# mean from some axis?
y_mean = m_new_pred["y_pred"].mean(axis = 0) # mean over 400 draws

# plot them 
plt.plot(time_test, y_mean)
plt.plot(time_train, y_train)
plt.plot(time_train, y_pred)
az.plot_hdi(time_train, m_pred["y_pred"], hdi_prob = .89)
az.plot_hdi(time_test, m_new_pred["y_pred"], hdi_prob = .89)
plt.plot(time_test, y_test)
plt.show();




#### on dummy data ##### 
time = np.arange(0, 30, 1) 
time_true = np.append(time, time)

# Amplitude of the sine wave is sine of a variable like time
sines1 = np.sin(time) + np.random.normal(0, 0.2, 30)
sines2 = np.sin(time) + np.random.normal(0, 0.3, 30)
coses1 = np.cos(time) + np.random.normal(0, 0.2, 30)
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
#x_week_waves = np.concatenate((np.cos(x_week), np.sin(x_week)), axis=0)
# model 
idx_true

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
y_preds = m_pred["y_pred"].mean(axis = 0) # mean over 500 draws

# try to plot it 
plt.plot(time_true[:30], y_preds[2, :][:30])
plt.plot(time_true[:30], y_preds[2, :][30:])
plt.plot(time_true[:30], y_preds[:, 15][:30])
plt.plot(time_true[:30], y_preds[:, 15][30:])
plt.plot(time_true[:30], y_true[:30])
plt.plot(time_true[30:], y_true[30:])
plt.show();

more_y = y_preds.mean(axis = 0)
plt.plot(time_true, more_y)

############### COORDS #################

time = np.arange(0, 30, 1) 
time_true = np.append(time, time)

# Amplitude of the sine wave is sine of a variable like time
sines1 = np.sin(time) + np.random.normal(0, 0.2, 30)
sines2 = np.sin(time) + np.random.normal(0, 0.3, 30)
coses1 = np.cos(time) + np.random.normal(0, 0.2, 30)
coses2 = np.cos(time) + np.random.normal(0, 0.4, 30)
line1 = 1 + 0.5 * time + np.random.normal(0, 0.2, 30) 
line2 = 0.5 + 0.3 * time + np.random.normal(0, 0.2, 30)

y1 = sines1 + coses1 + line1
y2 = sines2 + coses2 + line2

y_true = np.append(y1, y2)
idx_true = np.append(np.zeros(30, dtype = int), np.ones(30, dtype = int))

# plot it
plt.plot(time_double, y_true)

## set up variables
n = 1
p_week = 7
seasonality_prior_scale=1
N = 2
## scale p. 

coords = {"idx": np.unique(idx_true),
          "x": np.unique(time_true), 
          "obs_id": np.arange(idx_true.size)}

with pm.Model(coords=coords) as m1: 
    
    # shared 
    t_shared = pm.Data('t_shared', time_true, dims = "obs_id")
    idx_shared = pm.Data('idx_shared', idx_true, dims = "obs_id")

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
    beta_week_waves = pm.Normal('beta_week_waves', mu = 0, sd = seasonality_prior_scale, dims = ("idx_shared, ")) 
    # beta_month_waves = pm.Normal('beta_month_waves', mu = 0, sd = seasonality_prior_scale, shape = (2*n, N))

    beta_line = pm.Normal('beta_line', mu = 0, sd = 0.5, dims = "idx")
    alpha = pm.Normal('alpha', mu = 0, sd = 0.5, dims = "idx")
    

    ## new try 
    mu = alpha[idx_shared] + beta_line[idx_shared] * t_shared + pm.math.dot(x_week_waves.T, beta_week_waves[:, idx_shared])
    
    # sigma 
    sigma = pm.HalfCauchy('sigma', 0.05)
    
    # likelihood 
    y_pred = pm.Normal('y_pred', 
                       mu = mu,
                       sd = sigma,
                       observed = y_true,
                       dims = "obs_id")
    
# sample 
with m1: 
    m1_idata = pm.sample(return_inferencedata = True,
                        chains = 1,
                        #init = "adapt_diag",
                        draws = 500,
                        target_accept = .95)    
    

# predict
with m1:
    pm.set_data({"t_shared": time_true,
                 "idx_shared": idx_true})
    partial_pred = pm.fast_sample_posterior_predictive(
        m1_idata.posterior
    )
    az.from_pymc3_predictions(
        partial_pred, idata_orig=m1_idata, inplace=True
    )
    
# assign coords
y_coorded = m1_idata.predictions.assign_coords(
    t_shared = time_true, 
    idx_shared = idx_true)
y_coorded


############ one at a time ################

# Get x values of the sine wave
time = np.arange(0, 30, 0.1);

# Amplitude of the sine wave is sine of a variable like time
sines = np.sin(time) + np.random.normal(0, 0.2, 300)
coses = np.cos(time) + np.random.normal(0, 0.2, 300)
line = 0.5 * time + np.random.normal(0, 0.2, 300)
y = sines + coses + line

# plot it
plt.plot(time, y)
plt.plot(time, sines)
plt.plot(time, coses)
plt.plot(time, line)

# setup for model
n = 2
p_week = 6.5
#c = (2 * np.pi * np.arange(1, n + 1) / p)[:, None]
seasonality_prior_scale=2

# train/test
time_train = time[:200]
y_train = y[:200]
time_test = time[200:]
y_test = y[200:]


with pm.Model() as m5: 
    
    # shared 
    t_shared = pm.Data('t_shared', time_train)
    
    # estimate it? 
    p_1 = pm.Normal("p_1", mu = p_week, sd = 1)
    week = (2*np.pi*np.arange(1, n+1) / p_1)[:, None]
    
    # creating fourier
    x_week = week * t_shared
    
    # creating fourier
    x_tmp = c * t_shared
    x_waves = tt.concatenate((tt.cos(x_tmp), tt.sin(x_tmp)), axis=0)
    
    # alpha
    alpha = pm.Normal("alpha", mu = 0, sd = 1)
    
    # beta
    beta_waves = pm.Normal('beta_waves', mu=0, sd = seasonality_prior_scale, shape = 2*n) #2*n
    beta_line = pm.Normal('beta_line', mu = 0, sd = 1)
    
    # mu temp
    #mu_waves = pm.math.dot(x_waves.T, beta_waves) 
    #mu_line = beta_line * t_shared
    
    # mu = tt.sum(mu_tmp)
    mu = alpha + beta_line * t_shared + pm.math.dot(x_waves.T, beta_waves)
    
    # sigma 
    sigma = pm.HalfCauchy('sigma', 0.5, testval=1)
    
    # likelihood 
    y_pred = pm.Normal('y_pred', 
                       mu = mu,
                       sd = sigma,
                       observed = y_train)

# idata
with m5: 
    m5_idata = pm.sample(return_inferencedata = True,
                         draws = 500)

az.plot_trace(m5_idata)
m5_idata


######### COORDS #############

