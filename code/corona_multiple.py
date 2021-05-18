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
p_month = 30
c_week = (2 * np.pi * np.arange(1, n + 1) / p_week)[:, None]
c_month = (2 * np.pi * np.arange(1, n + 1) / p_month)[:, None]
seasonality_prior_scale=2
## unique for time and idx

print(pm.__version__)


with pm.Model() as m: 
    
    # shared 
    t_shared = pm.Data('t_shared', time_train)
    idx_shared = pm.Data('idx_shared', train_idx)
    
    # creating fourier
    x_week = c_week * t_shared
    x_month = c_month * t_shared
    
    x_week_waves = tt.concatenate((tt.cos(x_week), tt.sin(x_week)), axis=0)
    x_month_waves = tt.concatenate((tt.cos(x_month), tt.sin(x_month)), axis = 0)
    
    # beta
    beta_week_waves = pm.Normal('beta_week_waves', mu = 0, sd = seasonality_prior_scale, shape = (2*n, N)) 
    beta_month_waves = pm.Normal('beta_month_waves', mu = 0, sd = seasonality_prior_scale, shape = (2*n, N))
    beta_line = pm.Normal('beta_line', mu = 0, sd = 0.1, shape = N)
    
    # mu temp
    mu_week_waves = pm.math.dot(x_week_waves.T, beta_week_waves[:, idx_shared]) #[:, idx_shared]
    mu_month_waves = pm.math.dot(x_month_waves.T, beta_month_waves[:, idx_shared]) #[:, idx_shared]
    mu_line = beta_line[idx_shared] * t_shared
    
    #pm.math.dot(mu_week_waves[idx_shared], mu_line[idx_shared])
    # mu = tt.sum(mu_tmp)
    
    mu = mu_week_waves + mu_month_waves + mu_line
    
    # sigma 
    sigma = pm.HalfCauchy('sigma', 0.5)
    
    # likelihood 
    y_pred = pm.Normal('y_pred', 
                       mu = mu,
                       sd = sigma,
                       observed = y_train)
    
# idata
with m: 
    m_idata = pm.sample(return_inferencedata = True,
                        chains = 1,
                        init = "adapt_diag",
                        draws = 500)
    
# ooookay..
az.plot_trace(m_idata)

# predictive
with m: 
    m_pred = pm.sample_posterior_predictive(m_idata,
                                            samples = 500,
                                            var_names = ["y_pred"])

m_pred["y_pred"].shape
y_pred = m_pred["y_pred"].mean(axis = 0) # mean over 400 draws

# plot them 
plt.plot(time_train, y_pred)
plt.plot(time_train, y_train)
plt.show();

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