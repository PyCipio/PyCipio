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
import pickle

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

# fit model 
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
    
# sample from model 
with m: 
    m_trace = pm.sample(2000, return_inferencedata = True, target_accept = .99)

# check trace
az.plot_trace(m_trace)

# posterior predictive on new data. 
predictions = f.pp_test(
    m, 
    m_trace, 
    {'m_x': test[x_new].values, 'm_idx': test[idx_new].values},
    params = ["α", "β", "y_pred"]
    )

# plot posterior predictive on new data. 
f.plot_pp(predictions, train, test, d, x_new, y, idx_old)

# save the trace & related stuff. 
# make function?
model_fpath = "../models/m_individual.pickle"
with open(model_fpath, 'wb') as buff:
    pickle.dump({'model': m, 
                 'trace': m_trace, 
                 'pp': predictions,
                 'X_shared': x_,
                 'idx_shared': idx_}, buff)
    
## MSE
def MSE(train, y_colname, post_pred, y_pred_name, num_idx = 1): 
    """get mean squared error (MSE). 

    Args:
        y_true (np.array): true y vector (test data).
        post_pred ([type]): [description]
        y_pred_name ([type]): [description]
        num_idx (int, optional): Number of groups. Defaults to 1.

    Returns:
        [type]: [description]
    """    
    
    ''' 
    y_true: true y values
    y_pred: predicted y values 
    y_pred_name: name of outcome in model. 
    num_idx: number of groups
    '''
    
    # setting up needed variables to index. 
    y_true = train[y_colname].values
    _, total = post_pred[y_pred_name].shape
    num_ypred = int(total/num_idx)
    
    # initialize the list & rolling
    MSE = []
    num_ypred_rolling = num_ypred
    
    # loop over number of idx
    for i in range(num_idx):
        # get current y_pred & y_true
        y_pred_tmp = post_pred[y_pred_name][:, num_ypred_rolling-num_ypred:num_ypred_rolling].mean(axis = 0)
        y_true_tmp = y_true[num_ypred_rolling-num_ypred:num_ypred_rolling]
        
        # compute MSE 
        MSE_tmp = np.square(np.subtract(y_true_tmp, y_pred_tmp)).mean() 
        
        # store MSE in list
        MSE.append(MSE_tmp) 
        
        # add to rolling window
        num_ypred_rolling += num_ypred

    return MSE 


# perhaps a dictionary instead?
mse = MSE(train, "Wealth", predictions, "y_pred", num_idx = 4)

# save stuff to cross-reference
