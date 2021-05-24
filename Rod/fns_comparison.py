'''
VMP 2-5-21: 
checking the different analyses against each other. 
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

### individual model ### 
model_fpath = "../models/m_individual.pickle"
with open(model_fpath, "rb") as f:
    pickle_individual = pickle.load(f)

### multilevel model ### 
model_fpath = "../models/m_multi.pickle"
with open(model_fpath, "rb") as f: 
    pickle_multi = pickle.load(f)
    
## predictive plots
f.plot_pp(pickle_individual["pp"], train, test, d, x_new, y, idx_old, std = 3)
f.plot_pp(pickle_multi["pp"], train, test, d, x_new, y, idx_old, std = 3)
