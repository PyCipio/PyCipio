## import packages
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
import PyCipio3 as pc 
import pickle

## load data
d = pd.read_pickle("../data/data_ex1.pkl")

## instantiate class
sim = pc.PyCipio(d, time = "t", values = "y", index = "idx", split = 0.7)

## compile model
sim.fit(p1 = (7, 1), p2 = (30, 1), p1_mode = "additive", p2_mode = "additive")

## load idata into the class
sim.load_idata("../models/m_ex1_7-1_30-1")

## now we can check stuff
sim.plotting()
sim.plot_fit_idx("group_one")

## predictions 
sim.predict()
sim.plot_predict_idx("group_one")





### plot just one group ###
# 1. take out coords
pp = sim.m_idata.posterior_predictive["y_pred"].mean(axis = 0)
g1 = pp.sel(idx = "group_one")
g1.shape

m_pred_std = g1.std(axis = 0).data
m_pred = g1.mean(axis = 0).data
m_pred.shape
idx_shared = sim.m_idata.constant_data.idx_shared
t_shared = sim.m_idata.constant_data.t1_shared

### plot several groups ### 
all_g = pp.sortby(["idx"]).mean(axis = (0, 1)).data

# this is needed
test = pp.data
test2 = np.reshape(test, (1000, 2, 43))
means = test2.mean(axis = (0, 1))
sds = test2.std(axis = (0, 1))

plt.plot(range(0, 43), sds)

idx_onex = ["group_one" for i in range(43)]
idx_twox = ["group_two" for i in range(43)]
idx_truex = np.append(idx_onex, idx_twox)


for i in np.unique(idx_truex):
    all_g[(idx_truex == i)]
    np.mean()


all_g
### plot avg. of all groups ###




# 2. take out posterior predictive 
pp = sim.m_idata.posterior_predictive

# 3. assign coords
pp = pp.assign_coords(idx_shared = sim.idx_train)
pp = pp.assign_coords(t_shared = sim.t1_train)

sim.m_idata.posterior

### predict
sim.predict()


### testing stuff
idx = np.array([0, 0, 1, 1])
val = np.array([0, 1, 2, 3])

val[(idx == 1)]