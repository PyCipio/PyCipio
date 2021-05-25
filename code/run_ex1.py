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
import PyCipio as pc 
import pickle

##### fit mod 1: both n = 1 #####
## load data
d = pd.read_pickle("../data/data_ex1.pkl")

## check data
sns.lineplot(data = d, x = "t", y = "y", hue = "idx")

##### create class ######
sim1 = pc.PyCipio(d, time = "t", values = "y", index = "idx", split = 0.7)

sim1.fit(
    p1 = (7, 1), 
    p2 = (30, 1), 
    p1_mode = "additive", 
    p2_mode = "additive")

##### sample #####
sim1.sample_mod()

##### plotting #####
### updating checks and trace ###
sim1.plot_pp()
sim1.plot_trace()

### plot fit ###
sim1.plot_fit_idx(["group_one", "group_two"])

### plot pred ###
sim1.plot_predict_idx(["group_one", "group_two"])

### plot resid ###
sim1.plot_residuals(["group_one", "group_two"])

### get error ###
sim1.get_errors()

### save idata ###
sim1.save_idata("../models/m_ex1_7-1a_30-1a")

##### fit mod 2: both n = 2 #####
## load data
d = pd.read_pickle("../data/data_ex1.pkl")

##### create class ######
sim2 = pc.PyCipio(d, time = "t", values = "y", index = "idx", split = 0.7)

sim2.fit(
    p1 = (7, 2), 
    p2 = (30, 2), 
    p1_mode = "additive", 
    p2_mode = "additive")

##### sample #####
sim2.sample_mod()

##### plotting #####
### updating checks and trace ###
sim2.plot_pp()
sim2.plot_trace()

### plot fit ###
sim2.plot_fit_idx(["group_one", "group_two"])

### plot pred ###
sim2.plot_predict_idx(["group_one", "group_two"])

### plot resid ###
sim2.plot_residuals(["group_one", "group_two"])

### get error ###
sim2.get_errors()

### save idata ###
sim2.save_idata("../models/m_ex1_7-2a_30-2a")

##### fit mod 3: both n = 5 #####
## load data
d = pd.read_pickle("../data/data_ex1.pkl")

##### create class ######
sim3 = pc.PyCipio(d, time = "t", values = "y", index = "idx", split = 0.7)

sim3.fit(
    p1 = (7, 5), 
    p2 = (30, 5), 
    p1_mode = "additive", 
    p2_mode = "additive",
    divisor = 20)

##### sample #####
sim3.sample_mod()

##### plotting #####
### updating checks and trace ###
sim3.plot_pp()
sim3.plot_trace()

### plot fit ###
sim3.plot_fit_idx(["group_one", "group_two"])

### plot pred ###
sim3.plot_predict_idx(["group_one", "group_two"])

### plot resid ###
sim3.plot_residuals(["group_one", "group_two"])

### get error ###
sim3.get_errors()

### save idata ###
sim3.save_idata("../models/m_ex1_7-5a_30-5a")