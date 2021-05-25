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

## load data
d = pd.read_pickle("../data/data_ex3.pkl")

## check data
sns.lineplot(data = d, x = "t", y = "y", hue = "idx")

##### create class ######
sim = pc.PyCipio(d, time = "t", values = "y", index = "idx", split = 0.7)

##### fit mod: both just 1 #####
## NB: if there is only one argument to mode which does something (multiplicative).
## perhaps we should just set it as a true/false?
sim.fit(p1 = (7, 1), p2 = (75, 1), p1_mode = "multiplicative", p2_mode = "additive")

##### sample #####
sim.sample_mod()

##### plotting #####
sim.plot_trace()
sim.plot_pp()

### plot training ###
sim.plot_fit_idx(["Texas", "California", "Florida", "Nevada"])

### plot predict ###
sim.plot_predict_idx(["Texas", "California", "Florida", "Nevada"])

### save idata ###
sim.save_idata("../models/m_ex3_7-1_75-1")




