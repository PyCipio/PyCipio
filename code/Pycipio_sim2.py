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
import Pycipio2 as pc 

##### generate data #####
length = 120
time = np.arange(0, length, 1) 
time_true = np.append(time, time)

# Amplitude of the sine wave is sine of a variable like time
sines1 = np.sin(1.9*time) + np.sin(0.9*time) + 2*np.sin(0.22*time) + np.random.normal(0, 0.1, length)
sines2 = np.sin(2*time) + np.sin(time) + 2*np.sin(0.25*time) + np.random.normal(0, 0.1, length)
coses1 = np.cos(1.9*time) + np.cos(0.9 * time) + 2*np.cos(0.22*time) + np.random.normal(0, 0.1, length)
coses2 = np.sin(2*time) + np.cos(time) + 2*np.cos(0.25*time) + np.random.normal(0, 0.1, length)
line1 = 1 + 0.5 * time + np.random.normal(0, 0.1, length) 
line2 = 0.5 + 0.3 * time + np.random.normal(0, 0.1, length)

y1 = sines1 + coses1 + line1
y2 = sines2 + coses2 + line2

y_true = np.append(y1, y2)
idx_true = np.append(
    np.zeros(length, dtype = int), 
    np.ones(length, dtype = int))

d = pd.DataFrame({
    't': time_true,
    'y': y_true,
    'idx': idx_true
})

sns.lineplot(data = d, x = "t", y = "y", hue = "idx")

##### create class ######
sim = pc.PyCipio(d, time = "t", values = "y", index = "idx", split = 0.7)

##### fit mod: both just 1 #####
## NB: if there is only one argument to mode which does something (multiplicative).
## perhaps we should just set it as a true/false?
sim.fit(p1 = (7, 1), p2 = (30, 1), p1_mode = "additive", p2_mode = "additive")

##### sample #####
sim.sample_mod()

##### plotting #####
sim.plotting()

##### save idata ####
sim.m_idata.to_netcdf("../models/sim2.nc")