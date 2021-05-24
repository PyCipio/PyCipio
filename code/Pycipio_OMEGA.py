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
length = 60
time = np.arange(0, length, 1) 
time_true = np.append(time, time)

# Amplitude of the sine wave is sine of a variable like time
sines1 = np.sin(1.9*time) + np.sin(0.9*time) + 2*np.sin(0.22*time) + np.random.normal(0, 0.1, length)
sines2 = np.sin(2*time) + np.sin(time) + 2*np.sin(0.25*time) + np.random.normal(0, 0.1, length)
coses1 = np.cos(1.9*time) + np.cos(0.9*time) + 2*np.cos(0.22*time) + np.random.normal(0, 0.1, length)
coses2 = np.sin(2*time) + np.cos(time) + 2*np.cos(0.25*time) + np.random.normal(0, 0.1, length)
line1 = 1 + 0.5 * time + np.random.normal(0, 0.1, length) 
line2 = 0.5 + 0.3 * time + np.random.normal(0, 0.1, length)

y1 = sines1 + coses1 + line1
y2 = sines2 + coses2 + line2

y_true = np.append(y1, y2)

idx_one = ["group_one" for i in range(length)]
idx_two = ["group_two" for i in range(length)]
idx_true = np.append(idx_one, idx_two)

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
sim.fit(p1 = (7, 2), p2 = (30, 1), p1_mode = "additive", p2_mode = "additive")

##### sample #####
sim.sample_mod()

##### plotting #####
## issue: 1970 from somewhere.
sim.plotting()

### plot training ###
## issue: one better than the other
sim.plot_train_idx("group_one")

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