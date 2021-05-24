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
import pickle

##### generate data #####

## length
length = 120
time = np.arange(0, length, 1) 
time_true = np.append(time, time)

## reproducibility
random.seed(17)

## sine, cos & lines. 
sines1 = np.sin(1.9*time) + np.sin(0.9*time) + 2*np.sin(0.22*time) + np.random.normal(0, 0.1, length)
sines2 = np.sin(2*time) + np.sin(time) + 2*np.sin(0.25*time) + np.random.normal(0, 0.1, length)
coses1 = np.cos(1.9*time) + np.cos(0.9*time) + 2*np.cos(0.22*time) + np.random.normal(0, 0.1, length)
coses2 = np.sin(2*time) + np.cos(time) + 2*np.cos(0.25*time) + np.random.normal(0, 0.1, length)
line1 = 1 + 0.5 * time + np.random.normal(0, 0.1, length) 
line2 = 0.5 + 0.3 * time + np.random.normal(0, 0.1, length)

## put the decomposed signals together
y1 = sines1 + coses1 + line1
y2 = sines2 + coses2 + line2

## make the data ready 
y_true = np.append(y1, y2)
idx_one = ["group_one" for i in range(length)]
idx_two = ["group_two" for i in range(length)]
idx_true = np.append(idx_one, idx_two)

## dataframe
d = pd.DataFrame({
    't': time_true,
    'y': y_true,
    'idx': idx_true
})

## check the data
sns.lineplot(data = d, x = "t", y = "y", hue = "idx")

## save the data
with open('../data/data_ex1.pickle', 'wb') as f:
    pickle.dump(d, f)