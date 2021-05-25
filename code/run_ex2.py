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

d = pd.read_csv("../data/archive/Alcohol_Sales.csv")

## weird quick of the format - have to set it to pd.DateTime
d["DATE"] = pd.to_datetime(d["DATE"])

##### create class ######
sim = pc.PyCipio(d, time = "DATE", values = "S4248SM144NCEN", split = 0.7)

##### fit mod: both just 1 #####
## NB: if there is only one argument to mode which does something (multiplicative).
## perhaps we should just set it as a true/false?
sim.fit(p1 = (4, 4), p2 = (12, 6), p1_mode = "multiplicative", p2_mode = "multiplicative")

##### sample #####
sim.sample_mod()

##### plotting #####
sim.plotting()

### plot training ###
sim.plot_fit_idx()

### plot prediction ###
#sim.predict()

### save idata ###
sim.save_idata("../models/m_ex2_4-4M_12-6M")