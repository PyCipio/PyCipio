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
import PyCipio3_VMP as pc 
import pickle

## load data
d = pd.read_pickle("../data/data_ex1.pkl")

## instantiate class
sim = pc.PyCipio(d, time = "t", values = "y", index = "idx", split = 0.7)

## compile model
sim.fit(p1 = (7, 1), p2 = (30, 1), p1_mode = "additive", p2_mode = "additive")

## load idata into the class
sim.load_idata("../models/m_ex1_7-1_30-1")

## check that everything is good
sim.plot_pp()
sim.plot_trace()

## check fit 
sim.plot_fit_idx(["group_one", "group_two"])
sim.plot_fit_idx(["group_one"])

## check predictions 
sim.plot_predict_idx(["group_one", "group_two"])
sim.plot_predict_idx(["group_one"])

