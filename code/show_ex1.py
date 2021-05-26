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
d = pd.read_pickle("../data/data_ex1.pkl")

## instantiate class
sim1 = pc.PyCipio(d, time = "t", values = "y", index = "idx", split = 0.7)

## compile first model: 1 component each ## 
sim1.fit(
    p1 = (7, 1), 
    p2 = (30, 1), 
    p1_mode = "additive", 
    p2_mode = "additive")

## load idata into the class
sim1.load_idata("../models/m_ex1_7-1a_30-1a_20_0.2")

## check that everything is good
sim1.plot_pp(path = "../plots/ex1_first_plot_pp")
sim1.plot_trace(path = '../plots/ex1_first_plot_trace')

## check fit 
sim1.plot_fit_idx(
    ["group_one", "group_two"],
    path = "../plots/ex1_first_plot_fit_idx_all")

sim1.plot_fit_idx(
    ["group_one"],
    path = "../plots/ex1_first_plot_fit_idx_single")

## check predictions 
sim1.plot_predict_idx(
    ["group_one", "group_two"],
    path = "../plots/ex1_first_plot_predict_idx_all")
sim1.plot_predict_idx(
    ["group_one"],
    path = "../plots/ex1_first_plot_predict_idx_single")

## check residuals
sim1.plot_residuals(
    ["group_one", "group_two"],
    path = "../plots/ex1_first_residual_plots_all")
sim1.plot_residuals(
    ["group_one"],
    path = "../plots/ex1_first_residual_plots_single")

## errors
sim1.get_errors(path = "../plots/ex1_first_get_errors")

## compile second model: two components each ##

## load data
d = pd.read_pickle("../data/data_ex1.pkl")

## instantiate class
sim2 = pc.PyCipio(d, time = "t", values = "y", index = "idx", split = 0.7)

## compile first model: 1 component each ## 
sim2.fit(
    p1 = (7, 2), 
    p2 = (30, 2), 
    p1_mode = "additive", 
    p2_mode = "additive")

## load idata into the class
sim2.load_idata("../models/m_ex1_7-2a_30-2a_20_0.2")

## check that everything is good
sim2.plot_pp(path = "../plots/ex1_second_plot_pp")
sim2.plot_trace(path = '../plots/ex1_second_plot_trace')

## check fit 
sim2.plot_fit_idx(
    ["group_one", "group_two"],
    path = "../plots/ex1_second_plot_fit_idx_all")

sim2.plot_fit_idx(
    ["group_one"],
    path = "../plots/ex1_second_plot_fit_idx_single")

## check predictions 
sim2.plot_predict_idx(
    ["group_one", "group_two"],
    path = "../plots/ex1_second_plot_predict_idx_all")
sim2.plot_predict_idx(
    ["group_one"],
    path = "../plots/ex1_second_plot_predict_idx_single")

## check residuals
sim2.plot_residuals(
    ["group_one", "group_two"],
    path = "../plots/ex1_second_residual_plots_all")
sim2.plot_residuals(
    ["group_one"],
    path = "../plots/ex1_second_residual_plots_single")

## errors
sim2.get_errors(path = "../plots/ex1_second_get_errors")

## compile third model: five components each ##
## load data
d = pd.read_pickle("../data/data_ex1.pkl")

## instantiate class
sim3 = pc.PyCipio(d, time = "t", values = "y", index = "idx", split = 0.7)

## compile first model: 1 component each ## 
sim3.fit(
    p1 = (7, 5), 
    p2 = (30, 5), 
    p1_mode = "additive", 
    p2_mode = "additive")

## load idata into the class
sim3.load_idata("../models/m_ex1_7-5a_30-5a_20_0.2")

## check that everything is good
sim3.plot_pp(path = "../plots/ex1_third_plot_pp")
sim3.plot_trace(path = '../plots/ex1_third_plot_trace')

## check fit 
sim3.plot_fit_idx(
    ["group_one", "group_two"],
    path = "../plots/ex1_third_plot_fit_idx_all")

sim3.plot_fit_idx(
    ["group_one"],
    path = "../plots/ex1_third_plot_fit_idx_single")

## check predictions 
sim3.plot_predict_idx(
    ["group_one", "group_two"],
    path = "../plots/ex1_third_plot_predict_idx_all")
sim3.plot_predict_idx(
    ["group_one"],
    path = "../plots/ex1_third_plot_predict_idx_single")

## check residuals
sim3.plot_residuals(
    ["group_one", "group_two"],
    path = "../plots/ex1_third_residual_plots_all")
sim3.plot_residuals(
    ["group_one"],
    path = "../plots/ex1_third_residual_plots_single")

## errors
sim3.get_errors(path = "../plots/ex1_third_get_errors")
