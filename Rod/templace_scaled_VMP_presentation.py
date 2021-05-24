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

# load the first model
VMP_additive = az.from_netcdf("../models/VMP_additive.nc")
VMP_multiplicative = az.from_netcdf("../models/VMP2_multiplicative.nc")
VMP_components = az.from_netcdf("../models/VMP3_components.nc")

# updating checks
def updating_check(m_idata, n_prior = 50, n_posterior = 50): 
    fig, axes = plt.subplots(nrows = 2)

    az.plot_ppc(m_idata, group = "prior", num_pp_samples = n_prior, ax = axes[0])
    az.plot_ppc(m_idata, num_pp_samples = n_posterior, ax = axes[1])
    plt.draw()

# prior & posterior predictive. 
updating_check(VMP_additive)
updating_check(VMP_multiplicative)
updating_check(VMP_components)

