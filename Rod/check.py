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
from numpy.random import default_rng
rng = default_rng()
## https://oriolabril.github.io/oriol_unraveled/python/arviz/pymc3/xarray/2020/09/22/pymc3-arviz.html
a_ = 2
b_ = -0.4
x_ = np.linspace(0, 10, 31)
year_ = np.arange(2021-len(x_), 2021)
y_ = a_ + b_ * x_ + rng.normal(size=len(x_))

fig, ax = plt.subplots()
ax.plot(x_, y_, "o-")
ax.text(
    0.93, 0.9, r"$y_i = a + bx_i + \mathcal{N}(0,1)$", ha='right', va='top', transform=ax.transAxes, fontsize=18
)

ax.set_xticks(x_[::3])
ax.set_xticklabels(year_[::3])
ax.set_yticks([])
ax.set_xlabel("Year")
ax.set_ylabel("Quantity of interest");

year_
x_

coords = {"year": year_}
with pm.Model(coords=coords) as linreg_model:
    x = pm.Data("x", x_, dims="year")
    
    a = pm.Normal("a", 0, 3)
    b = pm.Normal("b", 0, 2)
    sigma = pm.HalfNormal("sigma", 2)
    
    y = pm.Normal("y", a + b * x, sigma, observed=y_, dims="year")