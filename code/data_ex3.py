# import packages
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
import random
from sklearn.preprocessing import MinMaxScaler
# import functions from fns file 
import fns as f

#Load covid data:
data = get_data(country = "USA", level = 2, start = date(2020,3,6), pr_capita = 100000)
subset = data[data['administrative_area_level_2'].isin(["New York", "Mississippi", "Florida", "Nevada", "Colorado", "California", "Texas"])]
subset = subset[["date", "administrative_area_level_2", "new_infected", "smoothed_new_infected", "new_infected_pr_capita"]]

#Remove NAs:
subset = subset[subset["new_infected"].notna()]
subset = subset[subset["smoothed_new_infected"].notna()]
subset = subset[subset["new_infected_pr_capita"].notna()]

##Check data
#sns.lineplot(data = subset, x = "date", y = "new_infected_pr_capita", hue = "administrative_area_level_2")

#Save as pickle
subset.to_pickle("../data/Example_3_covid.pkl")
