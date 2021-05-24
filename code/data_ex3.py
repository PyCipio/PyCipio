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
subset = data[data['administrative_area_level_2'].isin(["Mississippi", "Florida", "Nevada", "Colorado", "California", "Texas"])]
subset = subset[["date", "administrative_area_level_2", "new_infected_pr_capita"]]

#Remove NAs:
subset = subset[subset["new_infected_pr_capita"].notna()]

#Rename to common format:
subset = subset.rename(columns={"date": "t", "new_infected_pr_capita": "y", "administrative_area_level_2": "idx"})

##Check data
sns.lineplot(data = subset, x = "t", y = "y", hue = "idx")

#Save as pickle
subset.to_pickle("../data/data_ex3.pkl")
