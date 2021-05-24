#Imports
from covid19dh import covid19
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns

#New function
##Get data from covid19dh and preprocess
#Level sets granularity: 1 = Countrywise, 2 == statewise, 3 == state and countywise
def get_data(country = "USA", level = 3, start = date(2020,1,1), pr_capita = 100000):

    #Get data function
    x, src = covid19(country, level = level, start = start, verbose = False)
    ##clean up
    #Select columns:
    x = x[[
        'date',
        'id',
        'confirmed',
        'deaths',
        'population',
        'administrative_area_level_1',
        'administrative_area_level_2',
        'administrative_area_level_3']]

    #Add new infected columns and remove (very few NAs - one in the start of each state)
    x["new_infected"] = x.groupby(["administrative_area_level_2"])["confirmed"].diff()
    #Remove NAs
    x = x[x["new_infected"].notna()]
    #Remove peeked values with 3day rolling window 
    x['smoothed_new_infected'] = x.groupby('administrative_area_level_2')['new_infected'].rolling(3).mean().reset_index(0,drop=True)
    #Normalize new infected per capita within state
    x["new_infected_pr_capita"] = (x["smoothed_new_infected"]/x["population"])*pr_capita

    return x

