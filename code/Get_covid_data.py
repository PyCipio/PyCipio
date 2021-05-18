#Imports
from covid19dh import covid19
from datetime import date
import matplotlib.pyplot as plt

##Get data from covid19dh and preprocess
#Level sets granularity: 1 = Countrywise, 2 == statewise, 3 == state and countywise
def get_data(country = "USA", level = 3, start = date(2020,1,1)):
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
        'administrative_area_level_3',
        'latitude',
        'longitude']]
    
    return x


#New function
##Get data from covid19dh and preprocess
#Level sets granularity: 1 = Countrywise, 2 == statewise, 3 == state and countywise
def get_data(country = "USA", level = 3, start = date(2020,1,1)):

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
    x = x[x["new_infected"].notna()]
    x["new_infected_pr_capita"] = (x["new_infected"]/x["population"])*100000


    #Interpolate accumulated infected numbers - on a few days,
    #numbers are not reported and then aggregated on the following day. This is fixed by averaging accross such cases.  
    
    return x


data = get_data(level = 2, start = date(2020,1,1))

subset = data[data['administrative_area_level_2'].isin(["Florida"])]

plt.plot(subset.date,subset.new_infected)
plt.show()

subset2 = data[data['administrative_area_level_2'].isin(["Florida"])]
