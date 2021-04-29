#Imports
from covid19dh import covid19
from datetime import date

#Get data from covid19dh and preprocess
def get_data(country = "USA", level = 3, start = date(2020,1,1)):
    #Get data function
    x, src = covid19(country, level = level, start = start)
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
        'latitude', 'longitude']]
    
    return x

if __name__ == __main__:

