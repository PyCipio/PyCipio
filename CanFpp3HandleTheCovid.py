
#OH SHIT. THis stuff is in R right? i guess i have to save a pickle and open it up in there

from covid19dh import covid19
from datetime import date
from Get_covid_data import get_data

data = get_data(level = 2, start = date(2020,12,12))

data.to_csv("sample_data.csv",index = False, header = True)