'''
https://docs.pymc.io/notebooks/multilevel_modeling.html
'''

# packages
import warnings
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import xarray as xr
from theano import tensor as tt
import seaborn as sns 
import fns as f

# load our data 
d = pd.read_csv("../data/hh_budget.csv")

###### create variables #######

## COUNTRY 
countries = d.Country.unique()
country_len = len(countries)
country_lookup = dict(zip(countries, range(country_len)))
country = d["country_code"] = d.Country.replace(country_lookup).values

## YEARS 
Year_idx = d["Year_idx"] = pd.Categorical(d["Year"]).codes
Years = np.unique(d.Year_idx.values)
Years_list = list(Years) 

### assign to coords. 
coords = {"Year": Years_list, 
          "obs_id": np.arange(Year_idx.size),
          "Country": countries}


###### Complete pooling ###### 
with pm.Model(coords=coords) as m_pool: 
    year_idx = pm.Data("year_idx", Year_idx, dims="obs_id")
    a = pm.Normal("a", 400, sd=40)
    b = pm.Normal("b", 20, sd=5)
    mu = a + b * year_idx 
    sigma = pm.HalfCauchy("sigma", 5)
    y = pm.Normal("y", mu, sigma = sigma, observed = d.Wealth.values, dims = "obs_id")
    
    pool_idata = pm.sample(
        2000, tune = 2000, target_accept = .99,
        return_inferencedata = True
    )
    
az.plot_trace(pool_idata, compact=True)
pm.model_to_graphviz(m_pool)

#### partial pooling (different parameterizations exist) ######
with pm.Model(coords=coords) as m_partial:
    year_idx = pm.Data("year_idx", Year_idx, dims="obs_id")
    country_idx = pm.Data("country_idx", country, dims = "obs_id")
    
    # Hyperpriors:
    a = pm.Normal("a", 400, sd = 20)
    sigma_a = pm.HalfNormal("sigma_a", sd=10)
    b = pm.Normal('b', mu=20, sd=5)
    sigma_b = pm.HalfNormal('sigma_b', sd=10)

    # Varying intercepts & slopes:
    a_country = pm.Normal("a_country", mu=a, sigma=sigma_a, dims="Country") 
    b_country = pm.Normal("b_country", mu=b, sigma=sigma_b, dims="Country")
    
    # Expected value per contry: (mu I guess.)
    mu = a_country[country_idx] + b_country[country_idx] * Year_idx
    
    # Model error:
    sigma = pm.HalfCauchy('sigma', 5)

    y = pm.Normal("y", mu, sigma=sigma, observed=d.Wealth.values, dims = "obs_id")
    
    partial_idata = pm.sample(
        2000, tune=2000, target_accept=0.99, 
        return_inferencedata=True)


# plot trace
az.plot_trace(partial_idata, compact=True, chain_prop={"ls": "-"});
pm.model_to_graphviz(m_partial)

##### Non-pooled ###### 
with pm.Model(coords=coords) as m_unpooled:
    year_idx = pm.Data("year_idx", Year_idx, dims="obs_id")
    country_idx = pm.Data("country_idx", country, dims = "obs_id")
    a = pm.Normal("a", 400, sd=20, dims="Country")
    b = pm.Normal("b", 20, sd=5, dims="Country")

    mu = a[country_idx] + b[country_idx] * year_idx
    sigma = pm.HalfCauchy("sigma", 5)

    y = pm.Normal("y", mu, sigma=sigma, observed=d.Wealth.values, dims="obs_id")
    
    unpooled_idata = pm.sample(
        2000, tune=2000, target_accept=0.99, 
        return_inferencedata=True)

az.plot_trace(unpooled_idata, compact=True, chain_prop={"ls": "-"});
pm.model_to_graphviz(m_unpooled)

###### FITS PLOT FOR PARTIAL POOLING ######
#### not completely tight yet though (colors..)
xvals = xr.DataArray(Years_list, dims="Year", coords={"Year": Years_list})
post = partial_idata.posterior  # alias for readability
avg_a_country = post.a_country.mean(dim=("chain", "draw"))
avg_b_country = post.b_country.mean(dim=("chain", "draw"))
mu = (avg_a_country + avg_b_country * xvals).to_dataset(name="GDP per year")

_, ax = plt.subplots()
mu.plot.scatter(x="Year", y="GDP per year", alpha=0.2, color="k", ax=ax)  # scatter
ax.plot(xvals, mu["GDP per year"].T, "k-", alpha=0.2)
sns.lineplot(data = d, x = "Year_idx", y = "Wealth", hue = "Country")
ax.set_title("What the fuck is this?");

###### PLOT COMPARISON ###### 
#### PREP THE PLOT

### pulling out means & hdi. 
# pooled 
pool_means = pool_idata.posterior.mean(dim=("chain", "draw"))
pool_hdi = az.hdi(pool_idata)

# unpooled
unpooled_means = unpooled_idata.posterior.mean(dim=("chain", "draw"))
unpooled_hdi = az.hdi(unpooled_idata)

# partial
partial_means = partial_idata.posterior.mean(dim=("chain", "draw"))
partial_hdi = az.hdi(partial_idata)


## connect observations to country and year. 
unpooled_idata.observed_data = unpooled_idata.observed_data.assign_coords(
    {
        "Country": ("obs_id", countries[unpooled_idata.constant_data.country_idx]),
        "Year": (
            "obs_id",
            np.array(Years_list)[unpooled_idata.constant_data.year_idx],
        ),
    }
)

## MAKE THE PLOT 
fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharey=True, sharex=True)
for ax, c in zip(axes.ravel(), countries):
    sample_country_mask = unpooled_idata.observed_data.Country.isin([c])

    # plot obs:
    unpooled_idata.observed_data.where(sample_country_mask, drop=True).sortby("Year").plot.scatter(
        x="Year", y="y", ax=ax, alpha=0.4
    )

    # compute means y:
    y_unpooled = np.array(unpooled_means.a.sel(Country = c)) + (np.array(unpooled_means.b.sel(Country = c)) * Years)
    y_partial = np.array(partial_means.a_country.sel(Country = c)) + (np.array(partial_means.b_country.sel(Country = c)) * Years)
    y_pool = np.array(pool_means.a.sel()) + (np.array(pool_means.b.sel()) * Years) # also works without sel() but more flexible.
    
    # plot means x & y
    ax.plot(Years_list, y_unpooled, "b")
    ax.plot(Years_list, y_partial, "r--")
    ax.plot(Years_list, y_pool, "r")
    #ax.plot(Years_list, var_int_slope_means.a.sel(Country=c), "r--")

    ax.set_title(c)
    ax.set_xlabel("")
    ax.set_ylabel("Log radon level")
fig.tight_layout();

### getting predictions out & keeping the workflow?
