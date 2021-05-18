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

# load data 
d = pd.read_csv("../data/hh_budget.csv")

# check data
sns.lineplot(data = d, x = "Year", y = "Wealth", hue = "Country")

# specify x, y & grouping variable in orig. data &
# the name that will be used going forward (can be the same). 
y = "Wealth"
x_old = "Year"
idx_old = "Country"
x_new = "Year_idx"
idx_new = "idx"

# get idx for each group (e.g. country) and zero-index for time variable (e.g. year).
d = f.get_idx(d, idx_old, x_old, idx_new, x_new)

# use test/train split function
train, test = f.train_test(d, x_new) # check doc-string.

# we need N here as well for shape.
N = len(np.unique(train[idx_new]))

# first model - unpooled: 
with pm.Model() as m_unpooled: 
    
    # set priors
    α = pm.Normal('α', mu = 400, sd = 20, shape = N)
    β = pm.Normal('β', mu = 20, sd = 10, shape = N)
    ϵ = pm.HalfCauchy('ϵ', 5)
    
    # data that we change later (data containers). 
    x_unpooled = pm.Data('x_unpool', train[x_new].values) 
    idx_unpooled = pm.Data('idx_unpool', train[idx_new].values)
    
    # y pred 
    y_pred = pm.Normal('y_pred', 
                       mu = α[idx_unpooled] + β[idx_unpooled] * x_unpooled, 
                       sd = ϵ, 
                       observed = train[y].values)
    
    trace_unpooled = pm.sample(2000, return_inferencedata = True,
                        target_accept = .99)
    
pm.model_to_graphviz(m_unpooled)
az.plot_trace(trace_unpooled)

# pooled model (only intercept). 
with pm.Model() as m_intercept: 
    
    # hyper-priors
    α_μ = pm.Normal('α_μ', mu=400, sd=20)
    α_σ = pm.HalfNormal('α_σ', 10)
    
    # priors
    α = pm.Normal('α', mu=α_μ, sd=α_σ, shape=N)
    β = pm.Normal('β', mu=20, sd=5)
    ε = pm.HalfCauchy('ε', 5)
    
    # containers 
    x_intercept = pm.Data('x_int', train[x_new].values) #Year_ as data container (can be changed). 
    idx_intercept = pm.Data('idx_int', train[idx_new].values)
    
    y_pred = pm.Normal('y_pred',
                       mu=α[idx_intercept] + β * x_intercept,
                       sd=ε, observed=train[y].values)
    
    # trace 
    trace_intercept = pm.sample(2000, return_inferencedata = True,
                        target_accept = .99)

pm.model_to_graphviz(m_intercept)
az.plot_trace(trace_intercept)

# pooled model (intercept & slope)
with pm.Model() as m_pooled: 
    
    # hyper-priors
    α_μ = pm.Normal('α_μ', mu=400, sd=20)
    α_σ = pm.HalfNormal('α_σ', 10)
    β_μ = pm.Normal('β_μ', mu=20, sd=5)
    β_σ = pm.HalfNormal('β_σ', sd=10)
    
    # priors
    α = pm.Normal('α', mu=α_μ, sd=α_σ, shape=N)
    β = pm.Normal('β', mu=β_μ, sd=β_σ, shape=N)
    ε = pm.HalfCauchy('ε', 5)
    
    # containers 
    x_pool = pm.Data('x_pool', train[x_new].values) #does not need to be changed?
    idx_pool = pm.Data('idx_pool', train[idx_new].values)
    
    y_pred = pm.Normal('y_pred',
                       mu=α[idx_pool] + β[idx_pool] * x_pool,
                       sd=ε, observed=train[y].values)
    
    # trace 
    trace_pooled = pm.sample(2000, return_inferencedata = True,
                        target_accept = .99)

pm.model_to_graphviz(m_pooled)
az.plot_trace(trace_pooled)
az.plot_forest(trace_pooled)

# unpooled
unpooled_means = trace_unpooled.posterior.mean(dim=("chain", "draw"))
unpooled_hdi = az.hdi(trace_unpooled)

# simple plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
xticks = np.arange(0, 86, 6)
fontdict = {"horizontalalignment": "right", "fontsize": 10}
for ax, level in zip(axes, ["Basement", "Floor"]):
    unpooled_means_iter = unpooled_means.sel(Level=level).sortby("a")
    unpooled_hdi_iter = unpooled_hdi.sel(Level=level).sortby(unpooled_means_iter.a)
    unpooled_means_iter.plot.scatter(x="County", y="a", ax=ax, alpha=0.8)
    ax.vlines(
        np.arange(counties),
        unpooled_hdi_iter.a.sel(hdi="lower"),
        unpooled_hdi_iter.a.sel(hdi="higher"),
        color="orange",
        alpha=0.6,
    )
    ax.set(title=f"{level.title()} estimates", ylabel="Radon estimate", ylim=(-2, 4.5))
    ax.set_xticks(xticks)
    ax.set_xticklabels(unpooled_means_iter.County.values[xticks], fontdict=fontdict)
    ax.tick_params(rotation=30)
fig.tight_layout();


## plots
fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True, sharex=True)
for ax, c in zip(axes.ravel(), SAMPLE_COUNTIES):
    sample_county_mask = unpooled_idata.observed_data.County.isin([c])

    # plot obs:
    unpooled_idata.observed_data.where(sample_county_mask, drop=True).sortby("Level").plot.scatter(
        x="Level", y="y", ax=ax, alpha=0.4
    )

    # plot models:
    ax.plot([0, 1], unpooled_means.a.sel(County=c), "k:", alpha=0.5, label="No pooling")
    ax.plot([0, 1], pooled_means.a, "r--", label="Complete pooling")

    ax.plot([0, 1], theta["Mean log radon"].sel(County=c), "b", label="Partial pooling")

    ax.set_title(c)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=10)

axes[0, 0].set_ylabel("Log radon level")
axes[1, 0].set_ylabel("Log radon level")
axes[0, 0].legend(fontsize=8, frameon=True), axes[1, 0].legend(fontsize=8, frameon=True)
fig.tight_layout();