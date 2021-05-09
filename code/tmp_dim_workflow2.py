'''
VMP 3-5-21. 
reproducing this: 
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

# Import radon data
srrs2 = pd.read_csv(pm.get_data("srrs2.dat"))
srrs2.columns = srrs2.columns.map(str.strip)
srrs_mn = srrs2[srrs2.state == "MN"].copy()

# county level
srrs_mn["fips"] = srrs_mn.stfips * 1000 + srrs_mn.cntyfips
cty = pd.read_csv(pm.get_data("cty.dat"))
cty_mn = cty[cty.st == "MN"].copy()
cty_mn["fips"] = 1000 * cty_mn.stfips + cty_mn.ctfips

# more preprocessing 
srrs_mn = srrs_mn.merge(cty_mn[["fips", "Uppm"]], on="fips")
srrs_mn = srrs_mn.drop_duplicates(subset="idnum")
u = np.log(srrs_mn.Uppm).unique()
n = len(srrs_mn)
srrs_mn.head()

# dictionary
srrs_mn.county = srrs_mn.county.map(str.strip)
mn_counties = srrs_mn.county.unique()
counties = len(mn_counties)
county_lookup = dict(zip(mn_counties, range(counties)))

# local copies
county = srrs_mn["county_code"] = srrs_mn.county.replace(county_lookup).values
radon = srrs_mn.activity
srrs_mn["log_radon"] = log_radon = np.log(radon + 0.1).values
floor = srrs_mn.floor.values

# distribution of log radon
srrs_mn.log_radon.hist(bins=25);

# model 1
coords = {"Level": ["Basement", "Floor"], "obs_id": np.arange(floor.size)}

with pm.Model(coords=coords) as pooled_model:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id")
    a = pm.Normal("a", 0.0, sigma=10.0, dims="Level")

    theta = a[floor_idx]
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")

#pm.model_to_graphviz(pooled_model)
RANDOM_SEED = 42
with pooled_model:
    prior_checks = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
    idata_prior = az.from_pymc3(prior=prior_checks)

_, ax = plt.subplots()
idata_prior.prior.plot.scatter(x="Level", y="a", color="k", alpha=0.2, ax=ax)
ax.set_ylabel("Mean log radon level");

# pooled model
with pooled_model:
    pooled_trace = pm.sample(random_seed=RANDOM_SEED)
    pooled_idata = az.from_pymc3(pooled_trace)
az.summary(pooled_idata, round_to=2)

with pooled_model:
    ppc = pm.sample_posterior_predictive(pooled_trace, random_seed=RANDOM_SEED)
    pooled_idata = az.from_pymc3(pooled_trace, posterior_predictive=ppc, prior=prior_checks)

# something
hdi_helper = lambda ds: az.hdi(ds, input_core_dims=[["chain", "draw", "obs_id"]])
hdi_ppc = (
    pooled_idata.posterior_predictive.y.groupby(pooled_idata.constant_data.floor_idx)
    .apply(hdi_helper)
    .y
)
hdi_ppc

# levels?
level_labels = pooled_idata.posterior.Level[pooled_idata.constant_data.floor_idx]
pooled_idata.observed_data = pooled_idata.observed_data.assign_coords(Level=level_labels).sortby(
    "Level"
)

# plot
pooled_means = pooled_idata.posterior.mean(dim=("chain", "draw"))

_, ax = plt.subplots()
pooled_idata.observed_data.plot.scatter(x="Level", y="y", label="Observations", alpha=0.4, ax=ax)

az.plot_hdi(
    [0, 1],
    hdi_data=hdi_ppc,
    fill_kwargs={"alpha": 0.2, "label": "Exp. distrib. of Radon levels"},
    ax=ax,
)

az.plot_hdi(
    [0, 1], pooled_idata.posterior.a, fill_kwargs={"alpha": 0.5, "label": "Exp. mean HPD"}, ax=ax
)
ax.plot([0, 1], pooled_means.a, label="Exp. mean")

ax.set_ylabel("Log radon level")
ax.legend(ncol=2, fontsize=9, frameon=True);

# complete pool 
coords["County"] = mn_counties
with pm.Model(coords=coords) as unpooled_model:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id")
    county_idx = pm.Data("county_idx", county, dims="obs_id")
    a = pm.Normal("a", 0.0, sigma=10.0, dims=("County", "Level"))

    theta = a[county_idx, floor_idx]
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")
pm.model_to_graphviz(unpooled_model)

#### Totally unpooled #### 
coords["County"] = mn_counties
with pm.Model(coords=coords) as unpooled_model:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id")
    county_idx = pm.Data("county_idx", county, dims="obs_id")
    a = pm.Normal("a", 0.0, sigma=10.0, dims=("County", "Level"))

    theta = a[county_idx, floor_idx]
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")
pm.model_to_graphviz(unpooled_model)

with unpooled_model:
    unpooled_idata = pm.sample(return_inferencedata=True, random_seed=RANDOM_SEED)
    
# unpooled idata. 
az.plot_forest(
    unpooled_idata, var_names="a", figsize=(6, 32), r_hat=True, combined=True, textsize=8
);

# unpooled. 
unpooled_means = unpooled_idata.posterior.mean(dim=("chain", "draw"))
unpooled_hdi = az.hdi(unpooled_idata)

# plot
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

n_floor_meas = srrs_mn.groupby("county").sum().floor
uncertainty = unpooled_hdi.a.sel(hdi="higher", Level="Floor") - unpooled_hdi.a.sel(
    hdi="lower", Level="Floor"
)

plt.plot(n_floor_meas, uncertainty, "o", alpha=0.4)
plt.xlabel("Nbr floor measurements in county")
plt.ylabel("Estimates' uncertainty");

## crazy plot
SAMPLE_COUNTIES = (
    "LAC QUI PARLE",
    "AITKIN",
    "KOOCHICHING",
    "DOUGLAS",
    "CLAY",
    "STEARNS",
    "RAMSEY",
    "ST LOUIS",
)

unpooled_idata.observed_data = unpooled_idata.observed_data.assign_coords(
    {
        "County": ("obs_id", mn_counties[unpooled_idata.constant_data.county_idx]),
        "Level": (
            "obs_id",
            np.array(["Basement", "Floor"])[unpooled_idata.constant_data.floor_idx],
        ),
    }
)

fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True, sharex=True)
for ax, c in zip(axes.ravel(), SAMPLE_COUNTIES):
    sample_county_mask = unpooled_idata.observed_data.County.isin([c])

    # plot obs:
    unpooled_idata.observed_data.where(sample_county_mask, drop=True).sortby("Level").plot.scatter(
        x="Level", y="y", ax=ax, alpha=0.4
    )

    # plot both models:
    ax.plot([0, 1], unpooled_means.a.sel(County=c), "b")
    ax.plot([0, 1], pooled_means.a, "r--")

    ax.set_title(c)
    ax.set_xlabel("")
    ax.set_ylabel("Log radon level")
fig.tight_layout();

##### MULTILEVEL #######

### INTERCEPTS ###
with pm.Model(coords=coords) as partial_pooling:
    county_idx = pm.Data("county_idx", county, dims="obs_id")
    # Hyperpriors:
    a = pm.Normal("a", mu=0.0, sigma=10.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)

    # Varying intercepts:
    a_county = pm.Normal("a_county", mu=a, sigma=sigma_a, dims="County")

    # Expected value per county:
    theta = a_county[county_idx]
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")
pm.model_to_graphviz(partial_pooling)

with partial_pooling:
    partial_pooling_idata = pm.sample(tune=2000, return_inferencedata=True, random_seed=RANDOM_SEED)

with pm.Model(coords=coords) as unpooled_bis:
    county_idx = pm.Data("county_idx", county, dims="obs_id")
    a_county = pm.Normal("a_county", 0.0, sigma=10.0, dims="County")

    theta = a_county[county_idx]
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")

    unpooled_idata_bis = pm.sample(tune=2000, return_inferencedata=True, random_seed=RANDOM_SEED)

# crazy plot
N_county = srrs_mn.groupby("county")["idnum"].count().values

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
for ax, idata, level in zip(
    axes,
    (unpooled_idata_bis, partial_pooling_idata),
    ("no pooling", "partial pooling"),
):

    # add variable with x values to xarray dataset
    idata.posterior = idata.posterior.assign_coords({"N_county": ("County", N_county)})
    # plot means
    idata.posterior.mean(dim=("chain", "draw")).plot.scatter(
        x="N_county", y="a_county", ax=ax, alpha=0.9
    )
    ax.hlines(
        partial_pooling_idata.posterior.a.mean(),
        0.9,
        max(N_county) + 1,
        alpha=0.4,
        ls="--",
        label="Est. population mean",
    )

    # plot hdi
    hdi = az.hdi(idata).a_county
    ax.vlines(N_county, hdi.sel(hdi="lower"), hdi.sel(hdi="higher"), color="orange", alpha=0.5)

    ax.set(
        title=f"{level.title()} Estimates",
        xlabel="Nbr obs in county (log scale)",
        xscale="log",
        ylabel="Log radon",
    )
    ax.legend(fontsize=10)
fig.tight_layout();

county
coords
### Varying intercepts 
with pm.Model(coords=coords) as varying_intercept:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id")
    county_idx = pm.Data("county_idx", county, dims="obs_id")
    # Hyperpriors:
    a = pm.Normal("a", mu=0.0, sigma=10.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)

    # Varying intercepts:
    a_county = pm.Normal("a_county", mu=a, sigma=sigma_a, dims="County")
    # Common slope:
    b = pm.Normal("b", mu=0.0, sigma=10.0)

    # Expected value per county:
    theta = a_county[county_idx] + b * floor_idx
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")
pm.model_to_graphviz(varying_intercept)