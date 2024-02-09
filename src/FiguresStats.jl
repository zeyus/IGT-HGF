using Plots
using StatsPlots
using StatsFuns
using ArviZ
using PSIS
using ArviZPythonPlots
using HDF5
using MCMCChains
using MCMCChainsStorage
using Turing
using ArviZPlots

# Load data
hgf_sim_chains = h5open("data/igt_data_95_chains.h5", "r") do file
    read(file["Steingroever2011"], Chains)
end

plot_ppc(hgf_sim_chains)
hgf_sim_chains
# predict(hgf_sim_chains["Steingroever2011"], :u1, 1:100, 1:100)
plot_posterior(hgf_sim_chains; var_names=[
    "action_noise_group_mean",
    "action_noise_group_sd",
    "u_input_noise_group_mean",
    "u_input_noise_group_sd",
    "x_volatility_group_mean",
    "x_volatility_group_sd",
    ])
gcf()

plot_rank(hgf_sim_chains; var_names=[
    "action_noise_group_mean",
    "action_noise_group_sd",
    "u_input_noise_group_mean",
    "u_input_noise_group_sd",
    "x_volatility_group_mean",
    "x_volatility_group_sd",
    ])
gcf()

plot_violin(hgf_sim_chains; var_names=[
    "action_noise_group_mean",
    "action_noise_group_sd",
    ])
gcf()
plot_violin(hgf_sim_chains; var_names=[
    "u_input_noise_group_mean",
    "u_input_noise_group_sd",
    ])
gcf()
plot_violin(hgf_sim_chains; var_names=[
    "x_volatility_group_mean",
    "x_volatility_group_sd",
    ])
gcf()


distplot!(hgf_sim_chains; fill=true, alpha=0.5, var_names=["action_noise_group_mean"], groupname=:prior, label="prior")