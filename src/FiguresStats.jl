using HDF5, MCMCChains, MCMCChainsStorage, Turing
using Plots, StatsPlots
using ActionModels
using Feather
include("Data.jl")
# using ArviZ, PSIS, ArviZPlots, ArviZPythonPlots
cached_data_file = "./data/IGTdataSteingroever2014/IGTdata.feather"
cached_metadata_file = "./data/IGTdataSteingroever2014/IGTMetadata.h5"
if isfile(cached_data_file)
    @info "Loading cached data..."
    trial_data = Feather.read(cached_data_file)
    pats, n_subj = h5open(cached_metadata_file, "r") do file
        pats = read(file, "pats")
        n_subj = read(file, "n_subj")
        return pats, n_subj
    end
end
# patterns
pats = Dict(
    0 => "no_preference",
    1 => "preference_AB",
    2 => "preference_CD",
    4 => "preference_BD",
    5 => "preference_AB_BD",
    6 => "preference_CD_BD",
    8 => "preference_AC",
    10 => "preference_AB_AC"
)
# patterns
# pats = Dict(
#     6 => "preference_CD_BD",
# )
function TruncatedNormal(μ::Real = 0, σ::Real = 1; check_args::Bool = false, a::Real = 0, b::Real = 1)
    return Truncated(Normal(μ, σ; check_args=check_args), a, b)
end
# priors = Dict(
#     "action_noise" => Multilevel(
#         :subj,
#         TruncatedNormal,
#         ["action_noise_pattern_mean", "action_noise_pattern_sd"]),
#     "action_noise_pattern_mean" => TruncatedNormal(1.0, 1.0),
#     "action_noise_pattern_sd" => TruncatedNormal(0, 0.01),

#     "input_noises" => Multilevel(
#         :subj,
#         TruncatedNormal,
#         ["input_noise_pattern_mean", "input_noise_pattern_sd"]),
#     "input_noise_pattern_sd" => TruncatedNormal(0, 0.01),
#     "input_noise_pattern_mean" => TruncatedNormal(0.0, 1.0),
#     # "volatilities" => Multilevel(
#     #     :subj,
#     #     TruncatedNormal,
#     #     ["volatility_pattern_mean", "volatility_pattern_sd"]),
#     ("x1", "volatility") => Multilevel(
#         :subj,
#         TruncatedNormal,
#         ["volatility_pattern_mean", "volatility_pattern_sd"]),
#     ("x2", "volatility") => Multilevel(
#         :subj,
#         TruncatedNormal,
#         ["volatility_pattern_mean", "volatility_pattern_sd"]),
#     ("x3", "volatility") => Multilevel(
#         :subj,
#         TruncatedNormal,
#         ["volatility_pattern_mean", "volatility_pattern_sd"]),
#     ("x4", "volatility") => Multilevel(
#         :subj,
#         TruncatedNormal,
#         ["volatility_pattern_mean", "volatility_pattern_sd"]),
#     "volatility_pattern_mean" => TruncatedNormal(1.0, 1.0),
#     "volatility_pattern_sd" => TruncatedNormal(0, 0.01),
#     "initial_means" => Multilevel(
#         :subj,
#         TruncatedNormal,
#         ["initial_mean_pattern_mean", "initial_mean_pattern_sd"]),
#     "initial_mean_pattern_mean" => TruncatedNormal(0.0, 1.0),
#     "initial_mean_pattern_sd" => TruncatedNormal(0, 0.01),
#     "drifts" => Multilevel(
#         :subj,
#         TruncatedNormal,
#         ["drift_pattern_mean", "drift_pattern_sd"]),
#     "drift_pattern_mean" => TruncatedNormal(0.0, 1.0),
#     "drift_pattern_sd" => TruncatedNormal(0, 0.01),
#     "autoconnection_strengths" => Multilevel(
#         :subj,
#         TruncatedNormal,
#         ["autoconnection_strength_pattern_mean", "autoconnection_strength_pattern_sd"]),
#     "autoconnection_strength_pattern_mean" => TruncatedNormal(1.0, 1.0),
#     "autoconnection_strength_pattern_sd" => TruncatedNormal(0, 0.01),
#     "initial_precisions" => Multilevel(
#         :subj,
#         TruncatedNormal,
#         ["initial_precision_pattern_mean", "initial_precision_pattern_sd"]),
#     "initial_precision_pattern_mean" => TruncatedNormal(1.0, 1.0),
#     "initial_precision_pattern_sd" => TruncatedNormal(0, 0.01),
# )
# # Load data
# pvldelta_chains = Dict()
# h5open("data/igt_pvldelta_data_chains.h5", "r") do file
#     for pat in keys(pats)
#         g = open_group(file, "pattern_$pat")
#         pvldelta_chains[pat] = read(g, Chains)
#     end
# end

pvldelta_chains = Dict()
h5open("data/igt_pvldelta_data_chains_STAN.h5", "r") do file
    for pat in keys(pats)
        g = open_group(file, "pattern_$pat")
        pvldelta_chains[pat] = read(g, Chains)
    end
end

hgf_chains = Dict()
h5open("data/igt_hgf_by_subj_simplified_multiparam_data_chains.h5", "r") do file
    subj_chains = keys(file)
    for subj in subj_chains
        g = open_group(file, subj)
        hgf_chains[subj] = read(g, Chains)
    end
end
priors_unilevel = Dict(
    "action_noise" => TruncatedNormal(1.0, 0.01),
    "volatilities" => TruncatedNormal(0.5, 1.0),
    "drifts" => TruncatedNormal(0.5, 1.0),
)
keys(hgf_chains)
names(hgf_chains["subj_97"])
plot_parameter_distribution(hgf_chains["subj_97"],priors_unilevel)
plot(hgf_chains["subj_97"])
get_posteriors(hgf_chains["subj_97"])
# # print parameters from each model (we only need a single chain and pattern)
# for pat in keys(pats)
#     println("Pattern: $(pats[pat])")
#     println("PVL Delta")
#     println(names(pvldelta_chains[pat]))
#     println("HGF")
#     println(names(hgf_chains[pat]))
#     break
# end

# for pvl delta, we want to compare the group level differences
# :w′μ, :w′σ, :c'μ, :c'σ, :a'μ, :a'σ, :A'μ, :A'σ, 
# pvldelta_selected_params_sym::Vector{Symbol} = [
#     Symbol("wμ"),
#     Symbol("wσ"),
#     Symbol("cμ"),
#     Symbol("cσ"),
#     Symbol("aμ"),
#     Symbol("aσ"),
#     Symbol("Aμ"),
#     Symbol("Aσ"),
# ]

pvldelta_selected_params_sym::Vector{Symbol} = [
    Symbol("mu_A_pr"),
    Symbol("sd_A"),
    Symbol("mu_w_pr"),
    Symbol("sd_w"),
    Symbol("mu_c_pr"),
    Symbol("sd_c"),
    Symbol("mu_a_pr"),
    Symbol("sd_a"),
]

pvldelta_chains[4][:, pvldelta_selected_params_sym, :]
names(pvldelta_chains[2])

Plots.plot(pvldelta_chains[1][:, pvldelta_selected_params_sym, :])

# Plotting
for pat in keys(pats)
    # PVL Delta
    c = pvldelta_chains[pat][:, pvldelta_selected_params_sym, :]
    StatsPlots.plot(c, legend=false, size=(1200, 1600))
    Plots.savefig("figures/igt_pvldelta_$(replace(pats[pat], " " => "_")).png")
    # break
    # # HGF
    # StatsPlots.plot(hgf_chains[pat], legend=false, size=(1200, 1600))
    # Plots.savefig("figures/igt_hgf_$(replace(pats[pat], " " => "_")).png")
    # break
end
