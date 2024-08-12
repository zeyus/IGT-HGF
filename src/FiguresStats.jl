using HDF5, MCMCChains, MCMCChainsStorage, Turing
using Turing: AbstractMCMC
using Plots, StatsPlots
using ActionModels
using Feather
using ArviZ
using ArviZ: PSIS
using CategoricalArrays
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
pat_map = Dict(
    2 => "C + D (Good)",
    1 => "A + B (Bad)",
    4 => "B + D (Infrequent loss)",
    8 => "A + C (Frequent loss)",
    5 => "B", # -> B
    10 => "C", # -> C
    6 => "D", # -> D
    0 => "No Preference",
    # 16 => "preference_AD",
    # 17 => "preference_AD_AB", # -> A not C
    # 18 => "preference_AD_CD", # -> D
    # 21 => "preference_AD_BD_AB",
    # 24 => "preference_AD_AC", # -> A
    # 32 => "preference_BC",
    # 38 => "preference_CD_BC", # -> B
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
    for pat in pats
        g = open_group(file, "pattern_$pat")
        pvldelta_chains[pat] = read(g, Chains)
    end
end
hhgf_selected_params_sym::Vector{Symbol} = [
    # Symbol("action_noise"),
    Symbol("action_noise_pattern_mean"),
    # Symbol("action_noise_pattern_sd"),
    # Symbol("volatilities"),
    Symbol("volatility_pattern_mean"),
    # Symbol("volatility_pattern_sd"),
    # Symbol("drifts"),
    Symbol("drift_pattern_mean"),
    # Symbol("drift_pattern_sd"),
]
hhgf_chains = Dict()
h5open("data/igt_hgf_multilevel_multiparam_data_chains.h5", "r") do file
    for pat in pats
        g = open_group(file, "pat_$pat")
        hhgf_chains[pat] = read(g, Chains)
    end
end


hgf_chains = Dict()
h5open("data/igt_hgf_FIX_subj_simplified_multiparam_data_chains.h5", "r") do file
    subj_chains = keys(file)
    for subj in subj_chains
        g = open_group(file, subj)
        hgf_chains[subj] = read(g, Chains)
    end
end
priors_unilevel = Dict(
    "action_noise" => TruncatedNormal(0.5, 1.00; a = 0, b = 20),
    "volatilities" => TruncatedNormal(0.5, 1.0),
    "drifts" => TruncatedNormal(0.5, 1.0),
)
keys(hgf_chains)
names(hgf_chains["subj_97"])
plot_parameter_distribution(hgf_chains["subj_97"],priors_unilevel)
plot(hgf_chains["subj_97"])
get_posteriors(hgf_chains["subj_97"])

# now we group all of the subjects for each pattern
# we will then compare the group level differences
hgf_chains_grouped = Dict()
for (pat, label) in pat_map
    subj_ids = unique(trial_data[trial_data[!, :choice_pattern] .== pat, :subj])
    str_subj_ids = string.(subj_ids)
    str_subj_ids = ["subj_$id" for id in str_subj_ids]
    subj_chains = [hgf_chains[id] for id in str_subj_ids]
    hgf_chains_grouped[label] = chainscat(subj_chains...)
end
# plot(hgf_chains_grouped["preference_AB"])
# plot(hgf_chains_grouped["preference_CD"])
# plot(hgf_chains_grouped["no_preference"])

# plot_parameter_distribution(hgf_chains_grouped["preference_AB"],priors_unilevel)
# plot_parameter_distribution(hgf_chains_grouped["preference_CD"],priors_unilevel)
# plot_parameter_distribution(hgf_chains_grouped["no_preference"],priors_unilevel)

name_map = Dict(
    "action_noise" => "Action Noise",
    "volatilities" => "Volatility",
    "drifts" => "Drift",
)
priors_unilevel_renamed = Dict(
    "Action Noise" => priors_unilevel["action_noise"],
    "Volatility" => priors_unilevel["volatilities"],
    "Drift" => priors_unilevel["drifts"],
)


p = plot_parameter_distribution(
    replacenames(hgf_chains_grouped["C + D (Good)"], name_map), priors_unilevel_renamed
)
plot(p, size=(1200, 1200), plot_title="C + D (Good)", thickness_scaling = 1.5, legend=false)

# save the plot
Plots.savefig("figures/igt_hgf_C_D_Good.png")

p = plot_parameter_distribution(replacenames(hgf_chains_grouped["A + B (Bad)"], name_map), priors_unilevel_renamed)
plot(p, size=(1200, 1200), plot_title="A + B (Bad)", thickness_scaling = 1.5, legend=false)

# save the plot
Plots.savefig("figures/igt_hgf_A_B_Bad.png")

p = plot_parameter_distribution(replacenames(hgf_chains_grouped["B + D (Infrequent loss)"], name_map), priors_unilevel_renamed)
plot(p, size=(1200, 1200), plot_title="B + D (Infrequent loss)", thickness_scaling = 1.5, legend=false)

# save the plot
Plots.savefig("figures/igt_hgf_B_D_Infrequent_loss.png")

p = plot_parameter_distribution(replacenames(hgf_chains_grouped["A + C (Frequent loss)"], name_map), priors_unilevel_renamed)
plot(p, size=(1200, 1200), plot_title="A + C (Frequent loss)", thickness_scaling = 1.5, legend=false)

# save the plot
Plots.savefig("figures/igt_hgf_A_C_Frequent_loss.png")

p = plot_parameter_distribution(replacenames(hgf_chains_grouped["B"], name_map), priors_unilevel_renamed)
plot(p, size=(1200, 1200), plot_title="B", thickness_scaling = 1.5, legend=false)

# save the plot
Plots.savefig("figures/igt_hgf_B.png")

p = plot_parameter_distribution(replacenames(hgf_chains_grouped["C"], name_map), priors_unilevel_renamed)
plot(p, size=(1200, 1200), plot_title="C", thickness_scaling = 1.5, legend=false)

# save the plot
Plots.savefig("figures/igt_hgf_C.png")

p = plot_parameter_distribution(replacenames(hgf_chains_grouped["D"], name_map), priors_unilevel_renamed)
plot(p, size=(1200, 1200), plot_title="D", thickness_scaling = 1.5, legend=false)

# save the plot
Plots.savefig("figures/igt_hgf_D.png")

p = plot_parameter_distribution(replacenames(hgf_chains_grouped["No Preference"], name_map), priors_unilevel_renamed)
plot(p, size=(1200, 1200), plot_title="No Preference", thickness_scaling = 1.5, legend=false)

# save the plot
Plots.savefig("figures/igt_hgf_No_Preference.png")



# plot deck choices for each pattern on a single plot
grouped_trial_data = groupby(trial_data, [:choice_pattern, :subj])
choice_proportion_by_subject = combine(
    grouped_trial_data,
    AsTable(
        :choice,
    ) => (
        (x,) -> (
            choice_prop = [
                sum(x.choice .== 1) / length(x.choice),
                sum(x.choice .== 2) / length(x.choice),
                sum(x.choice .== 3) / length(x.choice),
                sum(x.choice .== 4) / length(x.choice),
            ],
            choice = [
                "A",
                "B",
                "C",
                "D",
            ],
            count = [
                sum(x.choice .== 1),
                sum(x.choice .== 2),
                sum(x.choice .== 3),
                sum(x.choice .== 4),
            ],
            # choice_prop_A = sum(x.choice .== 1) / length(x.choice),
            # choice_prop_B = sum(x.choice .== 2) / length(x.choice),
            # choice_prop_C = sum(x.choice .== 3) / length(x.choice),
            # choice_prop_D = sum(x.choice .== 4) / length(x.choice),
        )
    ) => AsTable,
    :choice_pattern => (x -> pat_map[x[1]]) => :choice_pattern_lbl,
)
choice_proportion_by_subject[!, :choice_pattern_lbl] = categorical(choice_proportion_by_subject[!, :choice_pattern_lbl])

levels!(choice_proportion_by_subject.choice_pattern_lbl, [
    pat_map[2],
    pat_map[1],
    pat_map[4],
    pat_map[8],
    pat_map[5],
    pat_map[10],
    pat_map[6],
    pat_map[0],
])

sort!(choice_proportion_by_subject, [:choice_pattern_lbl, :subj])

groupedboxplot(
    choice_proportion_by_subject[!, :choice],
    choice_proportion_by_subject[!, :choice_prop],
    group=choice_proportion_by_subject[!, :choice_pattern_lbl],
    bar_position=:dodge, bar_width=0.5,
    bar_spacing=0,
    xlabel="Deck Choice",
    ylabel="Proportion of Choices",
    title="Participant Deck Choice Proportions by Pattern",
    size=(1800, 1400),
    color_palette = :Set1_8,
    thickness_scaling = 2.0,
    linewidth = 1.0,
    markerstrokewidth = 0.5,
    margin = (10, :px),
)
# add number of samples to each bar (doesn't work)
choice_counts_by_subject = groupby(choice_proportion_by_subject, [:choice_pattern, :choice])
choice_counts_by_subject = combine(
    choice_counts_by_subject,
    :count => sum => :count,
    :choice_prop => median,
    :choice_pattern_lbl => first,
)



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

pvldelta_selected_params_sym::Vector{Symbol} = [
    Symbol("mu_w_pr"),
    Symbol("mu_c_pr"),
    Symbol("sd_w"),
    Symbol("sd_c"),
]

priors_pvldelta = Dict{String, Any}(
    "mu_A_pr" => Normal(0, 1),
    "mu_w_pr" => Normal(0, 1),
    "mu_a_pr" => Normal(0, 1),
    "mu_c_pr" => Normal(0, 1),
    "sd_A" => Uniform(1, 1.5),
    "sd_w" => Uniform(1, 1.5),
    "sd_a" => Uniform(1, 1.5),
    "sd_c" => Uniform(1, 1.5),
)




# now for the STAN model pvldelta

name_map_pvldelta = Dict(
    "mu_A_pr" => "A Mean",
    "sd_A" => "A SD",
    "mu_w_pr" => "w Mean",
    "sd_w" => "w SD",
    "mu_c_pr" => "c Mean",
    "sd_c" => "c SD",
    "mu_a_pr" => "a Mean",
    "sd_a" => "a SD",
)

priors_pvldelta = Dict(
    "A Mean" => priors_pvldelta["mu_A_pr"],
    "A SD" => priors_pvldelta["sd_A"],
    "w Mean" => priors_pvldelta["mu_w_pr"],
    "w SD" => priors_pvldelta["sd_w"],
    "c Mean" => priors_pvldelta["mu_c_pr"],
    "c SD" => priors_pvldelta["sd_c"],
    "a Mean" => priors_pvldelta["mu_a_pr"],
    "a SD" => priors_pvldelta["sd_a"],
)

p = plot_parameter_distribution(
    replacenames(pvldelta_chains[2], name_map_pvldelta), priors_pvldelta
)
plot(p, size=(1200, 2400), plot_title="C + D (Good)", thickness_scaling = 1.5, legend=false)

# save the plot
Plots.savefig("figures/igt_pvldelta_C_D_Good.png")

p = plot_parameter_distribution(
    replacenames(pvldelta_chains[1], name_map_pvldelta), priors_pvldelta
)
plot(p, size=(1200, 2400), plot_title="A + B (Bad)", thickness_scaling = 1.5, legend=false)

# save the plot
Plots.savefig("figures/igt_pvldelta_A_B_Bad.png")

p = plot_parameter_distribution(
    replacenames(pvldelta_chains[4], name_map_pvldelta), priors_pvldelta
)
plot(p, size=(1200, 2400), plot_title="B + D (Infrequent loss)", thickness_scaling = 1.5, legend=false)

# save the plot
Plots.savefig("figures/igt_pvldelta_B_D_Infrequent_loss.png")

p = plot_parameter_distribution(
    replacenames(pvldelta_chains[8], name_map_pvldelta), priors_pvldelta
)
plot(p, size=(1200, 2400), plot_title="A + C (Frequent loss)", thickness_scaling = 1.5, legend=false)


# save the plot
Plots.savefig("figures/igt_pvldelta_A_C_Frequent_loss.png")

p = plot_parameter_distribution(
    replacenames(pvldelta_chains[5], name_map_pvldelta), priors_pvldelta
)
plot(p, size=(1200, 2400), plot_title="B", thickness_scaling = 1.5, legend=false)

# save the plot
Plots.savefig("figures/igt_pvldelta_B.png")

p = plot_parameter_distribution(
    replacenames(pvldelta_chains[10], name_map_pvldelta), priors_pvldelta
)
plot(p, size=(1200, 2400), plot_title="C", thickness_scaling = 1.5, legend=false)

# save the plot
Plots.savefig("figures/igt_pvldelta_C.png")

p = plot_parameter_distribution(
    replacenames(pvldelta_chains[6], name_map_pvldelta), priors_pvldelta
)
plot(p, size=(1200, 2400), plot_title="D", thickness_scaling = 1.5, legend=false)

# save the plot
Plots.savefig("figures/igt_pvldelta_D.png")

p = plot_parameter_distribution(
    replacenames(pvldelta_chains[0], name_map_pvldelta), priors_pvldelta
)
plot(p, size=(1200, 2400), plot_title="No Preference", thickness_scaling = 1.5, legend=false)

# save the plot
Plots.savefig("figures/igt_pvldelta_No_Preference.png")









pvldelta_chains[4][:, pvldelta_selected_params_sym, :]
names(pvldelta_chains[2])
plot_parameter_distribution(pvldelta_chains[5],priors_pvldelta)

# Plots.plot(pvldelta_chains[1][:, pvldelta_selected_params_sym, :])

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






idata_turing_post = from_mcmcchains(pvldelta_chains[2])

ArviZ.psis(idata_turing_post.posterior)
ArviZ.plot_trace(idata_turing_post)
