using LinearAlgebra
using Feather
LinearAlgebra.BLAS.set_num_threads(1)  # possible julia/windows bug fix?
using Distributions, Optim, Turing, StatsFuns
using ActionModels
using HierarchicalGaussianFiltering
using HDF5
using MCMCChains
using MCMCChainsStorage
using StatsBase
using ParetoSmooth
using Turing: DynamicPPL
using Plots, StatsPlots
using CSV
using CategoricalArrays
using DataFrames
using Base.Threads

include("Data.jl")

using LogExpFunctions

autodiff = AutoReverseDiff(compile = true)


function TruncatedNormal(μ::Real = 0, σ::Real = 1; check_args::Bool = false, a::Real = 0, b::Real = 1)
    return Truncated(Normal(μ, σ; check_args=check_args), a, b)
end


function igt_hgf_action(agent::Agent, input::Union{Missing, Real})
    # input is a value, the outcome of the deck, negative or positive

    # get action_noise parameter
    action_noise = agent.parameters["action_noise"]

    # get previously selected deck
    prev_action = agent.states["action"]

    hgf = agent.substruct

    # create input vector
    if !ismissing(prev_action)
        input_vector = Vector{Union{Real, Missing}}(missing, 4)
        input_vector[prev_action] = input

        # Update HGF
        update_hgf!(hgf, input_vector)
    end

    # Expected values come from the HGF
    deck_predictions = [
        get_states(hgf, ("x1", "prediction_mean")),
        get_states(hgf, ("x2", "prediction_mean")),
        get_states(hgf, ("x3", "prediction_mean")),
        get_states(hgf, ("x4", "prediction_mean")),
    ]

    # Expected values are softmaxed with action noise
    action_probabilities = softmax(deck_predictions .* action_noise)

    # if any the action probabilities are not between 0 and 1 we throw an error to reject the sample
    if any(x -> !(0 <= x <= 1), action_probabilities)
        throw(
            RejectParameters(
                "With these parameters and inputs, the action probabilities became $action_probabilities, which should be between 0 and 1. Try other parameter settings",
            ),
        )
    end
    

    # final action distribution is Categorical
    action_distribution = Categorical(action_probabilities)
    return action_distribution    
end


parameter_groups = [
    # inputs
    ParameterGroup("biases", [("u1", "bias"), ("u2", "bias"), ("u3", "bias"), ("u4", "bias")], 0),
    ParameterGroup("input_noises", [("u1", "input_noise"), ("u2", "input_noise"), ("u3", "input_noise"), ("u4", "input_noise")], 1),

    # states
    ParameterGroup("volatilities", [("x1", "volatility"), ("x2", "volatility"), ("x3", "volatility"), ("x4", "volatility")], 1),
    ParameterGroup("initial_means", [("x1", "initial_mean"), ("x2", "initial_mean"), ("x3", "initial_mean"), ("x4", "initial_mean")], 0),
    ParameterGroup("drifts", [("x1", "drift"), ("x2", "drift"), ("x3", "drift"), ("x4", "drift")], 0),
    ParameterGroup("autoconnection_strengths", [("x1", "autoconnection_strength"), ("x2", "autoconnection_strength"), ("x3", "autoconnection_strength"), ("x4", "autoconnection_strength")], 1),
    ParameterGroup("initial_precisions", [("x1", "initial_precision"), ("x2", "initial_precision"), ("x3", "initial_precision"), ("x4", "initial_precision")], 1),
]
node_defaults = NodeDefaults(
    bias = 0,
    input_noise = 0,
    volatility = 1,
    coupling_strength = 1,
    drift = 0,
    autoconnection_strength = 1,
    initial_precision = 1,
    initial_mean = 0.25,
)
fixed_parameters = Dict(
    "biases" => 0,
    "input_noises" => 0,
    "autoconnection_strengths" => 1,
    "initial_precisions" => 1,
    "initial_means" => 0.25,
)
nodes = [
    ContinuousInput(name = "u1", input_noise = 1),
    ContinuousInput(name = "u2", input_noise = 1),
    ContinuousInput(name = "u3", input_noise = 1),
    ContinuousInput(name = "u4", input_noise = 1),
    ContinuousState(name = "x1", volatility = 1, initial_mean = 0, drift = 0),
    ContinuousState(name = "x2", volatility = 1, initial_mean = 0, drift = 0),
    ContinuousState(name = "x3", volatility = 1, initial_mean = 0, drift = 0),
    ContinuousState(name = "x4", volatility = 1, initial_mean = 0, drift = 0),
]

edges = Dict(
    ("u1", "x1") => ObservationCoupling(),
    ("u2", "x2") => ObservationCoupling(),
    ("u3", "x3") => ObservationCoupling(),
    ("u4", "x4") => ObservationCoupling(),
)



hgf = init_hgf(
    nodes = nodes,
    edges = edges,
    node_defaults = node_defaults,
    parameter_groups = parameter_groups,
)

@info "HGF Parameters"
@info get_parameters(hgf)


agent = init_agent(
    igt_hgf_action,
    parameters = Dict(
        "action_noise" => 1, # 1 = no noise
    ),
    substruct = hgf,
    save_history = true,
)

@info "Agent Parameters"
@info get_parameters(agent)



@info "Loading Data..."
trial_data::DataFrame = DataFrame()
pats::Vector{Int} = Vector{Int}(undef, 0)
n_subj::Vector{Int} = Vector{Int}(undef, 0)

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
else
    @info "Cached data not found, loading from RData..."
    trial_data = load_trial_data("./data/IGTdataSteingroever2014/IGTdata.rdata")

    @info "Segmenting Data..."
    trial_data.choice_pattern_ab = ifelse.(trial_data.ab_ratio .>= 0.65, 1, 0)
    trial_data.choice_pattern_cd = ifelse.(trial_data.cd_ratio .>= 0.65, 2, 0)
    trial_data.choice_pattern_bd = ifelse.(trial_data.bd_ratio .>= 0.65, 4, 0)
    trial_data.choice_pattern_ac = ifelse.(trial_data.ac_ratio .>= 0.65, 8, 0)

    trial_data.choice_pattern = trial_data.choice_pattern_ab .| trial_data.choice_pattern_cd .| trial_data.choice_pattern_bd .| trial_data.choice_pattern_ac

    # patterns
    pats = unique(trial_data.choice_pattern)

    # get number of unique subjects for each pattern
    n_subj = [length(unique(trial_data[trial_data.choice_pattern .== pat, :subj])) for pat in pats]

    # save trial_data, pats and n_subj to file
    @info "Saving data to cache file..."
    Feather.write(cached_data_file, trial_data)
    h5open(cached_metadata_file, "cw") do file
        write(file, "pats", pats)
        write(file, "n_subj", n_subj)
    end
end

priors = Dict(
    "action_noise" => TruncatedNormal(0.5, 1.00; a = 0, b = 20),
    "volatilities" => TruncatedNormal(0.5, 1.0),
    "drifts" => TruncatedNormal(0.5, 1.0),
)



hgf_chains = Dict()
h5open("data/igt_hgf_FIX_subj_simplified_multiparam_data_chains.h5", "r") do file
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

# now we group all of the subjects for each pattern
# we will then compare the group level differences
hgf_chains_grouped = Dict()
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
for (pat, label) in pat_map
    subj_ids = unique(trial_data[trial_data[!, :choice_pattern] .== pat, :subj])
    str_subj_ids = string.(subj_ids)
    str_subj_ids = ["subj_$id" for id in str_subj_ids]
    subj_chains = [hgf_chains[id] for id in str_subj_ids]
    hgf_chains_grouped[label] = chainscat(subj_chains...)
end

samplr = NUTS(1_500, 0.65; max_depth=20, Δ_max=0.75, init_ϵ = 0.1, adtype=autodiff)


# print info about number of patterns/groups and number of subjects in each
@info "Patterns and Subjects"
for (pat, n) in zip(pats, n_subj)
    @info("Pattern: $pat, Subjects: $n")
    # standardize outcome between -1 and 1 (keeping 0 as zero) by pattern
    @info "Normalizing outcome between -1 and 1..."
    max_abs_outcome = maximum(abs.(trial_data[trial_data.choice_pattern .== pat, :outcome]))
    trial_data[trial_data.choice_pattern .== pat, :outcome] = trial_data[trial_data.choice_pattern .== pat, :outcome] ./ max_abs_outcome
end

# # standardize outcome between -1 and 1
# max_abs_outcome = maximum(abs.(trial_data.outcome))
# trial_data.outcome = trial_data.outcome ./ max_abs_outcome
# @info "Normalizing outcome between -1 and 1..."


MODELS = fit_model(
    agent,
    priors,
    trial_data,
    sampler = samplr,
    independent_group_cols = [:subj],
    # independent_group_cols = [:choice_pattern],
    # multilevel_group_cols = [:subj],
    fixed_parameters = fixed_parameters,
    input_cols = [:outcome],
    action_cols = [:next_choice], # use the following row's choice as the action
    n_cores = 1,
    n_chains = 1,
    n_iterations = 3_000,
    verbose = true,
    progress = true,
)

ParetoSmooth.psis(
    MODELS[1],
    replacenames(
        hgf_chains["subj_1"],
        Dict(
            "action_noise" => "agent_parameters[()][\"action_noise\"]",
            "volatilities" => "agent_parameters[()][\"volatilities\"]",
            "drifts" => "agent_parameters[()][\"drifts\"]",
        ),
    ),
    source="mcmc",
)


ll = loglikelihood(
    MODELS[1],
    replacenames(
        hgf_chains["subj_1"],
        Dict(
            "action_noise" => "agent_parameters[()][\"action_noise\"]",
            "volatilities" => "agent_parameters[()][\"volatilities\"]",
            "drifts" => "agent_parameters[()][\"drifts\"]",
        ),
    ),
)

ParetoSmooth.psis_loo(
    MODELS[2],
    replacenames(
        hgf_chains["subj_2"],
        Dict(
            "action_noise" => "agent_parameters[()][\"action_noise\"]",
            "volatilities" => "agent_parameters[()][\"volatilities\"]",
            "drifts" => "agent_parameters[()][\"drifts\"]",
        ),
    )
)
DynamicPPL.VarInfo(MODELS[1])

trial_data[trial_data[!, :subj] .== 1, :choice] ≈ MODELS[1].args.actions[()][: , 1]
MODELS[1].args.actions[()][1:94 , 1] == trial_data[trial_data[!, :subj] .== 1, :choice][2:end]

trial_data[trial_data[!, :subj] .== 1, :]
ParetoSmooth.psis(MODELS[1], hgf_chains["subj_1"])

hgf_chains["subj_1"]
plot_trajectory(MODELS[1].args.agent, ("u1", "input_value"))
plot_trajectory(MODELS[1].args.agent, "action")

get_states(MODELS[1].args.agent)

get_history(MODELS[1].args.agent)

loglikelihood(
    MODELS[1],
    replacenames(
        hgf_chains["subj_1"],
        Dict(
            "action_noise" => "agent_parameters[()][\"action_noise\"]",
            "volatilities" => "agent_parameters[()][\"volatilities\"]",
            "drifts" => "agent_parameters[()][\"drifts\"]",
        ),
    )
)

predict(MODELS[1], hgf_chains["subj_1"])

gq = generated_quantities(MODELS[1], replacenames(
    hgf_chains["subj_1"],
    Dict(
        "action_noise" => "agent_parameters[()][\"action_noise\"]",
        "volatilities" => "agent_parameters[()][\"volatilities\"]",
        "drifts" => "agent_parameters[()][\"drifts\"]",
    ),
))

# get mode for each vector value in the generated quantities
predictions = mode(gq)
# compare to the actual choices
actual = trial_data[trial_data[!, :subj] .== 1, :choice][2:end]
# plot the actual and predicted choices
plot(actual, label="Actual")
plot!(predictions, label="Predicted")

pvldelta_chains

pat_map_short = Dict(
    2 => "C + D",
    1 => "A + B",
    4 => "B + D",
    8 => "A + C",
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



N = length(MODELS)
hgf_accuracies = Dict(
    "pattern" => Vector{Int}(undef, N),
    "pattern_label" => Vector{String}(undef, N),
    "subject" => Vector{Int}(undef, N),
    "accuracy" => Vector{Float64}(undef, N),
    "predicted_prop_A" => Vector{Float64}(undef, N),
    "predicted_prop_B" => Vector{Float64}(undef, N),
    "predicted_prop_C" => Vector{Float64}(undef, N),
    "predicted_prop_D" => Vector{Float64}(undef, N),
)

error_indices = []

for i in eachindex(MODELS)
    try
        @info "$(Threads.threadid()) Subject: $i"
        m = MODELS[i]
        @info "$(Threads.threadid()) Retrieving chains...$i"
        mc = replacenames(
            hgf_chains["subj_$i"],
            Dict(
                "action_noise" => "agent_parameters[()][\"action_noise\"]",
                "volatilities" => "agent_parameters[()][\"volatilities\"]",
                "drifts" => "agent_parameters[()][\"drifts\"]",
            ),
        )
        @info "$(Threads.threadid()) Generating quantities: $i"
        gq = generated_quantities(m, mc)

        # get mode for each vector value in the generated quantities
        predictions = mode(gq)
        # compare to the actual choices
        actual = trial_data[trial_data.subj .== i, :choice][2:end]
        p = trial_data[trial_data.subj .== i, :choice_pattern][1]
        hgf_accuracies["pattern"][i] = p
        hgf_accuracies["pattern_label"][i] = pat_map_short[p]
        hgf_accuracies["subject"][i] = i
        hgf_accuracies["accuracy"][i] = sum(actual .== predictions) / length(actual)
        hgf_accuracies["predicted_prop_A"][i] = sum(predictions .== 1) / length(actual)
        hgf_accuracies["predicted_prop_B"][i] = sum(predictions .== 2) / length(actual)
        hgf_accuracies["predicted_prop_C"][i] = sum(predictions .== 3) / length(actual)
        hgf_accuracies["predicted_prop_D"][i] = sum(predictions .== 4) / length(actual)
    catch e
        @warn "Error: $e"
        push!(error_indices, i)
    end
end

@warn "Errors: $error_indices"

hgf_accuracies_df = DataFrame(
    hgf_accuracies
)


CSV.write("data/accuracies_hgf.csv", hgf_accuracies_df)

# print mean accuracy for each group
# hgf_df_grouped = groupby(hgf_accuracies_df, :pattern)
hgf_df_grouped = combine(
    hgf_accuracies_df,
    :accuracy,
    :pattern => (x -> [pat_map_short[p] for p in x] ) => :pattern_label,
    :pattern => (x -> "HGF") => :model,
)

hgf_df_grouped.pattern_label = categorical(hgf_df_grouped.pattern_label)

levels!(hgf_df_grouped.pattern_label, [
    pat_map_short[2],
    pat_map_short[1],
    pat_map_short[4],
    pat_map_short[8],
    pat_map_short[5],
    pat_map_short[10],
    pat_map_short[6],
    pat_map_short[0],
])

boxplot(
    hgf_df_grouped.pattern_label,
    hgf_df_grouped.accuracy,
    # group = hgf_df_grouped.pattern_label, 
    xlabel="Pattern",
    ylabel="Mean Accuracy",
    title="Mean Accuracy by Pattern",
    legend=:topleft,
    size=(1800, 1200),
    thickness_scaling = 2,
    margin=(20, :px),
)

# plot the mean predicted proportions for each group
gf_df_grouped = combine(
    hgf_accuracies_df,
    AsTable([
        :predicted_prop_A,
        :predicted_prop_B,
        :predicted_prop_C,
        :predicted_prop_D,
        :pattern,
        :subject,
    ]
    ) => (
        (x,) -> (
            pattern_label = repeat([pat_map_short[p] for p in x.pattern], 4),
            deck = repeat(["A", "B", "C", "D"], inner=length(x.pattern)),
            deck_prop = vcat(
                x.predicted_prop_A,
                x.predicted_prop_B,
                x.predicted_prop_C,
                x.predicted_prop_D
            ),
            subject = repeat(x.subject, 4),
        )
    ) => AsTable,
)

hgf_df_grouped.pattern_label = categorical(hgf_df_grouped.pattern_label)

levels!(hgf_df_grouped.pattern_label, [
    pat_map_short[2],
    pat_map_short[1],
    pat_map_short[4],
    pat_map_short[8],
    pat_map_short[5],
    pat_map_short[10],
    pat_map_short[6],
    pat_map_short[0],
])





















# map the STAN variable names to the Turing variable names
# get number of subjects in group
base_pvldelta_param_map = Dict(
    "mu_A_pr" => "Aμ",
    "mu_w_pr" => "wμ",
    "mu_a_pr" => "aμ",
    "mu_c_pr" => "cμ",
    "sd_A" => "Aσ",
    "sd_w" => "wσ",
    "sd_a" => "aσ",
    "sd_c" => "cσ",
)
subj_params_map = Dict(
    "A_ind_pr" => "A_pr",
    "w_ind_pr" => "w_pr",
    "a_ind_pr" => "a_pr",
    "c_ind_pr" => "c_pr",
)

accuracies = Dict(
    "pattern" => Int[],
    "pattern_label" => String[],
    "subject" => Int[],
    "accuracy" => Float64[],
    "predicted_prop_AB" => Float64[],
    "predicted_prop_CD" => Float64[],
    "predicted_prop_BD" => Float64[],
    "predicted_prop_AC" => Float64[],
    "predicted_prop_A" => Float64[],
    "predicted_prop_B" => Float64[],
    "predicted_prop_C" => Float64[],
    "predicted_prop_D" => Float64[],
)
for target_group in pats
    @info "Group: $target_group"
    idx = findfirst(x -> x == target_group, pats)
    subjs_in_group = n_subj[idx]
    @info "N: $subjs_in_group"
    # copy
    group_pvldelta_param_map = copy(base_pvldelta_param_map)
    for i in 1:subjs_in_group
        for (src, dest) in subj_params_map
            group_pvldelta_param_map["$src.$i"] = "$dest[$i]"
        end
    end
    keep_params = [
        param for param in values(group_pvldelta_param_map)
    ]

    pvldelta_chains_group = replacenames(
        pvldelta_chains[target_group],
        group_pvldelta_param_map,
    )[keep_params]


    @info "Loading trial data..."
    trial_data_pat = trial_data[trial_data.choice_pattern .== target_group, :]
    trial_data_pat.subj_uid = join.(zip(trial_data_pat.study, trial_data_pat.subj), "_")
    # get unique subjects (subject_id and study_id)
    subjs = unique(trial_data_pat.subj_uid) 
    N = length(subjs)
    Tsubj = [length(trial_data_pat[trial_data_pat.subj_uid .== subj, :subj]) for subj in subjs]
    choice = Matrix{Int}(undef, N, maximum(Tsubj))

    # this array is overlarge but not sure how to reduce it
    # sparse array maybe?
    # deck_payoffs = Array{Float64, 3}(undef, N, maximum(Tsubj), 4)
    # actually initialize with zeros
    deck_payoffs = zeros(Float64, N, maximum(Tsubj))
    # deck_wl = Array{Int, 3}(undef, N, maximum(Tsubj), 4)
    deck_wl = zeros(Int, N, maximum(Tsubj))
    payoff_schemes = Vector{Int}(undef, N)
    @info "Loading subject wins, losses, and payoffs..."
    for (i, subj) in enumerate(subjs)
        subj_data = trial_data_pat[trial_data_pat.subj_uid .== subj, :]
        @inbounds choice[i, 1:Tsubj[i]] = subj_data.choice
        @inbounds deck_payoffs[i, 1:Tsubj[i]] = subj_data.outcome
        @inbounds deck_wl[i, 1:Tsubj[i]] = Int.(subj_data.outcome .< 0)
        @inbounds payoff_schemes[i] = subj_data.scheme[1]
    end

    @info "Creating PVL delta model..."
    gen_model = pvl_delta(choice, N, Tsubj, deck_payoffs, deck_wl)

    # get the generated quantities
    @info "Generating quantities...hold on to you rhat"
    gqp = generated_quantities(gen_model, pvldelta_chains_group)

    # get the mode of the generated quantities
    @info "Getting mode of generated quantities..."
    gqp_mode = mode(gqp)[1]

    # plot the actual and predicted choices
    # subj_id = 3
    # plot(choice[subj_id, 1:Tsubj[subj_id]], label="Actual", size=(1600, 1200), thickness_scaling = 2)
    # plot!(gqp_mode[subj_id, 1:Tsubj[subj_id]], label="Predicted")

    # calculate the predictive accuracy
    @info "Calculating accuracy..."
    for i in 1:N
        acc = sum(choice[i, 1:Tsubj[i]] .== gqp_mode[i, 1:Tsubj[i]]) / Tsubj[i]

        @info "Subject $i Accuracy: $acc"
        push!(accuracies["pattern"], target_group)
        push!(accuracies["pattern_label"], pat_map[target_group])
        push!(accuracies["subject"], i)
        push!(accuracies["accuracy"], acc)
        push!(accuracies["predicted_prop_AB"], (sum(gqp_mode[i, 1:Tsubj[i]] .== 1) + sum(gqp_mode[i, 1:Tsubj[i]] .== 2)) / Tsubj[i])
        push!(accuracies["predicted_prop_CD"], (sum(gqp_mode[i, 1:Tsubj[i]] .== 3) + sum(gqp_mode[i, 1:Tsubj[i]] .== 4)) / Tsubj[i])
        push!(accuracies["predicted_prop_BD"], (sum(gqp_mode[i, 1:Tsubj[i]] .== 2) + sum(gqp_mode[i, 1:Tsubj[i]] .== 4)) / Tsubj[i])
        push!(accuracies["predicted_prop_AC"], (sum(gqp_mode[i, 1:Tsubj[i]] .== 1) + sum(gqp_mode[i, 1:Tsubj[i]] .== 3)) / Tsubj[i])
        push!(accuracies["predicted_prop_A"], sum(gqp_mode[i, 1:Tsubj[i]] .== 1) / Tsubj[i])
        push!(accuracies["predicted_prop_B"], sum(gqp_mode[i, 1:Tsubj[i]] .== 2) / Tsubj[i])
        push!(accuracies["predicted_prop_C"], sum(gqp_mode[i, 1:Tsubj[i]] .== 3) / Tsubj[i])
        push!(accuracies["predicted_prop_D"], sum(gqp_mode[i, 1:Tsubj[i]] .== 4) / Tsubj[i])
    end
end

# save accuracies to csv
@info "Saving accuracies..."

accuracies_df = DataFrame(
    accuracies
)

CSV.write("data/accuracies_pvldelta_stan.csv", accuracies_df)

# print mean accuracy for each group
accuracies_df = CSV.read("data/accuracies_pvldelta_stan.csv", DataFrame)

# print proportion of each choice for each group to see if at least that is predicted correctly
@info "Mean Predicted Proportions"
df_grouped = groupby(accuracies_df, :pattern)
df_grouped = combine(df_grouped, :predicted_prop_AB => mean, :predicted_prop_CD => mean, :predicted_prop_BD => mean, :predicted_prop_AC => mean, :predicted_prop_A => mean, :predicted_prop_B => mean, :predicted_prop_C => mean, :predicted_prop_D => mean)



@info "Mean Accuracies"
df_grouped = groupby(accuracies_df, :pattern)
df_grouped = combine(df_grouped, :accuracy => mean, :pattern => (x -> pat_map_short[first(x)]) => :pattern_label)


df_grouped = combine(
    accuracies_df,
    :accuracy,
    :pattern => (x -> [pat_map_short[p] for p in x] ) => :pattern_label,
    :pattern => (x -> "PVL-Delta") => :model,
)
df_grouped.pattern_label = categorical(df_grouped.pattern_label)

levels!(df_grouped.pattern_label, [
    pat_map_short[2],
    pat_map_short[1],
    pat_map_short[4],
    pat_map_short[8],
    pat_map_short[5],
    pat_map_short[10],
    pat_map_short[6],
    pat_map_short[0],
])

df_both_models = vcat(df_grouped, hgf_df_grouped)


groupedboxplot(
    df_both_models.pattern_label,
    df_both_models.accuracy,
    group = df_both_models.model, 
    xlabel="Pattern",
    ylabel="Accuracy",
    title="Model Accuracy by Choice Pattern",
    legend=:topleft,
    size=(1800, 1200),
    thickness_scaling = 2,
    margin=(20, :px),
)

# save plot
savefig("figures/mean_accuracy_by_pattern_and_model.png")

# plot the mean accuracy for each group
@info "Plotting Mean Accuracies..."
scatter(
    df_grouped.pattern_label,
    df_grouped.accuracy_mean,
    group = df_grouped.pattern_label, 
    xlabel="Pattern",
    ylabel="Mean Accuracy",
    title="Mean Accuracy by Pattern",
    legend=:topleft,
    size=(1800, 1200),
    thickness_scaling = 2,
    margin=(20, :px),
)

# plot the mean predicted proportions for each group
@info "Plotting Mean Predicted Proportions..."
df_grouped = combine(
    accuracies_df,
    AsTable([
        :predicted_prop_A,
        :predicted_prop_B,
        :predicted_prop_C,
        :predicted_prop_D,
        :pattern,
        :subject,
    ]
    ) => (
        (x,) -> (
            pattern_label = repeat([pat_map_short[p] for p in x.pattern], 4),
            deck = repeat(["A", "B", "C", "D"], inner=length(x.pattern)),
            deck_prop = vcat(
                x.predicted_prop_A,
                x.predicted_prop_B,
                x.predicted_prop_C,
                x.predicted_prop_D
            ),
            subject = repeat(x.subject, 4),
        )
    ) => AsTable,
)

df_grouped.pattern_label = categorical(df_grouped.pattern_label)

levels!(df_grouped.pattern_label, [
    pat_map_short[2],
    pat_map_short[1],
    pat_map_short[4],
    pat_map_short[8],
    pat_map_short[5],
    pat_map_short[10],
    pat_map_short[6],
    pat_map_short[0],
])

df_grouped.deck = categorical(df_grouped.deck)
levels!(df_grouped.deck, ["A", "B", "C", "D"])


groupedboxplot(
    df_grouped[!, :deck],
    df_grouped[!, :deck_prop],
    group = df_grouped[!, :pattern_label],
    bar_position=:dodge, bar_width=0.5,
    bar_spacing=0,
    xlabel="Deck Choice",
    ylabel="Proportion of Choices",
    title="PVL-Delta Predicted Deck Choice Proportions by Pattern",
    size=(1800, 1400),
    color_palette = :Set1_8,
    thickness_scaling = 2.0,
    linewidth = 1.0,
    markerstrokewidth = 0.5,
    margin = (10, :px),
)



@info "Plotting Mean Predicted Proportions (HGF)..."
df_grouped = combine(
    hgf_accuracies_df,
    AsTable([
        :predicted_prop_A,
        :predicted_prop_B,
        :predicted_prop_C,
        :predicted_prop_D,
        :pattern,
        :subject,
    ]
    ) => (
        (x,) -> (
            pattern_label = repeat([pat_map_short[p] for p in x.pattern], 4),
            deck = repeat(["A", "B", "C", "D"], inner=length(x.pattern)),
            deck_prop = vcat(
                x.predicted_prop_A,
                x.predicted_prop_B,
                x.predicted_prop_C,
                x.predicted_prop_D
            ),
            subject = repeat(x.subject, 4),
        )
    ) => AsTable,
)

df_grouped.pattern_label = categorical(df_grouped.pattern_label)

levels!(df_grouped.pattern_label, [
    pat_map_short[2],
    pat_map_short[1],
    pat_map_short[4],
    pat_map_short[8],
    pat_map_short[5],
    pat_map_short[10],
    pat_map_short[6],
    pat_map_short[0],
])

df_grouped.deck = categorical(df_grouped.deck)
levels!(df_grouped.deck, ["A", "B", "C", "D"])


groupedboxplot(
    df_grouped[!, :deck],
    df_grouped[!, :deck_prop],
    group = df_grouped[!, :pattern_label],
    bar_position=:dodge, bar_width=0.5,
    bar_spacing=0,
    xlabel="Deck Choice",
    ylabel="Proportion of Choices",
    title="HGF Predicted Deck Choice Proportions by Pattern",
    size=(1800, 1400),
    color_palette = :Set1_8,
    thickness_scaling = 2.0,
    linewidth = 1.0,
    markerstrokewidth = 0.5,
    margin = (10, :px),
)

# save plot
savefig("figures/predicted_deck_choice_proportions_by_pattern_and_model_HGF.png")
