using LinearAlgebra
using Feather
LinearAlgebra.BLAS.set_num_threads(1)  # possible julia/windows bug fix?
using Tapir, Distributions, Optim, Turing, StatsFuns
using ActionModels
using HierarchicalGaussianFiltering
using Distributed
using HDF5
using MCMCChains
using MCMCChainsStorage
using StatsBase


delete_existing_chains = true
# using JLD
#using StatsFuns
include("Data.jl")

addprocs(3)
# include("Data.jl")
@everywhere using Feather
@everywhere using LinearAlgebra
@everywhere LinearAlgebra.BLAS.set_num_threads(1)  # possible julia/windows bug fix?
@everywhere using HierarchicalGaussianFiltering
@everywhere using ActionModels
@everywhere using Distributions
@everywhere using Tapir
@everywhere using LogExpFunctions
# @everywhere using Turing: AutoTapir
# @everywhere using StatsFuns

autodiff = AutoReverseDiff(compile = true)
# autodiff = AutoTapir(safe_mode=false)
# autodiff = AutoForwardDiff()

@everywhere function TruncatedNormal(μ::Real = 0, σ::Real = 1; check_args::Bool = false, a::Real = 0, b::Real = 1)
    return Truncated(Normal(μ, σ; check_args=check_args), a, b)
end


@everywhere function igt_hgf_action(agent::Agent, input::Union{Missing, Real})
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


fixed_parameters = Dict(
    # "volatilities" => 1,
)

node_defaults = NodeDefaults(
    bias = 0,
    input_noise = 1,
    volatility = 1,
    coupling_strength = 1,
    initial_mean = 0,
    drift = 0,
    autoconnection_strength = 1,
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


parameter_groups = [
    # inputs
    ParameterGroup("biases", [("u1", "bias"), ("u2", "bias"), ("u3", "bias"), ("u4", "bias")], 0),
    ParameterGroup("input_noises", [("u1", "input_noise"), ("u2", "input_noise"), ("u3", "input_noise"), ("u4", "input_noise")], 1),

    # states
    # ParameterGroup("volatilities", [("x1", "volatility"), ("x2", "volatility"), ("x3", "volatility"), ("x4", "volatility")], 1),
    ParameterGroup("initial_means", [("x1", "initial_mean"), ("x2", "initial_mean"), ("x3", "initial_mean"), ("x4", "initial_mean")], 0),
    ParameterGroup("drifts", [("x1", "drift"), ("x2", "drift"), ("x3", "drift"), ("x4", "drift")], 0),
    ParameterGroup("autoconnection_strengths", [("x1", "autoconnection_strength"), ("x2", "autoconnection_strength"), ("x3", "autoconnection_strength"), ("x4", "autoconnection_strength")], 1),
    ParameterGroup("initial_precisions", [("x1", "initial_precision"), ("x2", "initial_precision"), ("x3", "initial_precision"), ("x4", "initial_precision")], 1),
]


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
    "action_noise" => Multilevel(
        :subj,
        TruncatedNormal,
        ["action_noise_pattern_mean", "action_noise_pattern_sd"]),
    "action_noise_pattern_mean" => TruncatedNormal(1.0, 1.0),
    "action_noise_pattern_sd" => TruncatedNormal(0, 0.01),

    "input_noises" => Multilevel(
        :subj,
        TruncatedNormal,
        ["input_noise_pattern_mean", "input_noise_pattern_sd"]),
    "input_noise_pattern_sd" => TruncatedNormal(0, 0.01),
    "input_noise_pattern_mean" => TruncatedNormal(0.0, 1.0),
    # "volatilities" => Multilevel(
    #     :subj,
    #     TruncatedNormal,
    #     ["volatility_pattern_mean", "volatility_pattern_sd"]),
    ("x1", "volatility") => Multilevel(
        :subj,
        TruncatedNormal,
        ["volatility_pattern_mean", "volatility_pattern_sd"]),
    ("x2", "volatility") => Multilevel(
        :subj,
        TruncatedNormal,
        ["volatility_pattern_mean", "volatility_pattern_sd"]),
    ("x3", "volatility") => Multilevel(
        :subj,
        TruncatedNormal,
        ["volatility_pattern_mean", "volatility_pattern_sd"]),
    ("x4", "volatility") => Multilevel(
        :subj,
        TruncatedNormal,
        ["volatility_pattern_mean", "volatility_pattern_sd"]),
    "volatility_pattern_mean" => TruncatedNormal(1.0, 1.0),
    "volatility_pattern_sd" => TruncatedNormal(0, 0.01),
    "initial_means" => Multilevel(
        :subj,
        TruncatedNormal,
        ["initial_mean_pattern_mean", "initial_mean_pattern_sd"]),
    "initial_mean_pattern_mean" => TruncatedNormal(0.0, 1.0),
    "initial_mean_pattern_sd" => TruncatedNormal(0, 0.01),
    "drifts" => Multilevel(
        :subj,
        TruncatedNormal,
        ["drift_pattern_mean", "drift_pattern_sd"]),
    "drift_pattern_mean" => TruncatedNormal(0.0, 1.0),
    "drift_pattern_sd" => TruncatedNormal(0, 0.01),
    "autoconnection_strengths" => Multilevel(
        :subj,
        TruncatedNormal,
        ["autoconnection_strength_pattern_mean", "autoconnection_strength_pattern_sd"]),
    "autoconnection_strength_pattern_mean" => TruncatedNormal(1.0, 1.0),
    "autoconnection_strength_pattern_sd" => TruncatedNormal(0, 0.01),
    "initial_precisions" => Multilevel(
        :subj,
        TruncatedNormal,
        ["initial_precision_pattern_mean", "initial_precision_pattern_sd"]),
    "initial_precision_pattern_mean" => TruncatedNormal(1.0, 1.0),
    "initial_precision_pattern_sd" => TruncatedNormal(0, 0.01),
)


# priors = Dict(
#     "action_noise" => TruncatedNormal(1.0, 1.0),
#     "input_noises" => TruncatedNormal(0.0, 1.0),
# )


@everywhere agent = $agent
@everywhere priors = $priors
@everywhere fixed_parameters = $fixed_parameters
@everywhere trial_data = $trial_data
@everywhere autodiff = $autodiff

sampler = NUTS(1_500, 0.65; max_depth=20, Δ_max=0.75, init_ϵ = 0.1, adtype=autodiff)
@everywhere sampler = $sampler

# model = fit_model(
#     agent,
#     priors,
#     trial_data,
#     sampler = sampler,
#     independent_group_cols = [:choice_pattern],
#     multilevel_group_cols = [:subj],
#     fixed_parameters = fixed_parameters,
#     input_cols = [:outcome],
#     action_cols = [:next_choice], # use the following row's choice as the action
#     n_cores = 3,
#     n_chains = 3,
#     n_samples = 3_000,
#     verbose = true,
#     progress = true
# )

# priors = Turing.extract_priors(model)

# print(priors)
# exit()

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


result = fit_model(
    agent,
    priors,
    trial_data,
    sampler = sampler,
    # independent_group_cols = [:subj],
    independent_group_cols = [:choice_pattern],
    multilevel_group_cols = [:subj],
    # fixed_parameters = fixed_parameters,
    input_cols = [:outcome],
    action_cols = [:next_choice], # use the following row's choice as the action
    n_cores = 3,
    n_chains = 3,
    n_iterations = 3_000,
    verbose = true,
    progress = true,
)

chain_out_file = "./data/igt_hgf_multilevel_multiparam_data_chains.h5"
# save chains
if delete_existing_chains && isfile(chain_out_file)
    @warn "Deleting file: $chain_out_file"
    rm(chain_out_file)
end
h5open(chain_out_file, "w") do file
    # save a group for each result (choice_pattern)
    for (pat, chain) in result
        out_group = "choice_pattern_$pat"
        g = create_group(file, out_group)
        write(g, chain)
    end
end



rmprocs(workers())
