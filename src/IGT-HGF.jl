using Distributions
using Plots, StatsPlots
using ActionModels
using HierarchicalGaussianFiltering
using Distributed
using JLD
#using StatsFuns
include("Data.jl")

addprocs(4)

@everywhere include("Data.jl")

@everywhere using HierarchicalGaussianFiltering
@everywhere using ActionModels
@everywhere using Distributions
#@everywhere using StatsFuns

@everywhere function noisy_softmax(x::Vector, noise::Real)
    return exp.(x / noise) / sum(exp.(x / noise))
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
    expected_values = [
        get_states(hgf, ("x1", "prediction_mean")),
        get_states(hgf, ("x2", "prediction_mean")),
        get_states(hgf, ("x3", "prediction_mean")),
        get_states(hgf, ("x4", "prediction_mean")),
    ]

    # Expected values are softmaxed with action noise
    action_probabilities = noisy_softmax(expected_values, action_noise)

    # final action distribution is Categorical
    action_distribution = Categorical(action_probabilities)
    return action_distribution    
end

@everywhere input_nodes = [
    "u1",
    "u2",
    "u3",
    "u4",
]

# can try modelling volatility, but for now it's fixed
@everywhere state_nodes = [
    "x1",
    "x2",
    "x3",
    "x4",
    # optional
    # "xvol", # hgf's guess at how much things are changing over time -> huge loss of 1500 might
]

@everywhere edges = [
    Dict(
        "child" => "u1",
        "value_parents" => "x1",
    ),
    Dict(
        "child" => "u2",
        "value_parents" => "x2",
    ),
    Dict(
        "child" => "u3",
        "value_parents" => "x3",
    ),
    Dict(
        "child" => "u4",
        "value_parents" => "x4",
    ),
    # Dict(
    #     "child" => "x1",
    #     "volatility_parents" => "xvol", # shared, otherwise you could xvol1
    # ),

    # Dict(
    #     "child" => "x2",
    #     "volatility_parents" => "xvol",
    # ),
    # Dict(
    #     "child" => "x3",
    #     "volatility_parents" => "xvol",
    # ),
    # Dict(
    #     "child" => "x4",
    #     "volatility_parents" => "xvol",
    # ),
]

@everywhere shared_parameters = Dict(
    # value, names of all derived parameters
    "u_input_noise" => (
        1, 
        [
            ("u1", "input_noise"),
            ("u2", "input_noise"),
            ("u3", "input_noise"),
            ("u4", "input_noise")
        ],
    ),
    "x_volatility" => (
        1,
        [
            ("x1", "volatility"),
            ("x2", "volatility"),
            ("x3", "volatility"),
            ("x4", "volatility")
        ],
    ),
    "x_autoregression_strength" => (
        0,
        [
            ("x1", "autoregression_strength"),
            ("x2", "autoregression_strength"),
            ("x3", "autoregression_strength"),
            ("x4", "autoregression_strength")
        ],
    ),
    "x_autoregression_target" => (
        0,
        [
            ("x1", "autoregression_target"),
            ("x2", "autoregression_target"),
            ("x3", "autoregression_target"),
            ("x4", "autoregression_target")
        ],
    ),
    "x_initial_mean" => (
        0,
        [
            ("x1", "initial_mean"),
            ("x2", "initial_mean"),
            ("x3", "initial_mean"),
            ("x4", "initial_mean")
        ],
    ),
    "x_drift" => (
        0,
        [
            ("x1", "drift"),
            ("x2", "drift"),
            ("x3", "drift"),
            ("x4", "drift")
        ],
    ),
    "u_value_coupling" => (
        1,
        [
            ("u1", "x1", "value_coupling"),
            ("u2", "x2", "value_coupling"),
            ("u3", "x3", "value_coupling"),
            ("u4", "x4", "value_coupling")
        ],
    ),
    # do same for
    # strength, target, mean, drift (everything in get_parameters(hgf))
)

#set_parameters!(hgf, Dict("x_volatility" => 1))

@everywhere priors = Dict(
    "action_noise" => Multilevel(:subj, LogNormal, ["action_noise_group_mean", "action_noise_group_sd"]),
    "action_noise_group_mean" => Normal(),
    "action_noise_group_sd" => LogNormal(),

    "x_volatility" => Multilevel(:subj, Normal, ["x_volatility_group_mean", "x_volatility_group_sd"]),
    "x_volatility_group_mean" => Normal(),
    "x_volatility_group_sd" => LogNormal(0, 0.01),

    "u_input_noise" => Multilevel(:subj, LogNormal, ["u_input_noise_group_mean", "u_input_noise_group_sd"]),
    "u_input_noise_group_mean" =>  Normal(),
    "u_input_noise_group_sd" =>  LogNormal(),


    # ("x2", "volatility") => Normal(0,1),
    # ("x3", "volatility") => Normal(0,1),
    # ("x4", "volatility") => Normal(0,1),
    # ("u1", "input_noise") => LogNormal(0,1),
    # ("u2", "input_noise") => LogNormal(0,1),
    # ("u3", "input_noise") => LogNormal(0,1),
    # ("u4", "input_noise") => LogNormal(0,1),
)

# get_parameters(agent)

@everywhere hgf = init_hgf(
    input_nodes = input_nodes,
    state_nodes = state_nodes,
    edges = edges,
    shared_parameters = shared_parameters,
)

@everywhere agent = init_agent(
    igt_hgf_action,
    parameters = Dict(
        "action_noise" => 1, # 1 = no noise
    ),
    substruct = hgf,
)


get_parameters(hgf)
# reset!(agent)
# input = missing
# n_trials = 4


# for i in 1:n_trials
   
#     action = ActionModels.single_input!(agent, input)

#     input = 50 #somefunc(action)

# end

# plot of each four nodes (on same)
# get_history(agent, [("x2", "posterior_mean"), ])

# plot_trajectory(agent, "x1")
# plot_trajectory!(agent, "x2")





# stuff in a "missing" input to start
# ...

@everywhere trial_data = load_trial_data(
    "./data/IGTdataSteingroever2014/IGTdata.rdata",
    true
)

# start with the 15 subjects of 95 trials
@everywhere trials_95 = trial_data[trial_data.trial_length .== 95, :]

# for now only subj 1 and 2

@everywhere trials_95 = trials_95[trials_95.subj .<= 2, :]

result = fit_model(
    agent,
    priors,
    trial_data,
    independent_group_cols = ["trial_length"],
    multilevel_group_cols = ["subj"],
    input_cols = ["outcome"],
    action_cols = ["next_choice"], # use the following row's choice as the action
    n_cores = 4,
    #n_chains = 4,
    n_chains = 4,
    #n_samples = 1000,
    n_samples = 1000,
    verbose = false,
    progress = true,
)




# save("./data/igt_data_95_chains.jld", "chain", result, compress=true)
# saved_chain = load("./data/igt_data_95_chains.jld")
# get_posteriors(result)
# because we have individual level, the result is a dict of chains
# try with one example
result_1 = result[95]
plot(result_1)


plot_trajectory(agent, ("u1", "input_value"))

print(get_parameters(agent))
print(get_states(agent))

print(get_parameters(hgf))
print(get_states(hgf))

plot_trajectory(hgf, "")


rmprocs(workers())