using Distributions
using Plots, StatsPlots
using ActionModels
using HierarchicalGaussianFiltering
using Distributed
#using StatsFuns
include("Data.jl")

@everywhere include("Data.jl")

@everywhere using HierarchicalGaussianFiltering
@everywhere using ActionModels
@everywhere using Distributions
#@everywhere using StatsFuns

@everywhere function noisy_softmax(x::Vector, noise::Real)
    return exp.(x / noise) / sum(exp.(x / noise))
end

@everywhere function igt_hgf_action(agent::Agent, input::Real)
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
)

@everywhere priors = Dict(
    "action_noise" => LogNormal(0,1),
    ("x1", "volatility") => Normal(0,1),
    ("x2", "volatility") => Normal(0,1),
    ("x3", "volatility") => Normal(0,1),
    ("x4", "volatility") => Normal(0,1),
    ("u1", "input_noise") => LogNormal(0,1),
    ("u2", "input_noise") => LogNormal(0,1),
    ("u3", "input_noise") => LogNormal(0,1),
    ("u4", "input_noise") => LogNormal(0,1),
)

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

@everywhere trial_data = load_trial_data("./data/IGTdataSteingroever2014/IGTdata.rdata")

# start with the 15 subjects of 95 trials
@everywhere trials_95 = trial_data[trial_data.trial_length .== 95, :]

result = fit_model(
    agent,
    priors,
    trials_95,
    independent_group_cols = ["subj", "trial_length"],
    input_cols = ["outcome"],
    action_cols = ["choice"],
    n_cores = 8,
    n_chains = 4,
    n_samples = 1000,
    verbose = false,
    progress = false,
)

rmprocs(workers())

# because we have individual level, the result is a dict of chains
# try with one example
result_1 = result[(1, 95)]
plot(result_1)


plot_trajectory(agent, ("u1", "input_value"))

print(get_parameters(agent))
print(get_states(agent))

print(get_parameters(hgf))
print(get_states(hgf))

plot_trajectory(hgf, "")