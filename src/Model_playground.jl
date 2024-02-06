using Distributions
using Turing
using Plots, StatsPlots
using ActionModels
using Distributed
using HierarchicalGaussianFiltering
include("Data.jl")

@model function test(inputs::Vector, actions::Vector)
    σ ~ LogNormal(0, 1)
    μ ~ LogNormal(0, 1)
    for (action, input) in zip(actions, inputs)
        action ~ Normal(μ, σ) ## you could put PVL delta funcion / action model here
    end
end

inputs = [1, 2, 3, 4, 5]
actions = [1, 2, 3, 4, 5]

model = test(inputs, actions)

output = sample(model, NUTS(), 1000)

plot(output)


## ActionModels


rmprocs(workers())
addprocs(4, exeflags = "--project")
@everywhere @eval using HierarchicalGaussianFiltering
@everywhere @eval using ActionModels

@everywhere function rw(agent::Agent, input::Real)
    # get params
    learning_rate = agent.parameters["learning_rate"]
    action_noise = agent.parameters["action_noise"]

    # get previous value (states)
    prev_value = agent.states["value"]

    # Rescorla-Wagner update rule
    new_value = prev_value + learning_rate * (input - prev_value)

    # update states
    agent.states["value"] = new_value
    push!(agent.history["value"], new_value)

    # sample action
    action_distribution = Normal(new_value, action_noise)

    # return action distribution
    return action_distribution
end

agent = init_agent(
    rw,
    parameters = Dict(
        "learning_rate" => 0.1,
        "action_noise" => 0.1,
        # optionally you can  add an initial
        ("initial", "value") => 0.0
    ),
    states = Dict(
        "value" => 0.0,
    ),
)

simulated_actions = give_inputs!(agent, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# agent
agent
get_history(agent, "value")

plot_trajectory(
    agent,
    "value",
    #other plot args
)

# reset agent (removes history, resets states, etc)
reset!(agent)

# you c an set params
set_parameters!(agent, Dict("learning_rate" => 0.5))
# corresponding get
get_parameters(agent, "learning_rate")
# all params
get_parameters(agent)

# fit model to data
priors = Dict(
    "learning_rate" => LogitNormal(0, 1),
    "action_noise" => LogNormal(0, 1),
    ("initial", "value") => Normal(0, 1), # again optional (you can instead try manual params / starting points)

)



# make rw function available everywhere
# @everywhere @eval using agent = $agent # not necessary

results = fit_model(
    agent,
    priors,
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    n_chains = 4,
    n_cores = 4, # currently only works with ActionModels functions
    verbose = false,
)

# plot results
plot(results)

# plot param Distributions
plot_parameter_distribution(results, priors)

rmprocs(workers())


continuous_2level = premade_hgf("continuous_2level")

get_parameters(continuous_2level)
# nodes: u, x1, x2
# u = input noise

## ... prediction_mean, prediction_precision
#params 
# volatility -inf to inf but practially -10 to 1
# higher valitility, less certainty, faster updates (world can chang more)

# initial_precision, initial_mean

# drift, if you have drift 1 and no change,
# "autoregression_strength" / target -  the target defines the expected value around 
# if strength is 1, it will always go back

set_prameters!(continuous_2level, Dict(("x1", "autoregression_target") => 1, ("x1", "autoregression_strength") => 0.2))

# if "with params and inputs node n becomes negative
# this happens with a high enough jump, or it gets too certain
reset!(continuous_2level)
give_inputs!(continuous_2level, [1.1, 1.2, 1.2, 1.1, 1.2, 1.2, 1.1, 1.2, 1.2, 1.1, 1.2, 1.2, 1.1, 1.2, 1.2, 1.1, 1.2, 1.2, 3, 4])
                                    # (node, parameter)
plot_trajectory(continuous_2level, ("x1", "posterior")) # e.g. probability, average reward

function hgf_continuous_action(agent::Agent, input::Real)

    # extract parameters
    action_noise = agent.parameters["action_noise"]

    #  extract HGF
    hgf = agent.substruct

    # Update HGF
    update_hgf!(hgf, input)

    # get prediction
    prediction = get_states(hgf, ("x1", "prediction_mean"))

    action_distribution = Normal(prediction, action_noise)
end

hgf_agent = init_agent(hgf_continuous_action,
    parameters = Dict(
        "action_noise" => 0.1,
    ),
    substruct = continuous_2level,
)

simulated_actions = give_inputs!(hgf_agent, [1, 2, 3])

plot_trajectory(hgf_agent, ("x1", "prediction_mean"))
plot_trajectory(hgf_agent, ("x1", "prediction"))

reset!(hgf_agent)

priors = Dict(
    "action_noise" => LogNormal(0,1),
    ("x1", "volatility") => Normal(0,1),
)

result = fit_model(
    hgf_agent,
    priors,
    [1,2,3],
    [1,1,1],
)

plot(result)

plot_parameter_distribution(result, priors)




# ActionModels with DataFrames
# instead of giving inputs, you can give a dataframe (in fit_model)
# ActionModels supports dataframes, you can say which cols are inputs / actions, and it  supports hierarchichal models
# cols
# can also fit seperately for each subject, and it will parallelize over chains and subjects

# for Multilevel
# learning_rate = MultiLevel(...)


function softmax(x, action_noise)
    return exp.(x ./ action_noise) ./ sum(exp.(x ./ action_noise))
end

function igt_hgf_action(agent::Agent, input::Real)
    # input is a value, the outcome of the deck, negative or positive

    # get action_noise parameter
    action_noise = agent.parameters["action_noise"] # ...

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
    action_probabilities = softmax(expected_values, action_noise)

    # final action distribution is Categorical
    action_distribution = Categorical(action_probabilities)
    return action_distribution    
end

# input_nodes = [
#     Dict(
#         "name" => "u1",
#         # "input_noise" #if not specified, uses default

#     ),
# ]

# using all defaults
input_nodes = [
    "u1",
    "u2",
    "u3",
    "u4",
]

# can try modelling volatility, but for now it's fixed
state_nodes = [
    "x1",
    "x2",
    "x3",
    "x4",
    # optional
    # "xvol", # hgf's guess at how much things are changing over time -> huge loss of 1500 might
]

edges = [
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

shared_parameters = Dict(
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

hgf = init_hgf(
    input_nodes = input_nodes,
    state_nodes = state_nodes,
    edges = edges,
    shared_parameters = shared_parameters,
)

agent = init_agent(
    igt_hgf_action,
    parameters = Dict(
        "action_noise" => 1,
    ),
    substruct = hgf,
)

give_inputs!(agent, [1, 2, 3, 4, 5, 6, -1500, 8, 9, 10])

# e.g. a result from deck 2
update_hgf!(hgf, [missing, 1, missing, missing])

# nothing with above
plot_trajectory(hgf, "x1")

# something HierarchicalGaussianFiltering
plot_trajectory(hgf, "x2")

get_parameters(hgf) # lots



# if resicolra wagner, it can model as above (unless it models wins and losses seperately)

# assumme there is a real average value, and noise around it

# but really there's a probabilty of getting a losses

# could do binary HGF, get wether a loss or not

# can also model wins and losses seperately


# input noise should be very small, e.g. people just see 50 or 100, not 50.1 or 99.9

# shared vol or individual params...start without then try both


# one day active inference.

result = fit_model(
    hgf_agent,
    priors,
    dataframe,
    independent_group_cols = [:experiment],
    # if you want to fit seperately for each subject
    # this will run a lot faster
    # independent_group_cols = [:experiment, :ID],
    multilevel_group_cols = [:ID, :group],
    input_cols = [:input],
    action_cols = [:action],
    n_cores = 4,
)