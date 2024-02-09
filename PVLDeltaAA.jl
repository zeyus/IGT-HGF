using ForwardDiff
using Preferences
set_preferences!(ForwardDiff, "nansafe_mode" => true)
using Distributions
using Plots, StatsPlots
# using ActionModels
# using Distributed
using Turing
using Optim
using FillArrays
using StatsFuns
# using StatsBase
# using TypeUtils
# using ForwardDiff: value
include("src/Data.jl")




Turing.setprogress!(true)


@model function pvl_delta(actions::Matrix{Union{Missing, Int}}, N::Int, Tsubj::Vector{Int}, payoff_scheme::Int = 1, ::Type{T} = Float64) where {T}
    # Group Level Parameters
    Aμ ~ Normal(0, 1)
    Aσ ~ Uniform(0, 1.5)
    aμ ~ Normal(0, 1)
    aσ ~ Uniform(0, 1.5)
    cμ ~ Normal(0, 1)
    cσ ~ Uniform(0, 1.5)
    wμ ~ Normal(0, 1)
    wσ ~ Uniform(0, 1.5)
    
    # individual parameters
    A ~ filldist(LogitNormal(Aμ, Aσ), N)
    a ~ filldist(LogitNormal(aμ, aσ), N)
    # had to truncat to match paper
    # and avoid Inf values for θ
    c ~ filldist(truncated(LogNormal(cμ, cσ); upper=5), N)
    w ~ filldist(truncated(LogNormal(cμ, cσ); upper=5), N)
    
    if actions === missing
        actions = Matrix{Union{Missing, Int}}(undef, N, maximum(Tsubj))
    end

    for i in 1:N
        # Define values
        Evₖ = zeros(T, Tsubj[i], 4)
        
        # Set theta
        θ = 3^c[i] - 1
        payoffs = construct_payoff_sequence(payoff_scheme)
        for t in 1:Tsubj[i]
            # Get choice probabilities (random on first trial, otherwise softmax of expected utility)
            try
                choice_proabilities = t == 1 ? fill(0.25, 4) : softmax(θ .* Evₖ[t-1, :])
                # softmax choice (draw)
                actions[i, t] = rand(Categorical(choice_proabilities))
            catch e
                println("t: ", t)
                println("i: ", i)
                println("Evₖ[t-1, :]: ", Evₖ[t-1, :])
                println("θ: ", θ)
                println("c[i]", c[i])
                println("w[i]", w[i])
                println("A[i]", A[i])
                println("a[i]", a[i])
                throw(e)
            end
            # get the result for the selected deck
            k = actions[i, t]
            
            Xₜ = igt_deck_payoff!(actions[i, 1:t], payoffs, Int)
            # get prospect utility
            uₖ = (Xₜ >= 0) ? Xₜ^A[i] : -w[i] * abs(Xₜ)^A[i]
            
            # delta learning rule
            # get previous selection
            Evₖ₍ₜ₋₁₎ = t == 1 ? fill(0.0, 4) : Evₖ[t-1, :]
            # update expected value of selected deck
            Evₖ[t, k] = Evₖ₍ₜ₋₁₎[k] + a[i] * (uₖ - Evₖ₍ₜ₋₁₎[k])
            # all other decks carry on their previous value
            Evₖ[t, 1:end .!= k] = Evₖ₍ₜ₋₁₎[1:end .!= k]
        end
    end
    # return actions
end



N = 3
ntrials = 50
Tsubj = Vector{Int}(fill(ntrials, N))
simulated_choice = Matrix{Union{Missing, Int}}(fill(missing, N, ntrials))

# now let's fit the model to the simulated data
sim_model = pvl_delta(simulated_choice, N, Tsubj)
sim_chain = sample(
    sim_model,
    NUTS(),
    MCMCThreads(),
    1000,
    4,
    progress=true,
    verbose=false,
)

# plot the chain
plot(sim_chain)
# save plot
savefig("./figures/pvl_simulated_data.png")

# save chain summary / info to file


io = open("./data/pvl_sim_chain_summary.txt", "w")
show(io, MIME("text/plain"), sim_chain)
show(io, MIME("text/plain"), summarize(sim_chain))
show(io, MIME("text/plain"), hpd(sim_chain))
close(io)

##############################################
# Real Data                                  #
##############################################


# let's try with real data
trial_data = load_trial_data("./data/IGTdataSteingroever2014/IGTdata.rdata")

# all 95 subjects
trial_data_95 = trial_data[trial_data.trial_length .== 95, :]
subjs = unique(trial_data_95.subj)
N = length(subjs)
T = 95
Tsubj = fill(T, N)
choice = reshape(trial_data_95.choice, (N, T))
outcome = reshape(trial_data_95.outcome, (N, T))

model = pvl_delta(N, Tsubj, choice, outcome)

chain = sample(model, HMC(0.05, 10), NUTS(), 1000, 4, progress=true, verbose=false)


# save chain summary / info to file
io = open("./data/pvl_data_95_chain_summary.txt", "w")
show(io, MIME("text/plain"), chain)
show(io, MIME("text/plain"), summarize(chain))
show(io, MIME("text/plain"), hpd(chain))
close(io)

s = summarize(chain)
plot(summarize(chain))














print("Byeeeee...")
print("Byeeeee...")

exit()


##############################################
# Broken Garbage                             #
##############################################


# nsubs = 1
# ntrials = [10]
# X = [[
#     0,
#     100,
#     0,
#     0,
#     0,
#     0,
#     0,
#     0,
#     0,
#     0
# ]]

# model = prospect_theory_model(nsubs, ntrials, X)
# output = sample(model, NUTS(), 1000)
# plot(output)


function prospect_utility(loss_aversion::Real, shape::Real, experienced_reward::Real)
    # Prospect Theory utility function
    utility = (experienced_reward >= 0) ? experienced_reward^shape : -loss_aversion * abs(experienced_reward)^shape
    return utility
end

function delta_learning_rule(prev_value::Real, experienced_utility::Real, updating::Real)
    # Rescorla-Wagner update rule
    new_value = prev_value + updating * (experienced_utility - prev_value)
    return new_value
end

function softmax_choice_rule(expected_values::Vector, sensitivity::Real)
    terms = exp.(sensitivity * expected_values)
    print("terms: ", terms, "\n")
    # Softmax choice rule
    action_probabilities = 
        terms ./ sum(terms)

    return action_probabilities
end


function igt_action_pvl(agent::Agent, input::Vector{Union{Missing, Int64}})
    # input is a value, the outcome of the deck, negative or positive
    
    # get previously selected deck
    prev_action = agent.states["action"]
    print("prev_action: ", prev_action, "\n")
    net_rewards = agent.states["net_rewards"]
    if ismissing(net_rewards)
        net_rewards = [0, 0, 0, 0]
    end

    # only one of the inputs will be non-missing
    for (i, value) in enumerate(input)
        if !ismissing(value)
            net_rewards[i] += value
        end
    end

    if ismissing(prev_action)
        agent.parameters["loss_aversion"] = rand(Normal(
            agent.parameters["loss_aversion_mu"],
            agent.parameters["loss_aversion_sigma"]
        ))
        agent.parameters["shape"] = rand(Normal(
            agent.parameters["shape_mu"],
            agent.parameters["shape_sigma"]
        ))
        agent.parameters["updating"] = rand(Normal(
            agent.parameters["updating_mu"],
            agent.parameters["updating_sigma"]
        ))
        agent.parameters["consistency"] = rand(Normal(
            agent.parameters["consistency_mu"],
            agent.parameters["consistency_sigma"]
        ))
    end
    

    # get params
    loss_aversion = agent.parameters["loss_aversion"] # w, loss aversion parameter (weight of net losses over net rewards)
    shape = agent.parameters["shape"] # A, shape of utility function
    updating = agent.parameters["updating"] # a, the updating parameter, memory for past experiences
    consistency = agent.parameters["consistency"]  #exploration vs exploitation param
    sensitivity = 3^consistency - 1

    # get previous value (states)
    expected_values = agent.states["expected_values"]
    if ismissing(expected_values)
        expected_values = [0.0, 0.0, 0.0, 0.0] # X(t)^A
    end

    

    
    print("prev_action: ", prev_action, "\n")
    print("net_rewards: ", net_rewards, "\n")
    experienced_utility = prospect_utility(loss_aversion, shape, net_rewards[prev_action])
    print("experienced_utility: ", experienced_utility, "\n")

    expected_values[prev_action] = delta_learning_rule(expected_values[prev_action], experienced_utility, updating)

    # update states
    agent.states["expected_values"] = expected_values
    agent.states["net_rewards"] = net_rewards

    # predict next action
    # initially will be 1/4 (or [0.25, 0.25, 0.25, 0.25])
    action_probabilities = softmax_choice_rule(expected_values, sensitivity)
    print("action_probabilities: ", action_probabilities, "\n")
    # final action distribution is Categorical
    action_distribution = Categorical(action_probabilities)

    return action_distribution
end


# @everywhere function init_agent_with_priors_pvl_delta(action_function::Function, priors::Dict)


priors = Dict(
    "loss_aversion_mu" => Normal(0, 1),
    "loss_aversion_sigma" => Uniform(0, 1.5),
    "shape_mu" => Normal(0, 1),
    "shape_sigma" => Uniform(0, 1.5),
    "updating_mu" => Normal(0, 1),
    "updating_sigma" => Uniform(0, 1.5),
    "consistency_mu" => Normal(0, 1),
    "consistency_sigma" => Uniform(0, 1.5),
)

agent = init_agent(
    igt_action_pvl,
    parameters = Dict(
        "loss_aversion" => missing,
        "shape" => missing,
        "updating" => missing,
        "consistency" => missing,
        "loss_aversion_mu" => Normal(0, 1),
        "loss_aversion_sigma" => Uniform(0, 1.5),
        "shape_mu" => Normal(0, 1),
        "shape_sigma" => Uniform(0, 1.5),
        "updating_mu" => Normal(0, 1),
        "updating_sigma" => Uniform(0, 1.5),
        "consistency_mu" => Normal(0, 1),
        "consistency_sigma" => Uniform(0, 1.5),
    ),
    states = Dict(
        "expected_values" => [0.0, 0.0, 0.0, 0.0],
        "net_rewards" => [0, 0, 0, 0],
        "prev_action" => missing,
    ),
)

reset!(agent)

actions = [2,2,2,2,2,2,2,2,2,1]
inputs = [
    [missing, 100, missing, missing],
    [missing, 100, missing, missing],
    [missing, 100, missing, missing],
    [missing, 100, missing, missing],
    [missing, 100, missing, missing],
    [missing, 100, missing, missing],
    [missing, 100, missing, missing],
    [missing, 100, missing, missing],
    [missing, -1250, missing, missing],
    [100, missing, missing, missing]]

result = fit_model(
    agent,
    priors,
    inputs,
    actions,
    fixed_parameters=Dict(
        "loss_aversion" => missing,
        "shape" => missing,
        "updating" => missing,
        "consistency" => missing,
    ),
    n_cores = 1,
)
    # dataframe = dataframe
    # independent_group_cols = [:experiment],
    # if you want to fit seperately for each subject
    # this will run a lot faster
    # independent_group_cols = [:experiment, :ID],
    # multilevel_group_cols = [:ID, :group],
    # input_cols = [:input],
    # action_cols = [:action],
simulated_actions = give_inputs!(agent, [1, 2, 3, 4, 5, 6, -1500, 8, 9, 10])
