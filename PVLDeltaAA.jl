
using Distributions
using Plots, StatsPlots
using ActionModels
using Distributed
using Turing
using FillArrays
using StatsFuns
using JLD

include("src/Data.jl")




Turing.setprogress!(true)


Φ(x::Real) = cdf(Normal(0, 1), x)
@model function pvl_delta(N::Int, Tsubj::Vector{Int}, choice::Matrix{Int}, outcome::Matrix{Float64}, ::Type{T} = Float64) where {T}
    # Hyperparameters
    A_mu_pr ~ Normal(0, 1)
    A_sigma ~ Uniform(0, 1.5)
    alpha_mu_pr ~ Normal(0, 1)
    alpha_sigma ~ Uniform(0, 1.5)
    cons_mu_pr ~ Normal(0, 1)
    cons_sigma ~ Uniform(0, 1.5)
    lambda_mu_pr ~ Normal(0, 1)
    lambda_sigma ~ Uniform(0, 1.5)


    # individual parameters
    A_pr ~ filldist(Normal(0, 1), N)
    alpha_pr ~ filldist(Normal(0, 1), N)
    cons_pr ~ filldist(Normal(0, 1), N)
    lambda_pr ~ filldist(Normal(0, 1), N)

    # Transform subject-level raw parameters
    A = Φ.(A_mu_pr .+ A_sigma .* A_pr)
    alpha = Φ.(alpha_mu_pr .+ alpha_sigma .* alpha_pr) .* 2
    cons = Φ.(cons_mu_pr .+ cons_sigma .* cons_pr) .* 5
    lambda = Φ.(lambda_mu_pr .+ lambda_sigma .* lambda_pr) .* 10

    # For log likelihood calculation
    log_lik = Vector{T}(undef, N)

    for i in 1:N
        # Define values
        ev = zeros(T, 4)
        curUtil = 0.0
        theta = 0.0

        # Initialize values
        theta = 3^cons[i] - 1

        for t in 1:Tsubj[i]
            # softmax choice
            log_lik[i] += log(softmax(theta .* ev)[choice[i, t]])


            if outcome[i, t] >= 0
                curUtil = outcome[i, t]^alpha[i]
            else
                curUtil = -1 * lambda[i] * (-1 * outcome[i, t])^alpha[i]
            end

            # delta
            ev[choice[i, t]] += A[i] * (curUtil - ev[choice[i, t]])
        end
    end

    return log_lik
end














##############################################
# Simulated Data                             #
##############################################

# now let's simulate some data
N = 15  # Number of subjects
T = 95   # Max number of trials
Tsubj = fill(T, N)  # Number of trials per subject
# sample from priors
A_mu_pr = rand(Normal(0, 1))
A_sigma = rand(Uniform(0, 1.5))
alpha_mu_pr = rand(Normal(0, 1))
alpha_sigma = rand(Uniform(0, 1.5))
cons_mu_pr = rand(Normal(0, 1))
cons_sigma = rand(Uniform(0, 1.5))
lambda_mu_pr = rand(Normal(0, 1))
lambda_sigma = rand(Uniform(0, 1.5))

# individual parameters
A_pr = rand(Normal(0, 1), N)
alpha_pr = rand(Normal(0, 1), N)
cons_pr = rand(Normal(0, 1), N)
lambda_pr = rand(Normal(0, 1), N)

# Transform subject-level raw parameters
A = Φ.(A_mu_pr .+ A_sigma .* A_pr)
alpha = Φ.(alpha_mu_pr .+ alpha_sigma .* alpha_pr) .* 2
cons = Φ.(cons_mu_pr .+ cons_sigma .* cons_pr) .* 5
lambda = Φ.(lambda_mu_pr .+ lambda_sigma .* lambda_pr) .* 10

# now create simulated data from the model priors / assumptions
simulated_choice = zeros(Int, N, T)
simulated_outcome = zeros(Float64, N, T)


for i in 1:N
    # Define values
    ev = zeros(T, 4)
    curUtil = 0.0
    theta = 0.0
    # each subject gets their own decks
    payoff_scheme = construct_payoff_sequence(1)
    # Initialize values
    theta = 3^cons[i] - 1
    for t in 1:Tsubj[i]
        # softmax choice
        simulated_choice[i, t] = rand(Categorical(softmax(theta .* ev[i, :])))
        # get the result for the selected deck
        simulated_outcome[i, t] = igt_deck_payoff!(simulated_choice[i, 1:t], payoff_scheme)
        if simulated_outcome[i, t] >= 0
            curUtil = simulated_outcome[i, t]^alpha[i]
        else
            curUtil = -1 * lambda[i] * (-1 * simulated_outcome[i, t])^alpha[i]
        end
        # delta
        ev[i, simulated_choice[i, t]] += A[i] * (curUtil - ev[simulated_choice[i, t]])
    end
end

# now let's fit the model to the simulated data
sim_model = pvl_delta(N, Tsubj, simulated_choice, simulated_outcome)
sim_chain = sample(sim_model, HMC(0.05, 10), MCMCThreads(), 1000, 4, progress=true, verbose=true)

# save chain
save("./data/pvl_sim_chain.jld", "chain", sim_chain, compress=true)

# get mode for the 4 parameters
g_A_mu_pr_mode = mode(sim_chain[:"A_mu_pr"])
g_A_sigma_mode = mode(sim_chain[:"A_sigma"])

g_alpha_mu_pr_mode = mode(sim_chain[:"alpha_mu_pr"])
g_alpha_sigma_mode = mode(sim_chain[:"alpha_sigma"])

g_cons_mu_pr_mode = mode(sim_chain[:"cons_mu_pr"])
g_cons_sigma_mode = mode(sim_chain[:"cons_sigma"])

g_lambda_mu_pr_mode = mode(sim_chain[:"lambda_mu_pr"])
g_lambda_sigma_mode = mode(sim_chain[:"lambda_sigma"])


# get the modes for each subject
i_A_pr = fill(NaN, N)
i_alpha_pr = fill(NaN, N)
i_cons_pr = fill(NaN, N)
i_lambda_pr = fill(NaN, N)
for i in 1:N
    i_A_pr[i] = mode(sim_chain[Symbol("A_pr[$i]")])
    i_alpha_pr[i] = mode(sim_chain[Symbol("alpha_pr[$i]")])
    i_cons_pr[i] = mode(sim_chain[Symbol("cons_pr[$i]")])
    i_lambda_pr[i] = mode(sim_chain[Symbol("lambda_pr[$i]")])
end

g_A_mode = mean(Φ.(g_A_mu_pr_mode .+ g_A_sigma_mode * i_A_pr))
g_alpha_mode = mean(Φ.(g_alpha_mu_pr_mode .+ g_alpha_sigma_mode * i_alpha_pr) .* 2)
g_cons_mode = mean(Φ.(g_cons_mu_pr_mode .+ g_cons_sigma_mode * i_cons_pr) .* 5)
g_lambda_mode = mean(Φ.(g_lambda_mu_pr_mode .+ g_lambda_sigma_mode * i_lambda_pr) .* 10)


i_A = Matrix{Real}(undef, N, 3)
i_alpha = Matrix{Real}(undef, N, 3)
i_cons = Matrix{Real}(undef, N, 3)
i_lambda = Matrix{Real}(undef, N, 3)
# combine A and i_A_pr
for i in 1:N
    i_A[i, 2] = i
    i_A[i, 3] = i_A_pr[i]
    i_A[i, 1] = A[i]

    i_alpha[i, 2] = i
    i_alpha[i, 3] = i_alpha_pr[i]
    i_alpha[i, 1] = alpha[i]

    i_cons[i, 2] = i
    i_cons[i, 3] = i_cons_pr[i]
    i_cons[i, 1] = cons[i]

    i_lambda[i, 2] = i
    i_lambda[i, 3] = i_lambda_pr[i]
    i_lambda[i, 1] = lambda[i]
end
# sort by A
i_A = sortslices(i_A, dims=1, lt=(x,y) -> x[1] < y[1])
i_alpha = sortslices(i_alpha, dims=1, lt=(x,y) -> x[1] < y[1])
i_cons = sortslices(i_cons, dims=1, lt=(x,y) -> x[1] < y[1])
i_lambda = sortslices(i_lambda, dims=1, lt=(x,y) -> x[1] < y[1])


# plot A with line for group level mode and recovered group level mode
p1 = plot(1:N, i_A[:, 1], label="Simulated A", xlabel="Subject", ylabel="A", title="A group and individual level")
plot!(1:N, i_A[:, 3], label="Recovered A", line=:dash)
plot!([1, N], [g_A_mode, g_A_mode], label="Recovered group level mode", line=:dash)
plot!([1, N], [mean(A), mean(A)], label="Group level mode", line=:dot)

# plot alpha with line for group level mode and recovered group level mode
p2 = plot(1:N, i_alpha[:, 1], label="Simulated alpha", xlabel="Subject", ylabel="alpha", title="alpha group and individual level")
plot!(1:N, i_alpha[:, 3], label="Recovered alpha", line=:dash)
plot!([1, N], [g_alpha_mode, g_alpha_mode], label="Recovered group level mode", line=:dash)
plot!([1, N], [mean(alpha), mean(alpha)], label="Group level mode", line=:dot)

# plot cons with line for group level mode and recovered group level mode
p3 = plot(1:N, i_cons[:, 1], label="Simulated cons", xlabel="Subject", ylabel="cons", title="cons group and individual level")
plot!(1:N, i_cons[:, 3], label="Recovered cons", line=:dash)
plot!([1, N], [g_cons_mode, g_cons_mode], label="Recovered group level mode", line=:dash)
plot!([1, N], [mean(cons), mean(cons)], label="Group level mode", line=:dot)

# plot lambda with line for group level mode and recovered group level mode
p4 = plot(1:N, i_lambda[:, 1], label="Simulated lambda", xlabel="Subject", ylabel="lambda", title="lambda group and individual level")
plot!(1:N, i_lambda[:, 3], label="Recovered lambda", line=:dash)
plot!([1, N], [g_lambda_mode, g_lambda_mode], label="Recovered group level mode", line=:dash)
plot!([1, N], [mean(lambda), mean(lambda)], label="Group level mode", line=:dot)

plot(p1, p2, p3, p4, layout=(2, 2), legend=:outertop, size=(1600, 1600))

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

chain = sample(model, HMC(0.05, 10), MCMCThreads(), 1000, 4, progress=true, verbose=true)

# save chain
save("./data/pvl_data_95_chain.jld", "chain", chain, compress=true)

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
