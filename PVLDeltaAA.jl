#using ForwardDiff
#using Preferences
# using ActionModels
# using Distributed
using ReverseDiff, ForwardDiff, Tracker, Distributions, FillArrays, Optim, Turing, StatsFuns
using HDF5, MCMCChains, MCMCChainsStorage
using Tracker
using Turing: AutoReverseDiff, AutoForwardDiff

# using StatsBase
# using TypeUtils
# using ForwardDiff: value
include("src/Data.jl")



# set_preferences!(ForwardDiff, "nansafe_mode" => true)
delete_existing_chains = false
skip_existing_chains = true
progress = true
Turing.setprogress!(progress)
# Turing.setadbackend(:reversediff) # deprecated

struct CategoricalLogit <: Distributions.DiscreteUnivariateDistribution

    logitp::Vector{Float64}
    ncats::Int
end

# function Distributions.probs(d::CategoricalLogit)
#     return exp.(d.logitp .- logsumexp(d.logitp))
# end

function Base.convert(::Type{T}, x::Tracker.TrackedReal{Tt}) where {T <: Integer, Tt <: Real}
    if (T <: Integer)
        return convert(T, round(ForwardDiff.value(x)))
    end
    return convert(T, ForwardDiff.value(x))
end

function Base.convert(::Type{R}, t::T) where {R<:Real,T<:ReverseDiff.TrackedReal}
    if (R <: Integer)
        return convert(R, round(ReverseDiff.value(t)))
    end
    return convert(R, ReverseDiff.value(t))
end

function Base.convert(::Type{T}, x::Tracker.TrackedReal{Tracker.TrackedReal{Tt}}) where {T <: Real, Tt <: Real}
    return convert(T, ForwardDiff.value(ForwardDiff.value(x)))
end

function Distributions.insupport(d::CategoricalLogit, k::Real)
    return isinteger(k) && 1 <= k <= d.ncats
end

function Distributions.logpdf(d::CategoricalLogit, k::Real)
    k = convert(Int, k)
    r = (d.logitp .- logsumexp(d.logitp))[k]
    return r
end

function Base.minimum(d::CategoricalLogit)
    first(support(d))
end

function Base.maximum(d::CategoricalLogit)
    last(support(d))
end

function Distributions.support(d::CategoricalLogit)
    return Base.OneTo(d.ncats)
end

Distributions.sampler(d::CategoricalLogit) = Distributions.AliasTable(probs(d))

# function Distributions.quantile(d::CategoricalLogit, p; sorted::Bool=false, alpha::Real=1.0, beta::Real=alpha)
#     return Distributions._quantile(d, p, sorted, alpha, beta)
    
# end

function Base.convert(::Type{CategoricalLogit}, p::AbstractVector{<:Real})
    return CategoricalLogit(p, length(p))
end

# implementation of the rand function for categorical (logit)
function Distributions.rand(rng::AbstractRNG, d::CategoricalLogit)
    x = support(d)
    p = probs(d)
    n = length(p)
    draw = rand(rng, float(eltype(p)))
    cp = p[1]
    i = 1
    while cp <= draw && i < n
        @inbounds cp += p[i +=1]
    end
    return x[i]
end

function Distributions.ncategories(d::CategoricalLogit)
    return d.ncats
end

# function Base.iterate(d::CategoricalLogit, state::Int = 1)
#     if state > length(d.logitp)
#         return nothing
#     end
#     return state, state + 1
# end

# function Base.getindex(d::CategoricalLogit, i::Int)
#     return d.logitp[i]
# end

# function Base.lastindex(d::CategoricalLogit)
#     return d.ncats
# end

# function Base.collect(d::CategoricalLogit)
#     return collect(1:d.ncats)
# end

# function Distributions.ncategories(d::CategoricalLogit)
#     return d.ncats
# end

# Distributions.params(d::CategoricalLogit) = (d.logitp, d.ncats)
function noisy_softmax(x::Vector, noise::Real)
    return exp.(x / noise) / sum(exp.(x / noise))
end

@model function pvl_delta(actions::Matrix{Union{Missing, Int}}, ::Type{T} = Float64; N::Int, Tsubj::Vector{Int}, payoff_scheme::Vector{Union{Missing, Int}}) where {T}
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
        θ = log(3^c[i] - 1)
        payoffs = construct_payoff_sequence(payoff_scheme[i])
        for t in 1:Tsubj[i]
            # Get choice probabilities (random on first trial, otherwise softmax of expected utility)
            # softmax choice (draw)
            # pₖ = t == 1 ? fill(0.25, 4) : softmax(Tracker.value(θ) .* Tracker.value(Evₖ[t-1, :]))
            pₖ = t == 1 ? fill(0.25, 4) : noisy_softmax(Evₖ[t-1, :], θ)
            # try in log space
            # pₖ = t == 1 ? fill(log(0.25), 4) : θ .+ Evₖ[t-1, :]
            # try
            # kₜ ~ CategoricalLogit(pₖ, 4)
            # actions[i, t] = kₜ
            # actions[i, t] ~ CategoricalLogit(pₖ, 4)
            actions[i, t] ~ Categorical(pₖ, 4)
            # catch e
            #     println("t: ", t)
            #     println("i: ", i)
            #     if t > 1
            #         println("Evₖ[t-1, :]: ", Evₖ[t-1, :])
            #     end
            #     println("θ: ", θ)
            #     println("c[i]", c[i])
            #     println("w[i]", w[i])
            #     println("A[i]", A[i])
            #     println("a[i]", a[i])
            #     println("pₖ ", pₖ)
            #     throw(e)
            # end
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
            for j in 1:4
                if j != k
                    Evₖ[t, j] = Evₖ₍ₜ₋₁₎[j]
                end
            end
        end
    end
    return actions
end



# N = 3
# ntrials = 50
# Tsubj = Vector{Int}(fill(ntrials, N))
# simulated_choice = Matrix{Union{Missing, Int}}(fill(missing, N, ntrials))

# # now let's fit the model to the simulated data
# sim_model = pvl_delta(simulated_choice, N, Tsubj)
# sim_chain = sample(
#     sim_model,
#     NUTS(),
#     MCMCThreads(),
#     1000,
#     4,
#     progress=true,
#     verbose=false,
# )

# # plot the chain
# plot(sim_chain)
# # save plot
# savefig("./figures/pvl_simulated_data.png")

# # save chain summary / info to file


# io = open("./data/pvl_sim_chain_summary.txt", "w")
# show(io, MIME("text/plain"), sim_chain)
# show(io, MIME("text/plain"), summarize(sim_chain))
# show(io, MIME("text/plain"), hpd(sim_chain))
# close(io)

##############################################
# Real Data                                  #
##############################################


# let's try with real data
trial_data = load_trial_data("./data/IGTdataSteingroever2014/IGTdata.rdata")
# add a "choice pattern" column
# 1 = CD >= 0.65, 2 = AB >= 0.65, 4 = BD >= 0.65, 8 = AC >= 0.65
trial_data.choice_pattern_ab = ifelse.(trial_data.ab_ratio .>= 0.65, 1, 0)
trial_data.choice_pattern_cd = ifelse.(trial_data.cd_ratio .>= 0.65, 2, 0)
trial_data.choice_pattern_bd = ifelse.(trial_data.bd_ratio .>= 0.65, 4, 0)
trial_data.choice_pattern_ac = ifelse.(trial_data.ac_ratio .>= 0.65, 8, 0)

trial_data.choice_pattern = trial_data.choice_pattern_ab .| trial_data.choice_pattern_cd .| trial_data.choice_pattern_bd .| trial_data.choice_pattern_ac
# just one subject to test
#trial_data = trial_data[trial_data.subj .== 1, :]
#trial_data = trial_data[trial_data.study .== "Steingroever2011", :]

#patterns
pats = unique(trial_data.choice_pattern)

# get number of unique subjects for each pattern
n_subj = [length(unique(trial_data[trial_data.choice_pattern .== pat, :subj])) for pat in pats]

chain_out_file = "./data/igt_pvldelta_data_chains.h5"

# delete chain file if it exists
processed_patterns = []
if delete_existing_chains && isfile(chain_out_file)
    print("Deleting file: $chain_out_file")
    rm(chain_out_file)
elseif skip_existing_chains && isfile(chain_out_file)
    # get patterns that have already been processed
    h5open(chain_out_file, "r") do file
        processed_patterns = keys(file)
        # pats = setdiff(pats, processed_patterns)
    end
end
# print out
chains::Dict{String, Chains} = Dict()
for (pat, n) in zip(pats, n_subj)
    println("Pattern: $pat, n = $n")
    if "pattern_$pat" in processed_patterns
        println("Pattern $pat already processed, skipping...")
        continue
    end

    trial_data_pat = trial_data[trial_data.choice_pattern .== pat, :]
    subjs = unique(trial_data_pat.subj)
    N = length(subjs)
    Tsubj = [length(trial_data_pat[trial_data_pat.subj .== subj, :subj]) for subj in subjs]
    choice = Matrix{Union{Missing, Int}}(undef, N, maximum(Tsubj))
    # outcome = Matrix{Union{Missing, Float64}}(undef, N, maximum(Tsubj))
    payoff_schemes = Vector{Union{Missing, Int}}(undef, N)
    for (i, subj) in enumerate(subjs)
        subj_data = trial_data_pat[trial_data_pat.subj .== subj, :]
        choice[i, 1:Tsubj[i]] = subj_data.choice
        # outcome[i, 1:Tsubj[i]] = subj_data.outcome
        payoff_schemes[i] = subj_data.scheme[1]
    end

    model = pvl_delta(choice; N=N, Tsubj=Tsubj, payoff_scheme=payoff_schemes)

    chain = sample(
        model,
        NUTS(; adtype=AutoReverseDiff(true)),
        MCMCThreads(), # disable for debugging
        3000,
        3,
        progress=progress,
        verbose=false;
        save_state=true
    )
    chains["pattern_$pat"] = chain

    # save chain
    pattern_name = "pattern_$pat"
    h5open(chain_out_file, "cw") do file
        g = create_group(file, pattern_name)
        write(g, chain)
    end

end

# load chain

# h5open(chain_out_file, "r") do file
#     for pat in pats
#         pattern_name = "pattern_$pat"
#         chains[pattern_name] = read(file[pattern_name], Chains)
#     end
# end


rand(Categorical([0.25, 0.25, 0.25, 0.25]), 1)
# # save chain summary / info to file
# io = open("./data/pvl_data_95_chain_summary.txt", "w")
# show(io, MIME("text/plain"), chain)
# show(io, MIME("text/plain"), summarize(chain))
# show(io, MIME("text/plain"), hpd(chain))
# close(io)

# s = summarize(chain)
# plot(summarize(chain))














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
