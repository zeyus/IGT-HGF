using ReverseDiff, Distributions, FillArrays, Optim, Turing, StatsFuns
using HDF5, MCMCChains, MCMCChainsStorage
using Tracker
using Turing: AutoForwardDiff, ForwardDiff, AutoReverseDiff

include("src/Data.jl")
delete_existing_chains = false
skip_existing_chains = true
progress = true
optim_param_est = false

Turing.setprogress!(progress)

function ad_val(x::ReverseDiff.TrackedReal)
    return ReverseDiff.value(x)
end
function ad_val(x::ReverseDiff.TrackedArray)
    return ReverseDiff.value(x)
end

function ad_val(x::ForwardDiff.Dual)
    return ForwardDiff.value(x)
end

function ad_val(x::Real)
    return x
end

struct CategoricalLogit <: Distributions.DiscreteUnivariateDistribution
    logitp::AbstractArray{<:Real, 1}
    ncats::Int
end

# # This might have unintended consequences
# function Base.convert(::Type{R}, t::T) where {R<:Real,T<:ReverseDiff.TrackedReal}
#     if (R <: Integer)
#         return convert(R, round(ReverseDiff.value(t)))
#     end
#     return convert(R, ReverseDiff.value(t))
# end

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

function action_probabilities(x::AbstractVector{<:Real}, τ::Real)
    xₘₐₓ = maximum(x)
    xₙ = x .- xₘₐₓ
    return exp.(xₙ * τ) / sum(exp.(xₙ * τ))
end

function logit_action_probabilities(x::AbstractVector{<:Real}, τ::Real)
    xₘₐₓ = maximum(x)
    xₙ = x .- xₘₐₓ
    xₙ = exp.(logistic.(xₙ) * τ)
    return logit.(xₙ / sum(xₙ)) # + xₘₐₓ # ???
end

@model function pvl_delta(actions::Matrix{Union{Missing, Int}}, ::Type{T} = Float64; N::Int, Tsubj::Vector{Int}, deck_payoffs::Array{Int, 4}) where {T}
    # Group Level Parameters
    # Group level Shape mean
    A′μ ~ Normal(0, 1)
    # Group level Shape standard deviation
    A′σ ~ Uniform(0, 1.5)

    # Group level updating parameter mean
    a′μ ~ Normal(0, 1)
    # Group level updating parameter standard deviation
    a′σ ~ Uniform(0, 1.5)

    # Group level response consistency mean
    c′μ ~ Normal(0, 1)
    # Group level response consistency standard deviation
    c′σ ~ Uniform(0, 1.5)

    # Group Level Loss-Aversion mean
    w′μ ~ Normal(0, 1)
    # Group Level Loss-Aversion standard deviation
    w′σ ~ Uniform(0, 1.5)

    # individual parameters
    # Shape
    A′ ~ filldist(LogitNormal(A′μ, A′σ), N)
    # Updating parameter
    a′ ~ filldist(LogitNormal(a′μ, a′σ), N)
    # Response consistency
    c′ ~ filldist(truncated(LogNormal(c′μ, c′σ); upper=3), N)
    # Loss-Aversion
    w′ ~ filldist(truncated(LogNormal(w′μ, w′σ); upper=3), N)

    # create actions matrix if using the model for simulation
    if actions === missing
        actions = Matrix{Union{Missing, Int}}(undef, N, maximum(Tsubj))
    end

    for i in 1:N
        Aᵢ = A′[i] # Aᵢ = ϕ(A′ᵢ) -> achieved through logitnormal
        aᵢ = a′[i] # aᵢ = ϕ(a′ᵢ) -> achieved through logitnormal
        cᵢ = c′[i] # cᵢ = ϕ(c′ᵢ) * 5  -> achieved through truncated LogNormal
        wᵢ = w′[i] # wᵢ = ϕ(w′ᵢ) * 5 -> achieved through truncated LogNormal
        # Create expected value matrix
        Evₖ = zeros(T, Tsubj[i], 4)
        # Set theta (exploration vs exploitation)
        θ = 3^cᵢ - 1

        # start each deck at the first card
        deck_sequence_index = [1, 1, 1, 1]

        for t in 2:(Tsubj[i])
            # Get previous expected values (all decks start at 0)
            Evₖ₍ₜ₋₁₎ = Evₖ[t-1, :]

            # softmax choice
            Pₖ = action_probabilities(Evₖ₍ₜ₋₁₎, θ)

            # draw from categorical distribution based on softmax
            actions[i, t-1] ~ Categorical(Pₖ)

            # get selected deck: ad_val function avoids switching functions if using a different AD backend
            k = ad_val(actions[i, t-1])

            # get payoff for selected deck
            Xₜ = deck_payoffs[i, deck_sequence_index[k], k, 1]

            # get win/loss status: loss = 1, win = 0
            l = deck_payoffs[i, deck_sequence_index[k], k, 2]

            # increment deck sequence index (this is because each deck has a unique payoff sequence)
            deck_sequence_index[k] += 1

            # get prospect utility -> same as paper but without having to use logic: uₖ = (Xₜ >= 0) ? Xₜ^Aᵢ : -wᵢ * abs(Xₜ)^Aᵢ
            Xₜᴾ = abs(Xₜ)^Aᵢ
            uₖ = (l * -wᵢ * Xₜᴾ) + ((1 - l) * Xₜᴾ)

            # update expected value of selected deck, carry forward the rest
            Evₖ[t, :] = Evₖ₍ₜ₋₁₎

            # delta learning rule
            Evₖ[t, k] += aᵢ * (uₖ - Evₖ₍ₜ₋₁₎[k])
        end
    end
    return actions
end



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
# trial_data = trial_data[trial_data.subj .== 1, :]
# trial_data = trial_data[trial_data.study .== "Steingroever2011", :]

# patterns
pats = unique(trial_data.choice_pattern)

# get number of unique subjects for each pattern
n_subj = [length(unique(trial_data[trial_data.choice_pattern .== pat, :subj])) for pat in pats]

chain_out_file = "./data/igt_pvldelta_data_chains.h5"

# delete chain file if it exists
processed_patterns = []
if delete_existing_chains && isfile(chain_out_file)
    print("Deleting file: $chain_out_file, are you sure? (y/n)")
    # wait for user confirmation
    conf = readline()
    if conf == "y"
        rm(chain_out_file)
    else
        exit()
    end
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
    trial_data_pat.subj_uid = join.(zip(trial_data_pat.study, trial_data_pat.subj), "_")
    # get unique subjects (subject_id and study_id)
    subjs = unique(trial_data_pat.subj_uid) 
    N = length(subjs)
    Tsubj = [length(trial_data_pat[trial_data_pat.subj_uid .== subj, :subj]) for subj in subjs]
    choice = Matrix{Union{Missing, Int}}(undef, N, maximum(Tsubj))

    # this array is overlarge but not sure how to reduce it
    # sparse array maybe?
    deck_payoffs = Array{Int, 4}(undef, N, maximum(Tsubj), 4, 2)
    payoff_schemes = Vector{Int}(undef, N)
    for (i, subj) in enumerate(subjs)
        subj_data = trial_data_pat[trial_data_pat.subj_uid .== subj, :]
        choice[i, 1:Tsubj[i]] = subj_data.choice
        for j in 1:4
            results_j = round.(Int, subj_data[subj_data.choice .== j, :outcome])
            n_results_j = length(results_j)
            deck_payoffs[i, 1:n_results_j, j, 1] = results_j
            deck_payoffs[i, 1:n_results_j, j, 2] = Int.(results_j .< 0)
        end
        payoff_schemes[i] = subj_data.scheme[1]
    end
    # if simulated
    # deck_payoffs = construct_payoff_matrix_of_length(N, Tsubj, payoff_schemes)
    model = pvl_delta(choice; N=N, Tsubj=Tsubj, deck_payoffs=deck_payoffs)

    # generate MAP estimate
    n_chains = 3
    estimated_params = nothing
    if optim_param_est
        println("Using Optim to estimate initial parameters for $pat...")
        est_start = time()
        mle_estimate = optimize(model, MLE(), LBFGS(); autodiff = :reverse)
        est_time = time() - est_start
        println("Estimation for $pat took $est_time seconds")
        if Optim.converged(mle_estimate.optim_result)
            println("MLE estimate converged for $pat, using as initial parameters for sampling...")
            println(mle_estimate.values.array)
            print(mle_estimate)
            estimated_params = [repeat([mle_estimate.values.array], n_chains)...]
        else
            println("MLE estimate did not converge for $pat")
        end
    else
        println("Not using Optim to estimate initial parameters for $pat...")
    end
    # defaults: -1: uses max(n_samples/2, 500) ?!?, 0.65: target acceptance rate; Adtype is set to ForwardDiff by default
    # sampler = NUTS()
    # sampler = NUTS(500, 0.65; max_depth=10, adtype=AutoForwardDiff(; chunksize=0))
    sampler = NUTS(500, 0.65; max_depth=10, adtype=AutoReverseDiff(true)) 

    chain = sample(
        model,
        sampler,
        MCMCThreads(), # disable for debugging
        1_000,
        n_chains,
        progress=progress,
        verbose=false;
        save_state=false,
        initial_params=estimated_params,
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
