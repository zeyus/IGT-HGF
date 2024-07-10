# using CUDA
using Distributed
using Base.Threads: @threads
using Base: RefValue
using LinearAlgebra, ForwardDiff, Zygote, ReverseDiff, Distributions, FillArrays, Optim, Turing, StatsFuns
using Feather, HDF5, MCMCChains, MCMCChainsStorage, DataFrames
using Turing: AutoForwardDiff, ForwardDiff, AutoReverseDiff, AutoZygote



include("src/Data.jl")
include("src/LogCommon.jl")

delete_existing_chains = false
skip_existing_chains = true

# show progress bar
progress = true

# use Optim to estimate starting parameter values
optim_param_est = false
optim_est_type = :map  # :mle or :map

# if not using Optim, should we generate random starting parameters?
rand_param_est = false

# set AD backend
adtype = AutoReverseDiff(; compile=true)
# adtype = AutoZygote()
# number of warmup samples
n_warmup = 1_000
# target acceptance rate
target_accept = 0.5
# max tree depth
max_depth = 30
# initial step size
init_ϵ = 0.0 # if 0, turing will try to estimate it
# max divergence during doubling
# Δ_max = 
@info "Using NUTS Sampler with $n_warmup warmup samples, $target_accept target acceptance rate, $max_depth max tree depth, and $init_ϵ initial step size."
sampler = NUTS(
    n_warmup, # warmup samples
    target_accept; # target acceptance rate
    max_depth=max_depth, # max tree depth
    init_ϵ=init_ϵ, # initial step size
    adtype=adtype) 
# @info "Using HMCDA Sampler..."
# sampler = HMCDA(
#     n_warmup, # n adapt steps
#     target_accept, # target acceptance rate
#     0.3; # target leapfrog length
#     adtype=adtype
# )
# @info "Using HMC Sampler..."
# sampler = HMC(
#     0.1, # step size
#     5; # number of steps
#     adtype=adtype
# )
# @info "Using Particle Gibbs Sampler...(NOT WORKING)"
# sampler = PG(
#     10
# )

# number of chains to run
n_chains = 3
n_samples = 2_000

@info "Sampling will use $n_chains chains with $n_samples samples each."


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

function ad_val(x::T) where {T <: Real}
    return x
end
function ad_val(x::AbstractArray{ForwardDiff.Dual{T, V, N}}) where {T, V, N}
    return ForwardDiff.value.(x)
end
function ad_val(x::AbstractArray{ReverseDiff.TrackedReal{V, D, O}}) where {V, D, O}
    return ReverseDiff.value.(x)
end
function ad_val(x::AbstractArray{ReverseDiff.TrackedArray})
    return ReverseDiff.value.(x)
end
function ad_val(x::AbstractArray{T}) where {T <: AbstractFloat}
    return x
end

struct CategoricalLogit <: Distributions.DiscreteUnivariateDistribution
    logitp::AbstractArray{<:Real, 1}
    ncats::Int
end

# fix for convert for initial_params
function Base.convert(::Type{T}, t::Pair{String, Float64}) where {T<:Real}
    return t.second
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


# Helper function to calculate the inverse logit
function inv_logit(x::T) where T
    return 1 / (1 + exp(-x))
end

# Define the Phi_approx function
function Φₐₚₚᵣₒₓ(x)
    av_squared = x * x
    inv_logit(0.07056 * x * av_squared + 1.5976 * x)
end

# Define the custom adjoint for phi_approx
ReverseDiff.@forward function Φₐₚₚᵣₒₓ(x)
    av_squared = x * x
    f = inv_logit(0.07056 * x * av_squared + 1.5976 * x)
    da = f * (1 - f) * (3.0 * 0.07056 * av_squared + 1.5976)
    return f, (outgrad) -> outgrad * da
end


function action_probabilities!(Xᵢₙ::Vector{Float64}, Xₒᵤₜ::Vector{Float64}, x::AbstractArray{T}, θ::P) where {T <: Real, P <: Real}
    Xᵢₙ .= exp.((x .- maximum(x)) * θ)
    Xₒᵤₜ .= Xᵢₙ / sum(Xᵢₙ)
end

function action_probabilities(x::AbstractArray{T}, θ::P, N::Int = 4) where {T <: Real, P <: Real}
    Xᵢₙ = Vector{Float64}(undef, N)
    Xₒᵤₜ = Vector{Float64}(undef, N)
    action_probabilities!(Xᵢₙ, Xₒᵤₜ, ad_val(x), ad_val(θ))
    return Xₒᵤₜ
end


function logit_action_probabilities(x::AbstractVector{<:Real}, τ::Real)
    xₘₐₓ = maximum(x)
    xₙ = x .- xₘₐₓ
    xₙ = exp.(logistic.(xₙ) * τ)
    return logit.(xₙ / sum(xₙ)) # + xₘₐₓ # ???
end

@info "Defining Model..."
@model function pvl_delta(actions::Matrix{Union{Missing, Int}}, N::Int, Tsubj::Vector{Int}, deck_payoffs::Array{Float64, 3}, deck_wl::Array{Int, 3}, ::Type{T} = Float64) where {T}
    # min_pos_float = 0.0 + eps(0.0)
    min_pos_float = 1e-15 # eps is giving problems, for now set it to a very low value
    # Group Level Parameters
    # Group level Shape mean
    A′μ ~ Normal(0, 1)
    # Group level Shape standard deviation
    A′σ ~ Uniform(min_pos_float, 1.5)

    # Group level updating parameter mean
    a′μ ~ Normal(0, 1)
    # Group level updating parameter standard deviation
    a′σ ~ Uniform(min_pos_float, 1.5)

    # Group level response consistency mean
    c′μ ~ Normal(0, 1)
    # Group level response consistency standard deviation
    c′σ ~ Uniform(min_pos_float, 1.5)

    # Group Level Loss-Aversion mean
    w′μ ~ Normal(0, 1)
    # Group Level Loss-Aversion standard deviation
    w′σ ~ Uniform(min_pos_float, 1.5)

    # @info "Parameter values: "
    # @info A′μ, A′σ, a′μ, a′σ, c′μ, c′σ, w′μ, w′σ
    # # Shape
    # A′_pr ~ filldist(Normal(0, 1), N)
    # # Updating parameter
    # a′_pr ~ filldist(Normal(0, 1), N)
    # # Response consistency
    # c′_pr ~ filldist(Normal(0, 1), N)
    # # Loss-Aversion
    # w′_pr ~ filldist(Normal(0, 1), N)
    # Shape
    A′_pr ~ filldist(Normal(A′μ, A′σ), N)
    # Updating parameter
    a′_pr ~ filldist(Normal(a′μ, a′σ), N)
    # Response consistency
    c′_pr ~ filldist(Normal(c′μ, c′σ), N)
    # Loss-Aversion
    w′_pr ~ filldist(Normal(w′μ, w′σ), N)


    # vectorized version of the loop for setting subject-level params
    # A′ = Φₐₚₚᵣₒₓ.(A′μ .+ A′σ .* A′_pr)
    # a′ = Φₐₚₚᵣₒₓ.(a′μ .+ a′σ .* a′_pr) .* 2
    # c′ = Φₐₚₚᵣₒₓ.(c′μ .+ c′σ .* c′_pr) .* 5
    # w′ = Φₐₚₚᵣₒₓ.(w′μ .+ w′σ .* w′_pr) .* 10
    # θ = 3 .^ c′ .- 1

    A′ = Φₐₚₚᵣₒₓ.(A′_pr)
    a′ = Φₐₚₚᵣₒₓ.(a′_pr) .* 2
    c′ = Φₐₚₚᵣₒₓ.(c′_pr) .* 5
    w′ = Φₐₚₚᵣₒₓ.(w′_pr) .* 10
    θ = 3 .^ c′ .- 1

    max_trials = maximum(Tsubj)

    # create actions matrix if using the model for simulation
    if actions === missing
        actions = Matrix{Union{Missing, Int}}(undef, N, max_trials)
    end
    # Set up expected values and deck sequence index
    Evₖ = zeros(T, max_trials, 4, N)
    deck_sequence_index = ones(Int, N, 4)
    # print type of Evₖ
    # @info typeof(Evₖ)
    # print shape of actions
    # @info size(actions)
    # print maximum(Tsubj)
    # @info max_trials
    # loop over subjects
    for i in 1:N
        # loop over trials
        @inbounds n_trials = Tsubj[i]
        for t in 2:n_trials
            # Get previous expected values (all decks start at 0)
            Evₖ₍ₜ₋₁₎ = Evₖ[t-1, :, i]

            # softmax choice
            Pₖ = action_probabilities(Evₖ₍ₜ₋₁₎, θ[i])

            # draw from categorical distribution based on softmax
            @inbounds actions[i, t-1] ~ Categorical(Pₖ)

            # get selected deck: ad_val function avoids switching functions if using a different AD backend
            @inbounds k = ad_val(actions[i, t-1])

            # get payoff for selected deck
            @inbounds Xₜ = deck_payoffs[i, deck_sequence_index[i, k], k]
            # get win/loss status: loss = 1, win = 0
            @inbounds l = deck_wl[i, deck_sequence_index[i, k], k]

            # increment deck sequence index (this is because each deck has a unique payoff sequence)
            @inbounds deck_sequence_index[i, k] += 1

            # get prospect utility -> same as paper but without having to use boolean logic: uₖ = (Xₜ >= 0) ? Xₜ^Aᵢ : -wᵢ * abs(Xₜ)^Aᵢ
            Xₜᴾ = abs(Xₜ)^A′[i]
            uₖ = (l * -w′[i] * Xₜᴾ) + ((1 - l) * Xₜᴾ)

            # delta learning rule
            @inbounds Evₖ[t, :, i] .= Evₖ₍ₜ₋₁₎
            @inbounds Evₖ[t, k, i] += (1-a′[i]) * (uₖ - Evₖ₍ₜ₋₁₎[k]) + a′[i] * uₖ
            @inbounds Turing.@addlogprob! loglikelihood(Categorical(Pₖ), actions[i, t-1])
        end
    end
    return (actions, )
end



function anyMissingNaNInf(x)
    return any(ismissing, x) || any(isnan, x) || any(isinf, x)
end


##############################################
# Real Data                                  #
##############################################


# let's try with real data
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

    ############ IMPORTANT ################
    # Some experiments will be truncated, it's important to note this
    # but it should not be a problem for the model
    # this is just to test if the NaNs, missing are introduced
    # due to different lengths
    # but it does not seem so....uggghhhhh
    ####################################

    # kill all trials above 95 to avoid missing values (breaks model)
    @warn "This model is truncating trials above 95..."
    @warn "REMOVE AFTER DEBUGGING"
    trial_data = trial_data[trial_data.trial_idx .<= 95, :]
    # add a "choice pattern" column
    # 1 = CD >= 0.65, 2 = AB >= 0.65, 4 = BD >= 0.65, 8 = AC >= 0.65
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
# print trial data for choice_pattern 6 and study Wetzels
# tmp_trial_data = trial_data[trial_data.choice_pattern .== 6, :]
# tmp_trial_data = tmp_trial_data[tmp_trial_data.study .== "Wetzels", :]
# print(tmp_trial_data)
# exit()
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
        @info "Exiting..."
        exit()
    end
elseif skip_existing_chains && isfile(chain_out_file)
    @info "Chain file already exists, finding processed patterns..."
    # get patterns that have already been processed
    processed_patterns = h5open(chain_out_file, "r") do file
        pats = keys(file)
        # pats = setdiff(pats, processed_patterns)
        @info pats
        return pats
    end
end

chains::Dict{String, Chains} = Dict()
for (pat, n) in zip(pats, n_subj)
    pat_id = "pattern_$pat"
    @info "Pattern: $pat, n = $n"
    if pat_id in processed_patterns
        @info "Pattern $pat already processed, skipping..."
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
    # deck_payoffs = Array{Float64, 3}(undef, N, maximum(Tsubj), 4)
    # actually initialize with zeros
    deck_payoffs = zeros(Float64, N, maximum(Tsubj), 4)
    # deck_wl = Array{Int, 3}(undef, N, maximum(Tsubj), 4)
    deck_wl = zeros(Int, N, maximum(Tsubj), 4)
    payoff_schemes = Vector{Int}(undef, N)
    @info "Loading subject wins, losses, and payoffs..."
    for (i, subj) in enumerate(subjs)
        subj_data = trial_data_pat[trial_data_pat.subj_uid .== subj, :]
        @inbounds choice[i, 1:Tsubj[i]] = subj_data.choice
        for j in 1:4
            results_j = subj_data[subj_data.choice .== j, :outcome]
            n_results_j = length(results_j)
            @inbounds deck_payoffs[i, 1:n_results_j, j] = results_j
            @inbounds deck_wl[i, 1:n_results_j, j] = Int.(results_j .< 0)
        end
        @inbounds payoff_schemes[i] = subj_data.scheme[1]
    end



    # loop through one subject
    tmp_subj = subjs[1]
    tmp_subj_data = trial_data_pat[trial_data_pat.subj_uid .== tmp_subj, :]
    deck_sq_idx = ones(Int, 4)
    for row in eachrow(tmp_subj_data)
        deck = row.choice
        deck_idx = deck_sq_idx[deck]
        payoff = row.outcome
        wl = Int(payoff < 0)
        @info "Deck: $deck, Deck Index: $deck_idx, Payoff: $payoff, Win/Loss: $wl"
        @info "Deck Payoffs: $(deck_payoffs[1, deck_idx, :])"
        @info "Deck Wins/Losses: $(deck_wl[1, deck_idx, :])"
        deck_sq_idx[deck] += 1
        @info row
        # ensure the payoff equals the deck_paoffs value
        @assert payoff == deck_payoffs[1, deck_idx, deck]
    end

    @info "Done."
    @info "Creating model for pattern $pat..."
    model = pvl_delta(choice, N, Tsubj, deck_payoffs, deck_wl)
    

    # estimate initial parameters
    estimated_params = nothing
    if optim_param_est
        @info "Using Optim to estimate initial parameters for $pat..."
        est_start = time()
        optim_est_type == :map ? est_method = MAP() : est_method = MLE()
        optim_estimate = optimize(model, est_method, LBFGS(); autodiff = :reverse)
        est_time = time() - est_start
        @info "Estimation for $pat took $est_time seconds"
        if Optim.converged(optim_estimate.optim_result)
            @info "$optim_est_type estimate converged for $pat, using as initial parameters for sampling..."
            estimated_params = collect(Iterators.repeated(optim_estimate.values.array, n_chains))
        else
            @warn "$optim_est_type estimate did not converge for $pat"
        end
    elseif rand_param_est
        @info "Using random initial parameters for $pat..."
        # random initial parameters
        # min_pos_float = 0.0 + eps(0.0)
        min_pos_float = 1e-15 # eps is giving problems, for now set it to a very low value

        estimated_params = []
        for i in 1:n_chains
            random_params = Dict(
                "A′μ" => rand(Normal(0, 1)),
                "A′σ" => rand(Uniform(min_pos_float, 1.5)),
                "a′μ" => rand(Normal(0, 1)),
                "a′σ" => rand(Uniform(min_pos_float, 1.5)),
                "c′μ" => rand(Normal(0, 1)),
                "c′σ" => rand(Uniform(min_pos_float, 1.5)),
                "w′μ" => rand(Normal(0, 1)),
                "w′σ" => rand(Uniform(min_pos_float, 1.5)),
            )
            for j in 1:N
                random_params["A′_pr[$j]"] = rand(Normal(random_params["A′μ"], random_params["A′σ"]))
                random_params["a′_pr[$j]"] = rand(Normal(random_params["a′μ"], random_params["a′σ"]))
                random_params["c′_pr[$j]"] = rand(Normal(random_params["c′μ"], random_params["c′σ"]))
                random_params["w′_pr[$j]"] = rand(Normal(random_params["w′μ"], random_params["w′σ"]))
            end
            push!(estimated_params, random_params)
        end
    
    else
        @warn "Not estimating initial parameters for $pat..."
    end

    @info "Sampling for $pat..."
    
    chain = sample(
        model,
        sampler,
        MCMCThreads(), # disable for debugging
        n_samples,
        n_chains,
        progress=progress,
        verbose=true;
        save_state=false,
        initial_params=estimated_params,
    )
    # chain = sample(
    #     model,
    #     sampler,
    #     1_000
    # )
    chains[pat_id] = chain

    # save chain
    @info "Saving chain for $pat..."
    h5open(chain_out_file, "cw") do file
        g = create_group(file, pat_id)
        write(g, chain)
    end
    @info "Done with pattern $pat."
end
@info "Done."
