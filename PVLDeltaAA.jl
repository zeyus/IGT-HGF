# using CUDA
using ForwardDiff, Zygote, ReverseDiff, Distributions, FillArrays, Optim, Turing, StatsFuns
using HDF5, MCMCChains, MCMCChainsStorage
using Turing: AutoForwardDiff, ForwardDiff, AutoReverseDiff, AutoZygote


include("src/Data.jl")
include("src/LogCommon.jl")

delete_existing_chains = false
skip_existing_chains = true

# not implemented
use_gpu = false

# show progress bar
progress = true

# use Optim to estimate starting parameter values
optim_param_est = true
optim_est_type = :mle  # :mle or :map

# set AD backend
adtype = AutoReverseDiff(true)

# number of chains to run
n_chains = 1


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

# attempting a performance optimized version (need to compare)
function action_probabilities(x::AbstractVector{<:Real}, τ::Real, N::Int = 4)
    Xᵢₙ = Array{Float64}(undef, N)
    Xₒᵤₜ = Array{Float64}(undef, N)
    Xₘₐₓ = ad_val(maximum(x))
    ∑X::Float64 = 0.0
    τ₀::Float64 = ad_val(τ)
    for i in 1:N
        @inbounds Xᵢₙ[i] = exp(ad_val(x[i] - Xₘₐₓ)*τ₀)
        # ∑X += Xᵢₙ[i] # apparently julia's Base.sum is O(log(N)) vs this being O(N)
    end
    ∑X = sum(Xᵢₙ)
    for i in 1:N
        @inbounds Xₒᵤₜ[i] = Xᵢₙ[i] / ∑X
    end
    return Xₒᵤₜ
end

# original version
# function action_probabilities(x::AbstractVector{<:Real}, τ::Real)
#     xₘₐₓ = maximum(x)
#     xₙ = x .- xₘₐₓ
#     xₙ = exp.(xₙ * τ)
#     return xₙ / sum(xₙ)
# end

function logit_action_probabilities(x::AbstractVector{<:Real}, τ::Real)
    xₘₐₓ = maximum(x)
    xₙ = x .- xₘₐₓ
    xₙ = exp.(logistic.(xₙ) * τ)
    return logit.(xₙ / sum(xₙ)) # + xₘₐₓ # ???
end

@info "Defining Model..."
@model function pvl_delta(actions::Matrix{Union{Missing, Int}}, ::Type{T} = Float64; N::Int, Tsubj::Vector{Int}, deck_payoffs::Array{Float64, 3}, deck_wl::Array{Int, 3}) where {T}
    min_pos_float = eps(0.0)
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

    # individual parameters
    # Shape
    A′ ~ filldist(LogitNormal(A′μ, A′σ), N)
    # Updating parameter
    a′ ~ filldist(LogitNormal(a′μ, a′σ), N)
    # Response consistency
    c′ ~ filldist(truncated(LogNormal(c′μ, c′σ); upper=5), N)
    # Loss-Aversion
    w′ ~ filldist(truncated(LogNormal(w′μ, w′σ); upper=5), N)

    # create actions matrix if using the model for simulation
    if actions === missing
        actions = Matrix{Union{Missing, Int}}(undef, N, maximum(Tsubj))
    end

    # loop over subjects
    for i in 1:N
        # Shape
        @inbounds Aᵢ = A′[i] # Aᵢ = ϕ(A′ᵢ) -> achieved through logitnormal
        # Updating parameter
        @inbounds aᵢ = a′[i] # aᵢ = ϕ(a′ᵢ) -> achieved through logitnormal
        # Response consistency
        @inbounds cᵢ = c′[i] # cᵢ = ϕ(c′ᵢ) * 5  -> achieved through truncated LogNormal
        # Loss-Aversion
        @inbounds wᵢ = w′[i] # wᵢ = ϕ(w′ᵢ) * 5 -> achieved through truncated LogNormal

        # Set theta (exploration vs exploitation)
        θ = 3^cᵢ - 1

        # Expected value, this can be a vector, and we just update the deck.
        Evₖ = zeros(T, 4)
        
        # start each deck at the first card
        deck_sequence_index = [1, 1, 1, 1]

        # loop over trials
        @inbounds n_trials = Tsubj[i]
        for t in 2:(n_trials)
            # Get previous expected values (all decks start at 0)
            # Evₖ₍ₜ₋₁₎ = Evₖ[t-1, :]
            Evₖ₍ₜ₋₁₎ = Evₖ

            # softmax choice
            Pₖ = action_probabilities(Evₖ₍ₜ₋₁₎, θ)

            # draw from categorical distribution based on softmax
            @inbounds actions[i, t-1] ~ Categorical(Pₖ)

            # get selected deck: ad_val function avoids switching functions if using a different AD backend
            @inbounds k = ad_val(actions[i, t-1])

            # get payoff for selected deck
            @inbounds Xₜ = deck_payoffs[i, deck_sequence_index[k], k]
            # get win/loss status: loss = 1, win = 0
            @inbounds l = deck_wl[i, deck_sequence_index[k], k]

            # increment deck sequence index (this is because each deck has a unique payoff sequence)
            @inbounds deck_sequence_index[k] += 1

            # get prospect utility -> same as paper but without having to use boolean logic: uₖ = (Xₜ >= 0) ? Xₜ^Aᵢ : -wᵢ * abs(Xₜ)^Aᵢ
            Xₜᴾ = abs(Xₜ)^Aᵢ
            uₖ = (l * -wᵢ * Xₜᴾ) + ((1 - l) * Xₜᴾ)

            # update expected value of selected deck, carry forward the rest
            # Evₖ[t, :] = Evₖ₍ₜ₋₁₎

            # delta learning rule
            @inbounds Evₖ[k] += aᵢ * (uₖ - Evₖ₍ₜ₋₁₎[k])
            
            # add log likelihood for the choice
            Turing.@addlogprob! loglikelihood(Categorical(Pₖ), actions[i, t-1])
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
    deck_payoffs = Array{Float64, 3}(undef, N, maximum(Tsubj), 4)
    deck_wl = Array{Int, 3}(undef, N, maximum(Tsubj), 4)
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

    @info "Done."
    @info "Creating model for pattern $pat..."
    model = pvl_delta(choice; N=N, Tsubj=Tsubj, deck_payoffs=deck_payoffs, deck_wl=deck_wl)
    

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
    else
        @warn "Not using Optim to estimate initial parameters for $pat..."
    end

    @info "Sampling for $pat..."
    sampler = NUTS(1000, 0.40; adtype=adtype) 
    chain = sample(
        model,
        sampler,
        MCMCThreads(), # disable for debugging
        1_000,
        n_chains,
        progress=progress,
        verbose=true;
        save_state=false,
        initial_params=estimated_params,
    )
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
