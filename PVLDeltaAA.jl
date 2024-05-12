using ForwardDiff, Zygote, ReverseDiff, Distributions, FillArrays, Optim, Turing, StatsFuns
using HDF5, MCMCChains, MCMCChainsStorage
using Turing: AutoForwardDiff, ForwardDiff, AutoReverseDiff, AutoZygote

include("src/Data.jl")
include("src/LogCommon.jl")
delete_existing_chains = true
skip_existing_chains = true
progress = true
optim_param_est = true
# adtype = AutoReverseDiff(true)
# adtype = AutoZygote()
adtype = AutoForwardDiff()

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
function action_probabilities(x::AbstractVector{<:Real}, τ::Real)
    Xᵢₙ = Array{Float64}(undef, 4)
    Xₒᵤₜ = Array{Float64}(undef, 4)
    Xₘₐₓ = ad_val(maximum(x))
    ∑X::Float64 = 0.0
    τ₀::Float64 = ad_val(τ)
    @inbounds for i in 1:4
        Xᵢₙ[i] = exp(ad_val(x[i] - Xₘₐₓ)*τ₀)
        # ∑X += Xᵢₙ[i] # apparently julia's Base.sum is O(log(N)) vs this being O(N)
    end
    ∑X = sum(Xᵢₙ)
    @inbounds for i in 1:4
        Xₒᵤₜ[i] = Xᵢₙ[i] / ∑X
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
    c′ ~ filldist(truncated(LogNormal(c′μ, c′σ); upper=5), N)
    # Loss-Aversion
    w′ ~ filldist(truncated(LogNormal(w′μ, w′σ); upper=5), N)

    # create actions matrix if using the model for simulation
    if actions === missing
        actions = Matrix{Union{Missing, Int}}(undef, N, maximum(Tsubj))
    end

    # loop over subjects
    for i in 1:N
        @inbounds Aᵢ = A′[i] # Aᵢ = ϕ(A′ᵢ) -> achieved through logitnormal
        @inbounds aᵢ = a′[i] # aᵢ = ϕ(a′ᵢ) -> achieved through logitnormal
        @inbounds cᵢ = c′[i] # cᵢ = ϕ(c′ᵢ) * 5  -> achieved through truncated LogNormal
        @inbounds wᵢ = w′[i] # wᵢ = ϕ(w′ᵢ) * 5 -> achieved through truncated LogNormal
        # Set theta (exploration vs exploitation)
        θ = 3^cᵢ - 1

        # Create expected value matrix
        # Evₖ = zeros(T, Tsubj[i], 4)
        # This can be a vector, and we just update the deck.
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
            
            # add log likelihood
            @inbounds Turing.addlogprob!(logpdf(Normal(Evₖ₍ₜ₋₁₎[k], 1), Evₖ[k]))
        end
    end
    return actions
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
# @info "Removing trials above index 95..."
trial_data = trial_data[trial_data.trial_idx .<= 95, :]
# add a "choice pattern" column
# 1 = CD >= 0.65, 2 = AB >= 0.65, 4 = BD >= 0.65, 8 = AC >= 0.65
@info "Segmenting Data..."
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
        @info "Exiting..."
        exit()
    end
elseif skip_existing_chains && isfile(chain_out_file)
    @info "Chain file already exists, finding processed patterns..."
    # get patterns that have already been processed
    h5open(chain_out_file, "r") do file
        processed_patterns = keys(file)
        # pats = setdiff(pats, processed_patterns)
    end
end

chains::Dict{String, Chains} = Dict()
# priors_chain = nothing
# priors_chain_df = nothing
for (pat, n) in zip(pats, n_subj)
    @info "Pattern: $pat, n = $n"
    if "pattern_$pat" in processed_patterns
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
            
            # print("j:", j, " results_j: ", results_j, ", number of results: ", n_results_j, "\n")
            @inbounds deck_payoffs[i, 1:n_results_j, j] = results_j
            # if anyMissingNaNInf(deck_payoffs[i, :, j])
            #     @warn "Results for deck $j, $subj have missing, NaN, or Inf values for pattern $pat, skipping..."
            #     print(deck_payoffs[i, :, j])
            #     print(subj_data)
            #     print(results_j)
            #     exit()
            # end
            @inbounds deck_wl[i, 1:n_results_j, j] = Int.(results_j .< 0)
        end
        @inbounds payoff_schemes[i] = subj_data.scheme[1]
    end

    @info "Done."
    # if simulated
    # deck_payoffs = construct_payoff_matrix_of_length(N, Tsubj, payoff_schemes)
    @info "Loading model for pattern $pat..."
    # check if any values in any of the parameters we send to the model are missing, NA, NaN, etc
    # # choice
    # if anyMissingNaNInf(choice)
    #     @warn "Choice matrix has missing, NaN, or Inf values for pattern $pat, skipping..."
    #     print(choice)
    # end
    # # deck_payoffs
    # if anyMissingNaNInf(deck_payoffs)
    #     @warn "Deck payoffs matrix has missing, NaN, or Inf values for pattern $pat, skipping..."
    #     print(deck_payoffs)
    # end
    # # deck_wl
    # if anyMissingNaNInf(deck_wl)
    #     @warn "Deck win/loss matrix has missing, NaN, or Inf values for pattern $pat, skipping..."
    # end
    # # Tsubj
    # if anyMissingNaNInf(Tsubj)
    #     @warn "Tsubj has missing, NaN, or Inf values for pattern $pat, skipping..."
    # end
    # continue




    model = pvl_delta(choice; N=N, Tsubj=Tsubj, deck_payoffs=deck_payoffs, deck_wl=deck_wl)

    # generate MAP estimate
    n_chains = 3
    estimated_params = nothing
    
    if optim_param_est
        @info "Using Optim to estimate initial parameters for $pat..."
        est_start = time()
        # mle_estimate = optimize(model, MLE(), LBFGS(); autodiff = :reverse)
        mle_estimate = optimize(model, MLE(), LBFGS(); autodiff = :reverse)
        est_time = time() - est_start
        @info "Estimation for $pat took $est_time seconds"
        if Optim.converged(mle_estimate.optim_result)
            @info "MLE estimate converged for $pat, using as initial parameters for sampling..."
            # @info mle_estimate.values.array
            # @info mle_estimate
            estimated_params = [repeat([mle_estimate.values.array], n_chains)...]
        else
            @warn "MLE estimate did not converge for $pat"
        end
    else
        @warn "Not using Optim to estimate initial parameters for $pat..."
        # param_dict = Dict(
        #     "A′μ" => 0.0,
        #     "A′σ" => 1.0,
        #     "a′μ" => 0.0,
        #     "a′σ" => 1.0,
        #     "c′μ" => 0.0,
        #     "c′σ" => 1.0,
        #     "w′μ" => 0.0,
        #     "w′σ" => 1.0,
        #     # "A′" => fill(0.0, N),
        #     # "a′" => fill(0.0, N),
        #     # "c′" => fill(0.0, N),
        #     # "w′" => fill(0.0, N)
        # )
        # # we need to add params for each subject + 1 (e.g. A′[1], ..., A′[N + 1])
        # for i in 1:N
        #     param_dict["A′[$i]"] = 0.5
        #     param_dict["a′[$i]"] = 0.5
        #     param_dict["c′[$i]"] = 0.5
        #     param_dict["w′[$i]"] = 0.5
        # end
        # estimated_params =repeat([
        #     param_dict
        # ], n_chains)
    end

    # get one from priors out
    # priors_chain = sample(model, Prior(), 1_000, progress=progress, verbose=false)
    # println(priors_chain)
    # priors_chain_df = DataFrame(priors_chain)
    # print(summary(priors_chain_df))
    # break
    # defaults: -1: uses max(n_samples/2, 500) ?!?, 0.65: target acceptance rate; Adtype is set to ForwardDiff by default
    # sampler = NUTS()
    # sampler = NUTS(500, 0.65; max_depth=10, adtype=AutoForwardDiff(; chunksize=0))
    @info "Sampling for $pat..."
    sampler = NUTS(; adtype=adtype) 
    # start sampling but ignore warnings (for now...so many warnings about step size NaN, needs addressing)
    #with_logger(surpress_warnings) do
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
    #end
    chains["pattern_$pat"] = chain

    # save chain
    @info "Saving chain for $pat..."
    pattern_name = "pattern_$pat"
    h5open(chain_out_file, "cw") do file
        g = create_group(file, pattern_name)
        write(g, chain)
    end
    @info "Done with pattern $pat."
end

@info "Done."
