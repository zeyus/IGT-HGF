using Distributed

n_procs = 1

if n_procs > 1
    addprocs(n_procs)
end

@everywhere begin
    using Base.Threads: @threads
    using Base: RefValue
    using StatsFuns: normpdf
    using LogExpFunctions, Tapir, LinearAlgebra, ForwardDiff, Enzyme, Zygote, ReverseDiff, DistributionsAD, FillArrays, Optim, Turing #, StatsFuns
    using Feather, HDF5, MCMCChains, MCMCChainsStorage, DataFrames
    using Turing: AutoForwardDiff, ForwardDiff, AutoReverseDiff, AutoZygote, AutoTapir, AbstractMCMC.AbstractMCMCEnsemble
    using ADTypes: AutoEnzyme



    include("src/Data.jl")
    include("src/LogCommon.jl")

    delete_existing_chains = false
    skip_existing_chains = true

    # show progress bar
    progress = true
    # verbose sampling output
    verbose = true

    # parallel or serial
    parallel = true

    # use threads (true) or processes (false) (only if parallel is true)
    threads = true

    # use Optim to estimate starting parameter values
    optim_param_est = false

    # type of estimation to use maximum likelihood estimation (:mle) or maximum a posteriori estimation (:map)
    optim_est_type = :mle  # :mle or :map

    # if not using Optim, should we generate random starting parameters?
    rand_param_est = true
    # number of chains to run
    n_chains = 3
    n_samples = 3_000
    thinning = 1 # thinning factor, not needed these days, in the paper they used 5
    
    global chain_handler::AbstractMCMCEnsemble
    if parallel
        if threads
            @info "Running chains in parallel with threads..."
            chain_handler = MCMCThreads()
        else
            @info "Running chains in parallel with seperate processes..."
            
            chain_handler = MCMCDistributed()
        end
    else
        @info "Running chains serially..."
        chain_handler = MCMCSerial()
    end
    # set AD backend
    # @info "Using Enzyme as AD backend..."
    # Enzyme.API.runtimeActivity!(true)
    # adtype = AutoEnzyme()
    # @info "Using Tapir as AD backend..."
    # adtype = AutoTapir(; safe_mode=false)
    # @info "Using ReverseDiff as AD backend..."
    # adtype = AutoReverseDiff(; compile=true)
    @info "Using Zygote as AD backend..."
    adtype = AutoZygote()
    # @info "Using ForwardDiff as AD backend..."
    # adtype = AutoForwardDiff()
    # number of warmup samples
    n_warmup = 1_500
    # target acceptance rate
    target_accept = 0.45
    # max tree depth
    max_depth = 20
    # initial step size
    init_ϵ = 0.0 # if 0.0, turing will try to estimate it
    # max divergence during doubling
    # Δ_max = 
    # @info "Using NUTS Sampler with $n_warmup warmup samples, $target_accept target acceptance rate, $max_depth max tree depth, and $init_ϵ initial step size."
    sampler = NUTS(
        n_warmup, # warmup samples
        target_accept; # target acceptance rate
        max_depth=max_depth, # max tree depth
        init_ϵ=init_ϵ, # initial step size
        adtype=adtype) 
    # # @info "Using HMCDA Sampler..."
    # sampler = HMCDA(
    #     n_warmup, # n adapt steps
    #     target_accept, # target acceptance rate
    #     0.3; # target leapfrog length
    #     adtype=adtype
    # )
    # @info "Using HMC Sampler..."
    # sampler = HMC(
    #     0.45, # step size
    #     20; # number of steps
    #     adtype=adtype
    # )
    # @info "Using Particle Gibbs Sampler...(NOT WORKING)"
    # sampler = PG(
    #     10
    # )
    # @info "Using MH sampler..."
    # sampler = MH()
    # @info "Using Gibbs sampler..."
    # sampler = Gibbs(
    #     HMC(0.05, 10,
    #     :Aμ,
    #     :Aσ,
    #     :aμ,
    #     :aσ,
    #     :cμ,
    #     :cσ,
    #     :wμ,
    #     :wσ,
    #     :A_pr,
    #     :a_pr,
    #     :c_pr,
    #     :w_pr
    #     ; adtype=adtype
    #     ),
    #     PG(10,
    #     :actions
    #     )
    # )


end
@info "Sampling will use $n_chains chains with $n_samples samples each."


Turing.setprogress!(progress)

# ad_val(x::T) where {T <: Real} = x
# ad_val(x::ForwardDiff.Dual{T, V, N}) where {T, V, N} = ForwardDiff.value(x)
# ad_val(x::ReverseDiff.TrackedReal{V, D, O}) where {V, D, O} = ReverseDiff.value(x)


@info "Defining Model..."
@everywhere @model function pvl_delta(actions::Matrix{Int}, N::Int, Tsubj::Vector{Int}, deck_payoffs::Array{Float64, 2}, deck_wl::Array{Int, 2}, ::Type{T} = Float64) where {T <: Real}
    min_pos_float = T(0.0 + eps(0.0))
    zero_t = T(0.0)
    one_t = T(1.0)
    onefive_t = T(1.5)
    # min_pos_float = 1e-15 # eps is giving problems, for now set it to a very low value
    # Group Level Parameters
    # Group level Shape mean
    Aμ ~ Normal(zero_t, one_t; check_args=false)
    # Group level Shape standard deviation
    Aσ ~ Uniform(min_pos_float, onefive_t; check_args=false)

    # Group level updating parameter mean
    aμ ~ Normal(zero_t, one_t; check_args=false)
    # Group level updating parameter standard deviation
    aσ ~ Uniform(min_pos_float, onefive_t; check_args=false)

    # Group level response consistency mean
    cμ ~ Normal(zero_t, one_t; check_args=false)
    # Group level response consistency standard deviation
    cσ ~ Uniform(min_pos_float, onefive_t; check_args=false)

    # Group Level Loss-Aversion mean
    wμ ~ Normal(zero_t, one_t; check_args=false)
    # Group Level Loss-Aversion standard deviation
    wσ ~ Uniform(min_pos_float, onefive_t; check_args=false)

    # Shape
    A_pr ~ filldist(Normal(Aμ, Aσ; check_args=false), N)
    # Updating parameter
    a_pr ~ filldist(Normal(aμ, aσ; check_args=false), N)
    # Response consistency
    c_pr ~ filldist(Normal(cμ, cσ; check_args=false), N)
    # Loss-Aversion
    w_pr ~ filldist(Normal(wμ, wσ; check_args=false), N)

    A = normpdf.(0, 1, A_pr)
    a = normpdf.(0, 1, a_pr)
    θ = normpdf.(0, 1, c_pr) .* 5
    w = normpdf.(0, 1, w_pr) .* 5
    

    # create actions matrix if using the model for simulation
    # if actions === missing
    #     actions = Matrix{Union{Missing, Int}}(undef, N, max_trials)
    # end

    # vectorize calculation of prospect utility
    uₖ = (deck_wl .* -w .* abs.(deck_payoffs) .^ A) .+ ((1 .- deck_wl) .* abs.(deck_payoffs) .^ A)

    for s in 1:N
        # set expected values to 0
        Evₖ = zeros(T, 4)
        # set initial probabilities all to 0.25
        Pₖ = fill(T(0.25), 4)
        # set initial action
        @inbounds actions[s, 1] ~ Categorical(Pₖ; check_args=false)
        
        # loop over trials
        @inbounds n_trials = Tsubj[s] - 1
        @inbounds a_s = a[s]
        @inbounds θ_s = θ[s]
        for t in 1:n_trials
            # delta learning rule
            @inbounds Evₖ[actions[s, t]] = (1 - a_s) * Evₖ[actions[s, t]] + a_s * uₖ[s, t]

            # update probabilities
            Pₖ = softmax(Evₖ .* θ_s)

            # if any of Pk is > 1 or < 0, reject the sample


            @inbounds actions[s, t+1] ~ Categorical(Pₖ; check_args=false)            
        end
    end
    # debug print out all params and actions
    # @info "A: ", A
    # @info "a: ", a
    # @info "c: ", c
    # @info "w: ", w
    # @info "A_pr: ", A_pr
    # @info "a_pr: ", a_pr
    # @info "c_pr: ", c_pr
    # @info "w_pr: ", w_pr
    # @info "actions: ", actions

    return (actions, )
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

chain_out_file = "./data/igt_pvldelta_data_chains_normalized.h5"

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

# print info about number of patterns/groups and number of subjects in each
@info "Patterns and Subjects"
for (pat, n) in zip(pats, n_subj)
    @info("Pattern: $pat, Subjects: $n")
    # standardize outcome between -1 and 1 (keeping 0 as zero) by pattern
    @info "Normalizing outcome between -1 and 1..."
    max_abs_outcome = maximum(abs.(trial_data[trial_data.choice_pattern .== pat, :outcome]))
    trial_data[trial_data.choice_pattern .== pat, :outcome] = trial_data[trial_data.choice_pattern .== pat, :outcome] ./ max_abs_outcome
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
    choice = Matrix{Int}(undef, N, maximum(Tsubj))

    # this array is overlarge but not sure how to reduce it
    # sparse array maybe?
    # deck_payoffs = Array{Float64, 3}(undef, N, maximum(Tsubj), 4)
    # actually initialize with zeros
    deck_payoffs = zeros(Float64, N, maximum(Tsubj))
    # deck_wl = Array{Int, 3}(undef, N, maximum(Tsubj), 4)
    deck_wl = zeros(Int, N, maximum(Tsubj))
    payoff_schemes = Vector{Int}(undef, N)
    @info "Loading subject wins, losses, and payoffs..."
    for (i, subj) in enumerate(subjs)
        subj_data = trial_data_pat[trial_data_pat.subj_uid .== subj, :]
        @inbounds choice[i, 1:Tsubj[i]] = subj_data.choice
        @inbounds deck_payoffs[i, 1:Tsubj[i]] = subj_data.outcome
        @inbounds deck_wl[i, 1:Tsubj[i]] = Int.(subj_data.outcome .< 0)
        @inbounds payoff_schemes[i] = subj_data.scheme[1]
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
            @info "Aμ: ", estimated_params[1][1], ", Aσ: ", estimated_params[2][2]
            @info "aμ: ", estimated_params[1][3], ", aσ: ", estimated_params[2][4]
            @info "cμ: ", estimated_params[1][5], ", cσ: ", estimated_params[2][6]
            @info "wμ: ", estimated_params[1][7], ", wσ: ", estimated_params[2][8]
        else
            @warn "$optim_est_type estimate did not converge for $pat"
        end
    elseif rand_param_est
        @info "Using random initial parameters for $pat..."
        # random initial parameters
        min_pos_float = 0.0 + eps(0.0)
        # min_pos_float = 1e-15 # eps is giving problems, for now set it to a very low value

        estimated_params = []
        for i in 1:n_chains

            random_params::Vector{Float64} = [
                rand(Normal(0, 0.2)),
                rand(Uniform(min_pos_float, 0.2)),
                rand(Normal(0, 0.2)),
                rand(Uniform(min_pos_float, 0.2)),
                rand(Normal(0, 0.2)),
                rand(Uniform(min_pos_float, 0.2)),
                rand(Normal(0, 0.2)),
                rand(Uniform(min_pos_float, 0.2)),
            ]
            random_params_1 = Vector{Float64}(undef, N)
            random_params_2 = Vector{Float64}(undef, N)
            random_params_3 = Vector{Float64}(undef, N)
            random_params_4 = Vector{Float64}(undef, N)
            for j in 1:N
                random_params_1[j] = rand(Normal(random_params[1], random_params[2]))
                random_params_2[j] = rand(Normal(random_params[3], random_params[4]))
                random_params_3[j] = rand(Normal(random_params[5], random_params[6]))
                random_params_4[j] = rand(Normal(random_params[7], random_params[8]))
            end
            random_params = [random_params; random_params_1; random_params_2; random_params_3; random_params_4]
            push!(estimated_params, random_params)
        end
        @info "Starting parameters:"
        for i in 1:n_chains
            @info "Chain $i:"
            @info "Aμ: ", estimated_params[i][1], ", Aσ: ", estimated_params[i][2]
            @info "aμ: ", estimated_params[i][3], ", aσ: ", estimated_params[i][4]
            @info "cμ: ", estimated_params[i][5], ", cσ: ", estimated_params[i][6]
            @info "wμ: ", estimated_params[i][7], ", wσ: ", estimated_params[i][8]
            @info "Total number of parameters: ", length(estimated_params[i])
        end
    else
        @warn "Not estimating initial parameters for $pat..."
    end

    @info "Sampling for $pat..."
    
    chain = sample(
        model,
        sampler,
        chain_handler,
        n_samples,
        n_chains,
        progress=progress,
        verbose=verbose;
        save_state=true,
        initial_params=estimated_params,
        thinning=thinning
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
