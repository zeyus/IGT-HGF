using StanSample
using MonteCarloMeasurements
using Feather, HDF5, MCMCChains, MCMCChainsStorage, DataFrames
using Random
using Distributions


include("src/Data.jl")
include("src/LogCommon.jl")


# CONFIG
delete_existing_chains = false
skip_existing_chains = true

n_chains = 3
n_samples = 20_000
n_warmup = 10_000
adapt_delta = 0.5
thinning = 1
max_depth=20




PVLDeltaSteingroeverModel = "
data {
    int<lower=1> n_s;                           // # subjects
    int<lower=1> n_t_max;                       // max # trials per subject
    array[n_s] int<lower=1> n_t;                      // # trials per subject
    array[n_s, n_t_max] int<lower=1,upper=4> choice;  // # subj. x # trials matrix with choices
    array[n_s, n_t_max] int<lower=0,upper=1> trial_wl; // # subj. x # trials matrix with wins/losses
    matrix<lower=-2000,upper=2000>[n_s, n_t_max] net;  // Net amount of wins + losses   
}

parameters {
    // Group-level mean parameters
    real mu_A_pr;   
    real mu_w_pr;
    real mu_a_pr;  
    real mu_c_pr;

    // Group-level standard deviation  
    real<lower=0> sd_A;
    real<lower=0> sd_w;
    real<lower=0> sd_a;
    real<lower=0> sd_c;    

    // Individual-level parameters
    vector[n_s] A_ind_pr; 
    vector[n_s] w_ind_pr; 
    vector[n_s] a_ind_pr;   
    vector[n_s] c_ind_pr; 
}

transformed parameters {
    matrix[n_s, n_t_max] adjusted_net;

    real<lower=0,upper=1> mu_A;   
    real<lower=0,upper=5> mu_w;
    real<lower=0,upper=1> mu_a;  
    real<lower=0,upper=5> mu_c;

    // Individual-level parameters
    vector<lower=0,upper=1>[n_s] A_ind; 
    vector<lower=0,upper=5>[n_s] w_ind; 
    vector<lower=0,upper=1>[n_s] a_ind;   
    vector<lower=0,upper=5>[n_s] c_ind; 


    mu_A = Phi(mu_A_pr);
    mu_w = Phi(mu_w_pr) * 5;
    mu_a = Phi(mu_a_pr);
    mu_c = Phi(mu_c_pr) * 5;

    A_ind = Phi(A_ind_pr);
    w_ind = Phi(w_ind_pr) * 5;
    a_ind = Phi(a_ind_pr);
    c_ind = Phi(c_ind_pr) * 5;

    for (s in 1:n_s) {
       adjusted_net[s] = abs(net[s]) .^ A_ind[s];
    }
    
}

model {
    vector[4] p;
    vector[4] Ev;
    real theta;
    real v;

    // Prior on the group-level mean parameters
    mu_A_pr ~ normal(0, 1);
    mu_w_pr ~ normal(0, 1);
    mu_a_pr ~ normal(0, 1);
    mu_c_pr ~ normal(0, 1);

    // Prior on the group-level standard deviation
    sd_A ~ uniform(0, 1.5);
    sd_w ~ uniform(0, 1.5);
    sd_a ~ uniform(0, 1.5);
    sd_c ~ uniform(0, 1.5);

    // Individual-level parameters
    A_ind_pr ~ normal(mu_A_pr, sd_A);
    w_ind_pr ~ normal(mu_w_pr, sd_w);
    a_ind_pr ~ normal(mu_a_pr, sd_a);
    c_ind_pr ~ normal(mu_c_pr, sd_c);


    for (s in 1:n_s) {  // loop over subjects
        theta = pow(3, c_ind[s]) - 1;

        // Trial 1
        for (d in 1:4) {
            p[d] = 0.25;
            Ev[d] = 0;
        }
        choice[s,1] ~ categorical(p);
        // manually specify loglikelihood (maybe the problem is the matrix containing choices outside??)
        // target += categorical_lpmf(choice[s, 1] | p);
        print(\"target = \", target());
        // Remaining trials
        for (t in 1:(n_t[s] - 1)) {
            // if (net[s,t] >= 0) {
            //     v = pow(net[s,t], A_ind[s]);
            // } else {
            //     v = -1 * w_ind[s] * pow(abs(net[s,t]), A_ind[s]);
            // }
           
            v = (trial_wl[s, t] * -w_ind[s] * adjusted_net[s, t]) + ((1 - trial_wl[s, t]) * adjusted_net[s, t]);
            Ev[choice[s, t]] = (1 - a_ind[s]) * Ev[choice[s, t]] + a_ind[s] * v;
            
            p = softmax(Ev * theta);
            choice[s, t + 1] ~ categorical(p);
            // manually specify loglikelihood
            // target += categorical_lpmf(choice[s, t + 1] | p);
            // print loglikelihood
            // print(\"Loglikelihood: \", categorical_lpmf(choice[s, t + 1] | p), \"\\n\");
            // print a predicted draw vs actual
            //print(\"Predicted probs: \", p, \" Actual: \", choice[s, t + 1], \"\\n\");
            // print(\"target = \", );
        }
    }
}
"

function init(n_s::Int)
    init_params = Dict{String, Any}(
        "mu_A_pr" => rand(Normal(0,1)),
        "mu_w_pr" => rand(Normal(0,1)),
        "mu_a_pr" => rand(Normal(0,1)),
        "mu_c_pr" => rand(Normal(0,1)),
        "sd_A" => rand(Exponential(1.0)),
        "sd_w" => rand(Exponential(1.0)),
        "sd_a" => rand(Exponential(1.0)),
        "sd_c" => rand(Exponential(1.0)),
    )
    init_params["A_ind_pr"] = rand(Normal(init_params["mu_A_pr"], init_params["sd_A"]), n_s)
    init_params["w_ind_pr"] = rand(Normal(init_params["mu_w_pr"], init_params["sd_w"]), n_s)
    init_params["a_ind_pr"] = rand(Normal(init_params["mu_a_pr"], init_params["sd_a"]), n_s)
    init_params["c_ind_pr"] = rand(Normal(init_params["mu_c_pr"], init_params["sd_c"]), n_s)
    return init_params
end


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
    # @warn "This model is truncating trials above 95..."
    # @warn "REMOVE AFTER DEBUGGING"
    # trial_data = trial_data[trial_data.trial_idx .<= 95, :]
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

chain_out_file = "./data/igt_pvldelta_data_chains_STAN.h5"

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
    choice = ones(Int, N, maximum(Tsubj)) # init with ones (to avoid missing values for stan)

    # this array is overlarge but not sure how to reduce it
    # sparse array maybe?
    # deck_payoffs = Array{Float64, 3}(undef, N, maximum(Tsubj), 4)
    # actually initialize with zeros
    deck_payoffs = zeros(Float64, N, maximum(Tsubj))
    # deck_wl = Array{Int, 3}(undef, N, maximum(Tsubj), 4)
    deck_wl = zeros(Int, N, maximum(Tsubj))
    @info "Loading subject wins, losses, and payoffs..."
    for (i, subj) in enumerate(subjs)
        subj_data = trial_data_pat[trial_data_pat.subj_uid .== subj, :]
        @inbounds choice[i, 1:Tsubj[i]] = subj_data.choice
        @inbounds deck_payoffs[i, 1:Tsubj[i]] = subj_data.outcome
        @inbounds deck_wl[i, 1:Tsubj[i]] = Int.(subj_data.outcome .< 0)
    end
    observed_data = Dict(
        "n_s" => N,
        "n_t_max" => maximum(Tsubj),
        "n_t" => Tsubj,
        "choice" => choice,
        "trial_wl" => deck_wl,
        "net" => deck_payoffs
    )
    @info "Done."

    @info "Creating model for pattern $pat..."

    model = SampleModel("PVLDelta", PVLDeltaSteingroeverModel, pwd() * "\\.tmp")

    @info "Sampling for $pat..."
    rc = stan_sample(
        model,
        data=observed_data,
        init=init(N),
        use_cpp_chains=false,
        num_chains=n_chains,
        num_threads=4,
        num_samples=n_samples,
        num_warmups=n_warmup,
        delta=adapt_delta,
        engine=:nuts,
        algorithm=:hmc,
        thin=thinning,
        max_depth=max_depth,
        sig_figs=18,
        kappa=0.85,
        gamma=0.1,
        show_logging=true,
        diagnostics=true
    )
    
    if success(rc)
        @info "Sampling successful."
    else
        @error "Sampling failed."
        continue
    end
    chain = read_samples(model, :mcmcchains)


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
