using DataFrames
using RData
using Random
# using ForwardDiff: Dual, value
# Base.Integer(x::Dual) = Integer(value(x))


function load_trial_data(
    filename::String,
    add_missing_input::Bool = false
)::DataFrame
    # load the data
    objs = load(filename)

    # set trial lengths (str for indices)
    trials_label = ["95", "100", "150"]
    subj_uid = 0
    # Steingroever, H., Fridberg, D. J., Horstmann, A., Kjome, K. L., Kumari, V., Lane, S. D., Maia, T. V., McClelland, J. L., Pachur, T., Premkumar, P., Stout, J. C., Wetzels, R., Wood, S., Worthy, D. A., & Wagenmakers, E.-J. (2015). Data from 617 Healthy Participants Performing the Iowa Gambling Task: A “Many Labs” Collaboration. Journal of Open Psychology Data, 3(1), Article 1. https://doi.org/10.5334/jopd.ak
    scheme_map = Dict(
        "Fridberg" => 1,
        "Horstmann" => 2,
        "Kjome" => 3,
        "Maia" => 1,
        "Premkumar" => 3,
        "Steingroever2011" => 2,
        "SteingroverInPrep" => 2,
        "Wetzels" => 2,
        "Wood" => 3,
        "Worthy" => 1,
    )

    # create dataframe
    df = DataFrame(
        subj = Union{Missing, Int}[],
        trial_idx = Union{Missing, Int}[],
        study = Union{Missing, String}[],
        choice = Union{Missing, Int}[],
        wins = Union{Missing, Float64}[],
        losses = Union{Missing, Float64}[],
        trial_length = Union{Missing, Int}[],
        outcome = Union{Missing, Float64}[],
        scheme = Union{Missing, Int}[],
        next_choice = Union{Missing, Int}[],
        cd_ratio = Union{Missing, Float64}[],
        ab_ratio = Union{Missing, Float64}[],
        bd_ratio = Union{Missing, Float64}[],
        ac_ratio = Union{Missing, Float64}[],
        bc_ratio = Union{Missing, Float64}[],
        ad_ratio = Union{Missing, Float64}[],
        subj_study = Union{Missing, Int}[],
    )

    # populate the trial data dictionary
    for l in trials_label
        subj = parse.(Int, objs["index_$l"][: , 1])
        study = objs["index_$l"][:, 2]
        wins = Float64.(objs["wi_$l"])
        losses = Float64.(objs["lo_$l"])
        choice_t = transpose(trunc.(Int, objs["choice_$l"]))
        for i in eachindex(subj)
            subj_uid += 1
            n_trials = length(losses[i, :])
            CDRatio = (sum(choice_t[:, i] .== 3) + sum(choice_t[:, i] .== 4)) / n_trials
            ABRatio = (sum(choice_t[:, i] .== 1) + sum(choice_t[:, i] .== 2)) / n_trials
            BDRatio = (sum(choice_t[:, i] .== 2) + sum(choice_t[:, i] .== 4)) / n_trials
            ACRatio = (sum(choice_t[:, i] .== 1) + sum(choice_t[:, i] .== 3)) / n_trials
            BCRatio = (sum(choice_t[:, i] .== 2) + sum(choice_t[:, i] .== 3)) / n_trials
            ADRatio = (sum(choice_t[:, i] .== 1) + sum(choice_t[:, i] .== 4)) / n_trials
            
            if add_missing_input
                push!(df, (
                    subj_uid,
                    0,
                    study[i],
                    missing,
                    missing,
                    missing,
                    n_trials,
                    missing,
                    scheme_map[study[i]],
                    choice_t[1, i],
                    CDRatio,
                    ABRatio,
                    BDRatio,
                    ACRatio,
                    BCRatio,
                    ADRatio,
                    subj[i],
                ))
            end
            for j in 1:n_trials
                next_choice = j < n_trials ? choice_t[j + 1, i] : missing
                push!(df, (
                    subj_uid,
                    j,
                    study[i],
                    choice_t[j, i],
                    wins[i, j],
                    losses[i, j],
                    n_trials,
                    wins[i, j] + losses[i, j],
                    scheme_map[study[i]],
                    next_choice,
                    CDRatio,
                    ABRatio,
                    BDRatio,
                    ACRatio,
                    BCRatio,
                    ADRatio,
                    subj[i],
                ))
            end
        end
    end

    return df
end

mutable struct PayoffSequence
    scheme::Int
    sequence_length::Int
    shuffle_on_repeat::Bool
    A::Vector{Int}
    B::Vector{Int}
    C::Vector{Int}
    D::Vector{Int}
end

function payoff_scheme_1()::Array{Int, 3}
    A = fill(100, 40)
    # set -150s
    A[[3,18,28,37]] .+= -150
    # set -200s
    A[[7,15,26,32]] .+= -200
    # set -250s
    A[[9,14,27,33]] .+= -250
    # set -300s
    A[[5,22,17,38]] .+= -300
    # set -350s
    A[[10,12,24,31]] .+= -350

    B = fill(100, 40)
    # set -1250s
    B[[9,14,21,32]] .+= -1250

    C = fill(50, 40)
    # set -25s
    C[[12,17,25,34,35]] .+= -25
    # set -50s
    C[[3,5,7,9,10,20,24,26,30,39]] .+= -50
    # set -75s
    C[[13,18,29,37,40]] .+= -75

    D = fill(50, 40)
    # set -250s
    D[[10,20,29,35]] .+= -250

    # create losses, if A is negative, Aₗ = 1, else Aₗ = 0
    Aₗ = zeros(Int, 40)
    Bₗ = zeros(Int, 40)
    Cₗ = zeros(Int, 40)
    Dₗ = zeros(Int, 40)
    Aₗ = Int.(A .< 0)
    Bₗ = Int.(B .< 0)
    Cₗ = Int.(C .< 0)
    Dₗ = Int.(D .< 0)

    return [[A;; B;; C;; D];;; [Aₗ;; Bₗ;; Cₗ;; Dₗ]]
end



function payoff_scheme_2()::Array{Int, 3}
    A = fill(100, 10)
    A[[1,2,3,4,5]] += [-150, -200, -250, -300, -350]
    shuffle!(A)
    B = fill(100, 10)
    B[1] += -1250
    shuffle!(B)
    C = fill(50, 10)
    C[[1,2,3,4,5]] .+= -50
    shuffle!(C)
    D = fill(50, 10)
    D[1] += -250
    shuffle!(D)

    # create losses, if A is negative, Aₗ = 1, else Aₗ = 0
    Aₗ = zeros(Int, 10)
    Bₗ = zeros(Int, 10)
    Cₗ = zeros(Int, 10)
    Dₗ = zeros(Int, 10)
    Aₗ = Int.(A .< 0)
    Bₗ = Int.(B .< 0)
    Cₗ = Int.(C .< 0)
    Dₗ = Int.(D .< 0)

    return [[A;; B;; C;; D];;; [Aₗ;; Bₗ;; Cₗ;; Dₗ]]
end

function payoff_scheme_3()::Array{Int, 3}
    A = [
        100,120,-70,90,-190,100,-120,120,-140,-260,110,
        -220,90,-150,-80,110,-210,-20,-130,100,-130,-160,
        110,-240,100,-80,-120,-40,-110,120,-220,-80,-110,
        -120,-40,150,-10,-180,-200,110,-210,-70,-100,-110,
        -30,160,0,-170,-190,-130,-200,-60,-90,-100,-20,-80,
        10,-160,-180,-120,
    ]
    B = [
        100,80,110,120,90,100,90,120,-1140,80,110,100,90,
        -1370,120,130,110,90,100,120,-1630,110,140,130,
        100,110,120,120,140,110,130,-1860,120,110,130,
        150,110,150,120,140,140,150,130,120,140,
        -2090,120,160,130,150,150,160,140,130,150,
        170,130,-2330,140,160,
    ]
    C = [
        50,60,-10,55,5,45,0,45,10,-10,55,30,-10,45,45,
        40,25,-15,70,-10,60,40,55,30,15,10,55,40,-35,
        30,40,75,30,35,45,65,-20,50,-5,10,45,80,35,40,
        50,45,-15,55,0,15,50,60,40,45,55,50,-10,60,5,20
    ]
    D = [
        50,40,45,45,55,60,40,55,50,-190,55,40,60,40,45,
        55,65,70,50,-205,60,55,65,80,40,80,40,65,-245,
        60,65,75,60,65,-250,85,45,55,70,55,70,80,65,70,
        -270,90,50,60,75,60,75,85,70,75,85,95,55,-310,80,65
    ]

    # create losses, if A is negative, Aₗ = 1, else Aₗ = 0
    Aₗ = zeros(Int, 60)
    Bₗ = zeros(Int, 60)
    Cₗ = zeros(Int, 60)
    Dₗ = zeros(Int, 60)
    Aₗ = Int.(A .< 0)
    Bₗ = Int.(B .< 0)
    Cₗ = Int.(C .< 0)
    Dₗ = Int.(D .< 0)

    return [[A;; B;; C;; D];;; [Aₗ;; Bₗ;; Cₗ;; Dₗ]]
end

# From description of data in:
# Steingroever, H., Fridberg, D. J., Horstmann, A., Kjome, K. L., Kumari, V., Lane, S. D., Maia, T. V., McClelland, J. L., Pachur, T., Premkumar, P., Stout, J. C., Wetzels, R., Wood, S., Worthy, D. A., & Wagenmakers, E.-J. (2015). Data from 617 Healthy Participants Performing the Iowa Gambling Task: A “Many Labs” Collaboration. Journal of Open Psychology Data, 3(1), Article 1. https://doi.org/10.5334/jopd.ak
function construct_payoff_sequence(scheme::Int)::PayoffSequence
    # Fixed Sequence block of 40 trials
    if scheme == 1
        decks = payoff_scheme_1()

        return PayoffSequence(
            scheme,
            40,
            false,
            decks[:, 1, 1],
            decks[:, 2, 1],
            decks[:, 3, 1],
            decks[:, 4, 1],
        )
    # random blocks of 10 trials, with specific characteristics
    elseif scheme == 2
        decks = payoff_scheme_2()

        return PayoffSequence(
            scheme,
            10,
            true,
            decks[:, 1, 1],
            decks[:, 2, 1],
            decks[:, 3, 1],
            decks[:, 4, 1],
        )
    # specific sequence of 60 with decks that can run out
    elseif scheme == 3
        decks = payoff_scheme_3()

        return PayoffSequence(
            scheme,
            60,
            false,
            decks[:, 1, 1],
            decks[:, 2, 1],
            decks[:, 3, 1],
            decks[:, 4, 1],
        )
    else
        throw(ArgumentError("scheme must be 1, 2, or 3, not $scheme"))
    end
end

# Create a matrix of prepared deck sequences
# n_subj: number of subjects
# n_trial: number of trials per subject
# schemes: vector of scheme per subject
# returns: 4D array of payoff sequences (subject, trial, deck, payoff or loss)
function construct_payoff_matrix_of_length(n_subj::Int, n_trial::Vector{Int}, schemes::Vector{Int})::Array{Int, 4}
    # create a 4D array of payoff sequences
    payoffs = Array{Int, 4}(undef, n_subj, maximum(n_trial), 4, 2)
    # scheme 1 and 3 don't change so we can just create them once
    s1 = payoff_scheme_1()
    s1_len = size(s1, 1)
    s3 = payoff_scheme_3()
    s3_len = size(s3, 1)
    for i in 1:n_subj
        n_trials = n_trial[i]
        if schemes[i] != 2
            s_cur = schemes[i] == 1 ? s1 : s3
            s_cur_len = schemes[i] == 1 ? s1_len : s3_len
            while n_trials > s_cur_len
                s_cur = vcat(s_cur, s_cur)
                s_cur_len = size(s_cur, 1)
            end
            # trim to the correct length
            if n_trials < s_cur_len
                s_cur = s_cur[1:n_trials, :, :]
            end
            payoffs[i, 1:n_trials, :, :] = s_cur
        else
            # payoff scheme 2 is randomized for each sequence length
            s_cur = Array{Int, 3}(undef, 0, 4, 2)
            s_cur_len = 0
            while n_trials > s_cur_len
                s_cur = vcat(s_cur, payoff_scheme_2())
                s_cur_len = size(s_cur, 1)
            end
            # trim to the correct length
            if n_trials < s_cur_len
                s_cur = s_cur[1:n_trials, :, :]
            end
            payoffs[i, 1:n_trials, :, :] = s_cur
        end
    end
    return payoffs
end


function igt_deck_payoff!(choice_history::Vector{Union{Missing, T}}, payoffs::PayoffSequence, ::Type{T} = Int16)::Float64 where {T<:Integer}
    # choice_history = Integer.(value(choice_history))
    # get last selection
    deck = last(choice_history)
    # get the number of times this deck has been selected previously
    deck_use_count = count(x -> x == deck, choice_history[1:end-1])
    # get the payoff for this deck
    deck_index = deck_use_count + 1
    # either way we need to keep incrementing but wrap around
    m = deck_index % payoffs.sequence_length
    deck_index = m == 0 ? payoffs.sequence_length : m
    if deck_index > payoffs.sequence_length
        # some decks repeat, some need reshuffling
        if payoffs.shuffle_on_repeat
            new_payoffs = construct_payoff_sequence(payoffs.scheme)
            payoffs.A = new_payoffs.A
            payoffs.B = new_payoffs.B
            payoffs.C = new_payoffs.C
            payoffs.D = new_payoffs.D
        end
    end
    # get appropriate payoff
    if deck == 1
        payoff = payoffs.A[deck_index]
    elseif deck == 2
        payoff = payoffs.B[deck_index]
    elseif deck == 3
        payoff = payoffs.C[deck_index]
    elseif deck == 4
        payoff = payoffs.D[deck_index]
    else
        throw(ArgumentError("deck must be 1, 2, 3, or 4 not $deck"))
    end
    # does this need to be float?
    return Float64(payoff)
end
