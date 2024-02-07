using DataFrames
using RData
using Random

"""
    load_trial_data(
        filename::String,
    )::DataFrame

Load the trial data from the IGT data file.
"""
function load_trial_data(
    filename::String,
)::DataFrame
    # load the data
    objs = load(filename)

    # set trial lengths (str for indices)
    trials_label = ["95", "100", "150"]

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
        subj = Int[],
        trial_idx = Int[],
        study = String[],
        choice = Int[],
        wins = Float64[],
        losses = Float64[],
        trial_length = Int[],
        outcome = Float64[],
        scheme = Int[],
    )

    # populate the trial data dictionary
    for l in trials_label
        subj = parse.(Int, objs["index_$l"][: , 1])
        study = objs["index_$l"][:, 2]
        wins = Float64.(objs["wi_$l"])
        losses = Float64.(objs["lo_$l"])
        choice_t = transpose(trunc.(Int, objs["choice_$l"]))
        for i in eachindex(subj)
            n_trials = length(losses[i, :])
            for j in 1:n_trials
                push!(df, (
                    subj[i],
                    j,
                    study[i],
                    choice_t[j, i],
                    wins[i, j],
                    losses[i, j],
                    n_trials,
                    wins[i, j] + losses[i, j],
                    scheme_map[study[i]],
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

# From description of data in:
# Steingroever, H., Fridberg, D. J., Horstmann, A., Kjome, K. L., Kumari, V., Lane, S. D., Maia, T. V., McClelland, J. L., Pachur, T., Premkumar, P., Stout, J. C., Wetzels, R., Wood, S., Worthy, D. A., & Wagenmakers, E.-J. (2015). Data from 617 Healthy Participants Performing the Iowa Gambling Task: A “Many Labs” Collaboration. Journal of Open Psychology Data, 3(1), Article 1. https://doi.org/10.5334/jopd.ak
function construct_payoff_sequence(scheme::Int)::PayoffSequence
    # Fixed Sequence block of 40 trials
    if scheme == 1
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

        return PayoffSequence(
            scheme,
            40,
            false,
            A,
            B,
            C,
            D,
        )
    # random blocks of 10 trials, with specific characteristics
    elseif scheme == 2
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

        return PayoffSequence(
            scheme,
            10,
            true,
            A,
            B,
            C,
            D,
        )
    # specific sequence of 60 with decks that can run out
    elseif scheme == 3
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

        return PayoffSequence(
            scheme,
            60,
            false,
            A,
            B,
            C,
            D,
        )
    else
        throw(ArgumentError("scheme must be 1, 2, or 3"))
    end
end


function igt_deck_payoff!(choice_history::Vector{Int}, payoffs::PayoffSequence)::Float64
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
        throw(ArgumentError("deck must be 1, 2, 3, or 4"))
    end
    # does this need to be float?
    return Float64(payoff)
end
