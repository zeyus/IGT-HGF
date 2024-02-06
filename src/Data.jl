using DataFrames
using RData

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

    # create a trial data dictionary
    trial_data = Dict()

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
    )

    # populate the trial data dictionary
    for l in trials_label
        subj = parse.(Int, objs["index_$l"][: , 1])
        study = objs["index_$l"][:, 2]
        wins = Float64.(objs["wi_$l"])
        losses = Float64.(objs["lo_$l"])
        choice_t = transpose(trunc.(Int, objs["choice_$l"]))
        print(typeof(choice_t[:, 1]), "\n")
        for i in eachindex(subj)
            n_trials = length(losses[i, :])
            for j in 1:n_trials
                # for some reason there is always a win, even if there's a loss
                # so if the loss is <0, we need to set the win to for the outcome col
                outcome = losses[i, j] < 0 ? losses[i, j] : wins[i, j]
                push!(df, (
                    subj[i],
                    j,
                    study[i],
                    choice_t[j, i],
                    wins[i, j],
                    losses[i, j],
                    n_trials,
                    outcome,
                ))
            end
        end
    end

    return df
end