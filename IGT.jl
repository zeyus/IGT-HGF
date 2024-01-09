"""
  IGT.jl

  Heirarchichal Gaussian Filter implementation of the Iowa Gambling Task
"""

using HierarchicalGaussianFiltering
using ActionModels
using StatsPlots
using Plots: plot, plot!  # manual import to fix linting issue
using Distributions
using RData
using Random

theme(:dark)

"""
    plot_subject_w_l_choices(
        subj::Int,
        study::String,
        wins::Vector{Int},
        losses::Vector{Int},
        choices::Vector{Int},
        title::String = "IGT Subject Choices"
    )

Plot the subject's wins, losses, and choices for the IGT.
"""
function plot_subject_w_l_choices(
    subj::Int,
    study::String,
    wins::Vector{Int},
    losses::Vector{Int},
    choices::Vector{Int},
    title::String = "IGT Subject Choices"
)::Plots.Plot
    gr(size=(1200, 1200))
    p1 = scatter(
        1:length(wins),
        wins,
        label="Wins",
        legend=:outertop,
        markerstrokewidth=0.5,
    )

    p2 = scatter(
        1:length(losses),
        losses,
        label="Losses",
        legend=:outertop,
        markerstrokewidth=0.5,
    )

    p3 = scatter(
        1:length(choices),
        choices,
        label="Choices",
        legend=:outertop,
        markerstrokewidth=0.5,
    )

    p4 = plot(
        1:length(wins),
        cumsum(wins .+ losses),
        label="Wins+Losses",
        legend=:outertop,
    )

    return plot(
        p1,
        p2,
        p3,
        p4,
        layout=(2,2),
        plot_title="$title: $subj ($study)",
    )
end



"""
    load_trial_data(
        filename::String,
    )::Dict{String, Dict{String, Any}}

Load the trial data from the IGT data file.
"""
function load_trial_data(
    filename::String,
)::Dict{String, Dict{String, Any}}
    # load the data
    objs = load(filename)

    # set trial lengths (str for indices)
    trials_label = ["95", "100", "150"]

    # create a trial data dictionary
    trial_data = Dict()

    # populate the trial data dictionary
    for l in trials_label
        trial_data[l] = Dict(
            # list of  subjects
            "subj" => parse.(Int, objs["index_$l"][: , 1]),
            # the study author
            "study" => objs["index_$l"][: , 2],
            # participant choice
            "choice" => transpose(trunc.(Int, objs["choice_$l"])),
            # participant wins
            "wins" => trunc.(Int, objs["wi_$l"]),
            # participant losses
            "losses" => trunc.(Int, objs["lo_$l"]),
        )
    end

    return trial_data
end


"""
    get_trial_data(
        trial_data::Dict{String, Dict{String, Any}},
        trial_length::String,
        subj::Int,
    )::Dict{String, Any}

Get the trial data for a given subject and trial length.
"""
function get_trial_data(
    trial_data::Dict{String, Dict{String, Any}},
    trial_length::String,
    subj::Int,
)::Dict{String, Any}
    # get the study
    study = trial_data[trial_length]["study"][subj]

    # get the wins
    wins = trial_data[trial_length]["wins"][subj, :]

    # get the losses
    losses = trial_data[trial_length]["losses"][subj, :]

    # get the choices
    choices = trial_data[trial_length]["choice"][:, subj]

    return Dict(
        "subj" => subj,
        "study" => study,
        "wins" => wins,
        "losses" => losses,
        "choices" => choices,
    )
end


"""
    random_trial(
        trial_data::Dict{String, Dict{String, Any}},
    )::Dict{String, Any}

Get a random trial from the trial data.
"""
function random_trial(
    trial_data::Dict{String, Dict{String, Any}},
)::Dict{String, Any}
    # get a random trial length
    trial_length_labels = keys(trial_data)
    random_trial_length = rand(trial_length_labels)

    # get a random subject
    get_trial_data(
        trial_data,
        random_trial_length,
        rand(1:length(trial_data[random_trial_length]["subj"])),
    )
end

trial_data = load_trial_data("data/IGTdataSteingroever2014/IGTdata.rdata")

random_trial_data = random_trial(trial_data)

plot_subject_w_l_choices(
    random_trial_data["subj"],
    random_trial_data["study"],
    random_trial_data["wins"],
    random_trial_data["losses"],
    random_trial_data["choices"],
)

