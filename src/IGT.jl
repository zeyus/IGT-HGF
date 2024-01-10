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
using CategoricalArrays

theme(:dark)

"""
    add_scale_breaks!(
        p::Plots.Plot,
        axis::Symbol,
        n_breaks::Int,
        break_symbol::String = "≠",
        axis_offset::Float64 = 0.0,
    )::Plots.Plot

Add scale breaks to a plot.
"""
function add_scale_breaks!(
    p::Plots.Plot,
    axis::Symbol,
    n_breaks::Int,
    break_symbol::String = "≠",
    axis_offset::Float64 = 0.0,
)::Plots.Plot
    # default is 1.06: see https://docs.juliaplots.org/latest/generated/attributes_axis/
    w_fact = p.subplots[1].attr[axis].plotattributes[:widen] == :auto ? 1.06 : p.attr[axis][:widen]
    # nudge so breaks are centered between ticks
    nudge = (1.0 - w_fact) / 2.0
    # distance between breaks
    break_dist = w_fact/(n_breaks+1)
    # all break locations
    breaks = [(break_dist*n)+nudge for n in 1:(n_breaks)]
    break_line = [axis_offset for _ in 1:(length(breaks))]
    break_text = [text(
        break_symbol,
        0.3,
        p.attr[:foreground_color],
        :center,
    ) for _ in 1:(length(breaks))]

    # Annotations are a vector of tuples of ((x, y), text)
    # if they use relative (%) positions.
    break_xytxt = collect(
        zip(
            zip(
                axis == :xaxis ? breaks : break_line,
                axis == :xaxis ? break_line : breaks,
            ),
            break_text
        )
    )

    p = annotate!(
        p,
        break_xytxt,
        textsize=0.5,
    )
    return p
end

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
    cwins = CategoricalArray(wins)
    nwins = length(unique(wins))
    closses = CategoricalArray(losses)
    nlosses = length(unique(losses))
    cchoices = CategoricalArray(choices)
    ntrials = length(wins)
    x = 1:ntrials

    # plot the wins
    p1 = scatter(
        x,
        cwins,
        label="Wins",
        legend=:outertop,
        markerstrokewidth=0.5,
        yscale=:identity,
        ylabel="Win amount",
    )
    p1 = add_scale_breaks!(
        p1,
        :yaxis,
        nwins-1,
    )

    # plot the losses
    p2 = scatter(
        x,
        closses,
        label="Losses",
        legend=:outertop,
        markerstrokewidth=0.5,
        yscale=:identity,
        ydiscrete_values=0:100:1000,
        ylabel="Loss amount",
    )
    p2 = add_scale_breaks!(
        p2,
        :yaxis,
        nlosses-1,
    )

    # plot subject choices
    p3 = scatter(
        x,
        cchoices,
        label="Choices",
        legend=:outertop,
        markerstrokewidth=0.5,
        yscale=:identity,
        ylabel="Deck Choice",
    )

    # plot running total
    p4 = plot(
        x,
        cumsum(wins .+ losses),
        label="Wins+Losses",
        legend=:outertop,
        ylabel="Balance",
    )

    return plot(
        p1,
        p2,
        p3,
        p4,
        layout=(2,2),
        plot_title="$title: $subj ($study)",
        xlabel="Trial",
        link=:x,
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

p = plot_subject_w_l_choices(
    random_trial_data["subj"],
    random_trial_data["study"],
    random_trial_data["wins"],
    random_trial_data["losses"],
    random_trial_data["choices"],
)

