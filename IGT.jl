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



objs = load("data/IGTdataSteingroever2014/IGTdata.rdata")

trials_label = ["95", "100", "150"]
trials_n = [95, 100, 150]
trial_data = Dict()

for l in trials_label
    trial_data[l] = Dict(
        "subj" => objs["index_$l"][: , 1],
        "study" => objs["index_$l"][: , 2],
        "choice" => transpose(objs["choice_$l"]),
        "wins" => objs["wi_$l"],
        "losses" => objs["lo_$l"],
    )
end

# Plot the data
for (i,l) in enumerate(trials_label)
    display(
        plot(1:trials_n[i],
        trial_data[l]["choice"],
        labels=permutedims(trial_data[l]["subj"]),
        legend=:topleft,
        title="IGT data for $l trials")
    )
end
