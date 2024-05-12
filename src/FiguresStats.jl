using HDF5, MCMCChains, MCMCChainsStorage, Turing
using Plots, StatsPlots
using ArviZ, PSIS, ArviZPlots, ArviZPythonPlots


# patterns
pats = Dict(
    0 => "no_preference",
    1 => "preference_AB",
    2 => "preference_CD",
    4 => "preference_BD",
    5 => "preference_AB_BD",
    6 => "preference_CD_BD",
    8 => "preference_AC",
    10 => "preference_AB_AC"
)
# patterns
pats = Dict(
    6 => "preference_CD_BD",

)
# Load data
pvldelta_chains = Dict()
h5open("data/igt_pvldelta_data_chains.h5", "r") do file
    for pat in keys(pats)
        g = open_group(file, "pattern_$pat")
        pvldelta_chains[pat] = read(g, Chains)
    end
end

hgf_chains = Dict()
h5open("data/igt_hgf_data_chains.h5", "r") do file
    for pat in keys(pats)
        g = open_group(file, "choice_pattern_$pat")
        hgf_chains[pat] = read(g, Chains)
    end
end


# print parameters from each model (we only need a single chain and pattern)
for pat in keys(pats)
    println("Pattern: $(pats[pat])")
    println("PVL Delta")
    println(names(pvldelta_chains[pat]))
    println("HGF")
    println(names(hgf_chains[pat]))
    break
end

# for pvl delta, we want to compare the group level differences
# :w′μ, :w′σ, :c'μ, :c'σ, :a'μ, :a'σ, :A'μ, :A'σ, 
pvldelta_selected_params_sym::Vector{Symbol} = [
    Symbol("w′μ"),
    Symbol("w′σ"),
    Symbol("c′μ"),
    Symbol("c′σ"),
    Symbol("a′μ"),
    Symbol("a′σ"),
    Symbol("A′μ"),
    Symbol("A′σ"),
]




# Plotting
for pat in keys(pats)
    # PVL Delta
    c = pvldelta_chains[pat][:, pvldelta_selected_params_sym, :]
    StatsPlots.plot(c, legend=false, size=(1200, 1600))
    Plots.savefig("figures/igt_pvldelta_$(replace(pats[pat], " " => "_")).png")
    # break
    # # HGF
    # StatsPlots.plot(hgf_chains[pat], legend=false, size=(1200, 1600))
    # Plots.savefig("figures/igt_hgf_$(replace(pats[pat], " " => "_")).png")
    # break
end
