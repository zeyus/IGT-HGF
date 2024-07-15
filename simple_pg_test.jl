using BenchmarkTools

# Define the N x M payoff matrix
N = 1000
M = 1000
payoff = rand(-10:10, N, M)

# Define the M length shape vector
shape = rand(1:5, M)

abs_payoff = abs.(payoff)

# Benchmark the abs. operation
@btime abs.($payoff)

# Benchmark the abs. operation with broadcasting
@btime $abs_payoff .^ $shape

# Benchmark the full operation
@btime abs.($payoff) .^ $shape'
