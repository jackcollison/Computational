# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: November, 2022

# Include model
include("Model.jl")

# Baseline results
λ = -4.0
p = Primitives()
V̄ = zeros(36)
V̄_opt = SolveEV(p, λ, V̄; verbose=true)

# Simulation results
𝐏 = 1 ./ (1 .+ exp.(-(p.P₁ - p.P₀)))
results = SolveCCP(p, λ, 𝐏; verbose=true)
test1 = results[1]
test2 = results[2]

test = test1 .* test2