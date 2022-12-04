# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: November, 2022

# Include libraries and model
using Optim
include("Model.jl")

# Baseline results
λ = -4.0
p = Primitives()
V̄ = zeros(36)
SolveEV(p, λ, V̄; verbose=true)

# Simulation results
𝐏 = 1 ./ (1 .+ exp.(-(p.P₁ - p.P₀)))
SolveCCP(p, λ, 𝐏; verbose=true)

# Log-likelihood at true value
NXFP(p, [λ], 𝐏; verbose=true)

# Optimization
@time λ̂ = optimize(λ -> NXFP(p, λ, 𝐏), [-4.0], BFGS())
λ̂.minimizer
