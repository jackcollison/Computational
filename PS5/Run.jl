# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: October, 2022

##########################################################
###################### PROGRAM RUNNER ####################
##########################################################

# Include model
include("Model.jl")

# Initialize
results = Initialize()

# Get primitives
P = Primitives()
G = Grids()
S = Shocks()

# Draw shocks
sᵢₜ, Sₜ = DrawShocks(S, P.N, P.T)

next_policy_func, next_value_func = Bellman(P, G, S, results)

# Solve household problems, update capital path, and run autoregression
SolveHousehold(results, true)
K_path = CapitalSimulation(results, sᵢₜ, Sₜ)
â₀, â₁, b̂₀, b̂₁, R̂² = AutoRegression(R, Sₜ, K_path)