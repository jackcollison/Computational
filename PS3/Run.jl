# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: September, 2022

##################################################
################### MODELLING ####################
##################################################

# Include packages and set cores
using Distributed, Plots
# addprocs(4)

# Include model
include("Model.jl")
include("Diagnostics.jl")

# Get primitives
@unpack N, n, Jᴿ, σ, β, η, Π, Π₀, α, δ, A, na = Primitives()

# Solve model with idiosyncratic uncertainty and social security
results = Initialize(0.11, [3.0 ; 0.5], 0.42)
@time SolveModel(results, true, 0.7)

# Plot value function of retiree at model-age fifty and policy function of worker at model-age twenty
plot(A, results.value_func[50, :, 1], labels = "Value Function", legend = :bottomright, title = "Value Function (Model-Age 50)")
plot([A A], [A results.policy_func[20, :, :]], labels = ["45 degree" "High Type" "Low Type"], legend = :bottomright, title = "Savings Function (Model-Age 20)")

# Format baseline results
FormatResults(results)

# Solve model with idiosyncratic uncertainty and no social security
@everywhere results = Initialize(0.0, [3.0 ; 0.5], 0.42)
@time SolveModel(results, true, 0.7)
FormatResults(results)

# Solve model with social security and no uncertainty
@everywhere results = Initialize(0.11, [0.5], 0.42)
@time SolveModel(results, true, 0.9, 5e-3)
FormatResults(results)

# Solve model without social security and no uncertainty
@everywhere results = Initialize(0.0, [0.5], 0.42)
@time SolveModel(results, true, 0.9, 5e-3)
FormatResults(results)

# Solve model with idiosyncratic uncertainty, social security, and exogenous labor
@everywhere results = Initialize(0.11, [3.0 ; 0.5], 1.0)
@time SolveModel(results, true, 0.7)
FormatResults(results)

# Solve model with idiosyncratic uncertainty, no social security, and exogenous labor
@everywhere results = Initialize(0.0, [3.0 ; 0.5], 1.0)
@time SolveModel(results, true, 0.7)
FormatResults(results)