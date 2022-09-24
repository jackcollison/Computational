##################################################
################### MODELLING ####################
##################################################

# Required packages
using Distributed
@everywhere using Plots

# Set cores
addprocs(4)

# Include model
@everywhere include("Model.jl")

# Get primitives
@everywhere @unpack N, n, Jᴿ, σ, β, η, Π, Π₀, α, δ, A, na = Primitives()

# Solve model with idiosyncratic uncertainty and social security
@everywhere results = Initialize(0.11, [3.0 ; 0.5], 0.42)
@time SolveModel(results, true, 0.7)

# Solve model with idiosyncratic uncertainty and no social security
@everywhere results = Initialize(0.0, [3.0 ; 0.5], 0.42)
@time SolveModel(results, true, 0.7)