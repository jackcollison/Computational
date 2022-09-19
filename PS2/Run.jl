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
@everywhere include("Diagnostics.jl")

# Get primitives and results
@everywhere @unpack β, α, S, ns, Π, A, na = Primitives()
@everywhere results = Initialize()

# Solve model
@time SolveModel(results, true)

# Policy and value function plot
plot(A, results.value_func, labels = ["Employed" "Unemployed"])
plot(A, [results.policy_func A], labels = ["Employed" "Unemployed" "45 degree"])
plot(A, results.policy_func - [A A], labels = ["Employed" "Unemployed"])

# Invariant distribution
plot(A, results.μ, labels = ["Employed" "Unemployed"])

# Wealth distribution
W = WealthDistribution(results)
plot([W[1:na, 1] W[(na + 1):(2 * na), 1]], [W[1:na, 2] W[(na + 1):(2 * na), 2]], labels = ["Employed" "Unemployed"])

# Lorenz Curve and Gini index
Lorenz = LorenzCurve(W)
plot([Lorenz[:,1] Lorenz[:,1]], [Lorenz[:,2] Lorenz[:,1]], labels = ["Lorenz Curve" "45 degree"])
Gini(Lorenz)

# λ plot and welfare analysis
l = λ(results)
plot(A, l, labels = ["Employed" "Unemployed"])

# Welfare comparison
WelfareComparision(results, l)

# Proportion of population changing to complete markets
PreferComplete(results, l)