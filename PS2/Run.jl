##################################################
################### MODELLING ####################
##################################################

# Required packages
using Plots

# Include model
include("Model.jl")
include("Diagnostics.jl")

# Get primitives and results
@unpack β, α, S, ns, Π, A, na = Primitives()
results = Initialize()

# Solve model
SolveModel(results, true)

# Policy and value function plot
plot(A, results.value_func, labels = ["Employed" "Unemployed"])
plot(A, [results.policy_func A], labels = ["Employed" "Unemployed" "45 degree"])

# Invariant distribution
plot(A, results.μ, labels = ["Employed" "Unemployed"])

# Wealth distribution
W = WealthDistribution(results, -2)
## TODO: Calculate and plot wealth distribution

# Lorenz Curve and Gini index
lorenz = LorenzCurve(results)
Gini(results)
## TODO: Calculate and plot Lorenz Curve and Gini index

# λ plot and welfare analysis
λ = λ(results)
plot(A, λ, labels = ["Employed" "Unemployed"])

# Welfare comparison
WelfareComparision(results, λ)

# Proportion of population changing to complete markets
PreferComplete(results, λ)