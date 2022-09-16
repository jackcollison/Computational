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
@time SolveModel(results, true)

# Policy and value function plot
plot(A, results.value_func, labels = ["Employed" "Unemployed"])
plot(A, [results.policy_func A], labels = ["Employed" "Unemployed" "45 degree"])
plot(A, results.policy_func - [A A], labels = ["Employed" "Unemployed"])

# Invariant distribution
plot(A, results.μ, labels = ["Employed" "Unemployed"])

# Wealth distribution
W = WealthDistribution(results)
plot(A, W, labels = ["Employed" "Unemployed"])

# Lorenz Curve and Gini index
lorenz = LorenzCurve(W)

Gini(results)
## TODO: Calculate and plot Lorenz Curve and Gini index

# λ plot and welfare analysis
λ = λ(results)
plot(A, λ, labels = ["Employed" "Unemployed"])

# Welfare comparison
WelfareComparision(results, λ)

# Proportion of population changing to complete markets
PreferComplete(results, λ)