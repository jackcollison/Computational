# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: September, 2022

##################################################
################### MODELLING ####################
##################################################

# Include packages
using Plots

# Include model
include("Model.jl")
include("Diagnostics.jl")

# Get primitives
@unpack N, n, Jᴿ, σ, β, η, α, δ, A, na = Primitives()

# Social security scenarios
SS = 0.11
NSS = 0.0

# Idiosyncratic uncertainty scenarios
Z = [3.0 ; 0.5]
Zᶜ = [0.5]
Π = [0.9261 0.0739; 0.0189 0.9811] 
Πᶜ = hcat(1.)
Π₀ = [0.2037 0.7963]
Π₀ᶜ = hcat(1.)

# Labor elasticity scenarios
γ = 0.42
γᶜ = 1.0

# Verbosity
verbose = true

# Solve household problem with idiosyncratic uncertainty and social security
results = Initialize(SS, Z, γ, Π, Π₀)
@time SolveHH(results)

# Plot various functions at various model-ages
plot(A, results.value_func[50, :, 1], labels = "Value Function", legend = :bottomright, title = "Value Function (Model-Age 50)")
plot(A, results.value_func[20, :, 1], labels = "Value Function", legend = :bottomright, title = "Value Function (Model-Age 20)")
plot([A A], [A results.policy_func[20, :, :]], labels = ["45 degree" "High Type" "Low Type"], legend = :bottomright, title = "Savings Function (Model-Age 20)")
plot(A, results.labor_supply[20, :, :], labels = ["High Type" "Low Type"], title = "Labor Decision (Model-Age 20)")
plot(A, results.policy_func[20, :, :] .- A, labels = ["High Type" "Low Type"], title = "Savings Decision (Model-Age 20)")

# Solve model with idiosyncratic uncertainty and social security
results = Initialize(SS, Z, γ, Π, Π₀)
@time SolveModel(results, verbose, 0.7)
FormatResults(results)

# Solve model with idiosyncratic uncertainty and no social security
results = Initialize(NSS, Z, γ, Π, Π₀)
@time SolveModel(results, verbose, 0.7)
FormatResults(results)

# Solve model with social security and no uncertainty
results = Initialize(SS, Zᶜ, γ, Πᶜ, Π₀ᶜ)
@time SolveModel(results, verbose, 0.1)
FormatResults(results)

# Solve model without social security and no uncertainty
results = Initialize(NSS, Zᶜ, γ, Πᶜ, Π₀ᶜ)
@time SolveModel(results, verbose, 0.1)
FormatResults(results)

# Solve model with idiosyncratic uncertainty, social security, and exogenous labor
results = Initialize(SS, Z, γᶜ, Π, Π₀)
@time SolveModel(results, verbose, 0.7)
FormatResults(results)

# Solve model with idiosyncratic uncertainty, no social security, and exogenous labor
results = Initialize(NSS, Z, γᶜ, Π, Π₀)
@time SolveModel(results, verbose, 0.7)
FormatResults(results)