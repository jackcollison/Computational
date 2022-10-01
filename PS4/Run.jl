# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: September, 2022

##################################################
################### MODELLING ####################
##################################################

# Include packages
using Plots

# Include models
include("Model.jl")
include("/Users/jackcollison/Desktop/Wisconsin/Coursework/Second Year/Computational/PS3/Model.jl")

# Get primitives
@unpack N, n, Jᴿ, σ, β, η, α, δ, A, na = Primitives()

# Social security
SS = 0.11
NSS = 0.0

# Idiosyncratic uncertainty
Z = [3.0 ; 0.5]
Π = [0.9261 0.0739; 0.0189 0.9811] 
Π₀ = [0.2037 0.7963]

# Labor elasticity
γ = 0.42

# Hyperparameters
T = 30
ρ = 0.7
verbose = true

# Solve model with idiosyncratic uncertainty and social security
SS¹ = Initialize(SS, Z, γ, Π, Π₀)
@time SolveModel(SS¹, verbose, ρ)

# Solve model with idiosyncratic uncertainty and no social security
SS² = Initialize(NSS, Z, γ, Π, Π₀)
@time SolveModel(SS², verbose, ρ)

# Solve transition model
results = InitializeTransition(SS¹, SS², Z, γ, Π, Π₀, T, verbose)
@time SolveModelTransition(results, SS¹, SS², verbose, ρ)

# TESTING
# Check value function
plot(A, hcat(SS¹.value_func[20, :, :], results.V₀[20, :, :]))
plot(1:T, results.r)

# Plot capital path
plot(1:T, results.K, label = "Capital Path", legend = :bottomright)
hline!([SS¹.K], linestyle=:dash, color = :darkred, label = :none)
hline!([SS².K], linestyle=:dash, color = :darkred, label = :none)

# Plot labor path
plot(1:T, results.L, label = "Wage Path", legend = :bottomright)
hline!([SS¹.L], linestyle=:dash, color = :darkred, label = :none)
hline!([SS².L], linestyle=:dash, color = :darkred, label = :none)

# Plot wage path
plot(1:T, results.w, label = "Wage Path", legend = :bottomright)
hline!([SS¹.w], linestyle=:dash, color = :darkred, label = :none)
hline!([SS².w], linestyle=:dash, color = :darkred, label = :none)

# Plot interest rate path
plot(1:T, results.r, label = "Wage Path", legend = :bottomright)
hline!([SS¹.r], linestyle=:dash, color = :darkred, label = :none)
hline!([SS².r], linestyle=:dash, color = :darkred, label = :none)