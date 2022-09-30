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

# Initialize results
results = InitializeTransition(SS, NSS, Z, γ, Π, Π₀, T, ρ, verbose)
@time SolveHHTransition(results, true)