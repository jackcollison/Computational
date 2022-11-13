# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: November, 2022

# Include libraries
using DataFrames, StatFiles, Optim, CSV, Random, SharedArrays, Plots

# Include helpers
include("toolbox.jl")
include("aux.jl")

# Read data
data = DataFrame(load("/Users/jackcollison/Desktop/Wisconsin/Coursework/Second Year/Computational/Q2/PS1/MortgagePerformanceData.dta"))

# Create outcome column
data[!, :T] .= 0.0
data.T = ifelse.(data.i_close_0 .== 1, 1.0, data.T)
data.T = ifelse.((data.i_close_0 .== 0) .& (data.i_close_1 .== 1), 2.0, data.T)
data.T = ifelse.((data.i_close_0 .== 0) .& (data.i_close_1 .== 0) .& (data.i_close_2 .== 1), 3.0, data.T)
data.T = ifelse.((data.i_close_0 .== 0) .& (data.i_close_1 .== 0) .& (data.i_close_2 .== 0), 4.0, data.T)

# Select relevant variables
t = Float64.(Array(select(data, :T)))
x = Float64.(Array(select(data, :score_0, :rate_spread, :i_large_loan, :i_medium_loan, :i_refinance, :age_r, :cltv, :dti, :cu, :first_mort_r, :i_FHA, :i_open_year2, :i_open_year5)))
z = Float64.(Array(select(data, :score_0, :score_1, :score_2)))

# Initial parameter vector guess
α₀ =  0.0
α₁ = -1.0
α₂ = -1.0
β = Array(fill(0.0, size(x, 2)))
γ = 0.3
ρ = 0.5

# Quadrature points
Q = initialize_quadrature_integration()

# Shocks from Halton sequences (required for quadrature method too)
u₀, u₁, u₂ = initialize_ghk(;use_halton = true)
ε₀, ε₁, ε₂ = initialize_accept_reject(ρ; use_halton = true)

# Quadrature 
@elapsed QL = likelihood(α₀, α₁, α₂, β, γ, ρ, t, x, z, Q[1], Q[2], u₀, u₁, u₂, ε₀, ε₁, ε₂; method = "quadrature")

# Plot quadrature likelihood
L1 = [QL[i] for i = 1:size(QL,1) if t[i] == 1]
L2 = [QL[i] for i = 1:size(QL,1) if t[i] == 2]
L3 = [QL[i] for i = 1:size(QL,1) if t[i] == 3]
L4 = [QL[i] for i = 1:size(QL,1) if t[i] == 4]
histogram(L1)
histogram!(L2)
histogram!(L3)
histogram!(L4)

# GHK 
@elapsed GHKL = likelihood(α₀, α₁, α₂, β, γ, ρ, t, x, z, Q[1], Q[2], u₀, u₁, u₂, ε₀, ε₁, ε₂; method = "ghk")

# Plot GHK likelihood
L1 = [GHKL[i] for i = 1:size(GHKL,1) if t[i] == 1]
L2 = [GHKL[i] for i = 1:size(GHKL,1) if t[i] == 2]
L3 = [GHKL[i] for i = 1:size(GHKL,1) if t[i] == 3]
L4 = [GHKL[i] for i = 1:size(GHKL,1) if t[i] == 4]
histogram(L1)
histogram!(L2)
histogram!(L3)
histogram!(L4)

# Accept-Reject 
@elapsed ARL = likelihood(α₀, α₁, α₂, β, γ, ρ, t, x, z, Q[1], Q[2], u₀, u₁, u₂, ε₀, ε₁, ε₂; method = "accept_reject")

# Plot GHK likelihood
L1 = [ARL[i] for i = 1:size(ARL,1) if t[i] == 1]
L2 = [ARL[i] for i = 1:size(ARL,1) if t[i] == 2]
L3 = [ARL[i] for i = 1:size(ARL,1) if t[i] == 3]
L4 = [ARL[i] for i = 1:size(ARL,1) if t[i] == 4]
histogram(L1)
histogram!(L2)
histogram!(L3)
histogram!(L4)