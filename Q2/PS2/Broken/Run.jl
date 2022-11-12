# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: November, 2022

# Include libraries
using DataFrames, StatFiles, Optim

# Include model
include("Model.jl")

# Read data
data = DataFrame(load("/Users/jackcollison/Desktop/Wisconsin/Coursework/Second Year/Computational/Q2/PS1/MortgagePerformanceData.dta"))

# Create outcome column
data[!, :T] .= 0.0
data.T = ifelse.(data.i_close_0 .== 1, 1.0, data.T)
data.T = ifelse.((data.i_close_0 .== 0) .& (data.i_close_1 .== 1), 2.0, data.T)
data.T = ifelse.((data.i_close_0 .== 0) .& (data.i_close_1 .== 0) .& (data.i_close_2 .== 1), 3.0, data.T)
data.T = ifelse.((data.i_close_0 .== 0) .& (data.i_close_1 .== 0) .& (data.i_close_2 .== 0), 4.0, data.T)

# Select relevant variables
T = Float64.(Array(select(data, :T)))
X = Float64.(Array(select(data, :score_0, :rate_spread, :i_large_loan, :i_medium_loan, :i_refinance, :age_r, :cltv, :dti, :cu, :first_mort_r, :i_FHA, :i_open_year2, :i_open_year5)))
Z = Float64.(Array(select(data, :score_0, :score_1, :score_2)))

# Read data
p₁ = readdlm("/Users/jackcollison/Desktop/Wisconsin/Coursework/Second Year/Computational/Q2/PS2/KPU_d1_l20.csv", ',', Float64)
p₂ = readdlm("/Users/jackcollison/Desktop/Wisconsin/Coursework/Second Year/Computational/Q2/PS2/KPU_d2_l20.csv", ',', Float64)

# Initialize parameters
α₀ = 0.0
α₁ = -1.0
α₂ = -1.0
β = zeros(size(X, 2))
γ = 0.3
ρ = 0.5

# Quadrature method
w = p₂[:,3]
u = p₂[:,1:2]
LQ(w, u, α₀, α₁, α₂, β, γ, ρ, T[1], X[1,:], Z[1,:])

# GHK method
GHKLikelihood(α₀, α₁, α₂, β, γ, ρ, T, X, Z)