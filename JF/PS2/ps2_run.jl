using DataFrames, StatFiles, LinearAlgebra, Optim, DelimitedFiles, Distributions, Random, LineSearches
include("ps2_functions.jl")

# read nodes and weights for Gaussian Quadrature
Q1 = readdlm("JF/PS2/KPU_d1_l20.asc", ',') # 1-dim quadrature nodes and weights
Q2 = readdlm("JF/PS2/KPU_d2_l20.asc", ',') # 2-dim quadrature nodes and weights

# read data
df = DataFrame(load("JF/PS1/Mortgage_performance_data.dta"))
df_X = df[:, ["score_0", "rate_spread", "i_large_loan", "i_medium_loan", "i_refinance", "age_r", "cltv", "dti", "cu", "first_mort_r", "i_FHA", "i_open_year2", "i_open_year3", "i_open_year4", "i_open_year5"]]
X = Matrix(df_X) # time-invariant characteristics
X = Float64.(X)
df_Z = df[:, ["score_0", "score_1", "score_2"]]
Z = Matrix(df_Z) # time-variantt characteristics
Z = Float64.(Z)

df_Y = df[:, ["i_close_0", "i_close_1", "i_close_2"]]
Y = Matrix(df_Y)
Y = Float64.(Y)
Y_4 = 1 .- sum(Y, dims=2)
Y = hcat(Y, Y_4) # observed choice

# random draw
Random.seed!(20221118) # set seed
U_ghk = rand(Uniform(0,1), 200) # draw from uniform distribution
U_ghk = reshape(U_ghk, :, 2) # reshape to 100 x 2 matrix
U_ar = rand(Normal(), 300) # draw from normal dist
U_ar = reshape(U_ar, :, 3) # reshape to 100 x 3 matrix

# default parameter
ρ = 0.5
α = [0.0, -1.0, -1.0]
γ = 0.3
β = zeros(size(X)[2])

θ_init = vcat(ρ, α, γ, β)

# compute log-likelihood using each method
LL_quadrature = LL(θ_init, Y, X, Z, Q1, Q2, U_ghk, method="quadrature")
LL_ghk = LL(θ_init, Y, X, Z, Q1, Q2, U_ghk, method="ghk")
LL_ar = LL(θ_init, Y, X, Z, Q1, Q2, U_ar, method="ar")

# estimate parameter using quadrature method
@elapsed opt = Optim.optimize(θ -> - LL(θ, Y, X, Z, Q1, Q2, U_ghk, method="quadrature"), θ_init, LBFGS(linesearch=BackTracking()))