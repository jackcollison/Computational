using Plots, Optim, DataFrames, CSV, Tables, Distributed, SharedArrays, ProgressMeter, Statistics, StatFiles, Random, Distributions

cd("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS9 - Solution")

# Parallel processing
# workers()
# addprocs(4)

# Call toolbox and auxiliar functions
include("toolbox.jl");
include("aux.jl");

# Load data
df = DataFrame(load("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS9 - Solution/Mortgage_performance_data.dta"))

# Define X, Y, and Z.
df_x = select(df,
              :score_0, :rate_spread, :i_large_loan, :i_medium_loan,
              :i_refinance, :age_r, :cltv, :dti, :cu,  :first_mort_r, :i_FHA,
              :i_open_year2, :i_open_year3, :i_open_year4, :i_open_year5)
df_t = select(df, :duration)
df_z = select(df, :score_0, :score_1, :score_2)

# Define matrices.
x = Float64.(Array(df_x))
z = Float64.(Array(df_z))
t = Float64.(Array(df_t))

# Initial parameter vector guess
α₀ =  0.0;
α₁ = -1.0;
α₂ = -1.0;
β   = Array(fill(0.0, size(x)[2]));
γ   = 0.3;
ρ   = 0.5;

# Quadrature points
q_grids = initialize_quadrature_integration()

# Random shocks
u₀_h, u₁_h, u₂_h = initialize_ghk(;use_halton = true)
ε₀_h, ε₁_h, ε₂_h = initialize_accept_reject(ρ; use_halton = true)
u₀_p, u₁_p, u₂_p = initialize_ghk(;use_halton = false)
ε₀_p, ε₁_p, ε₂_p = initialize_accept_reject(ρ; use_halton = false)

# Quadrature Method Integration
@time likelihoods_quadrature = likelihood(α₀, α₁, α₂, β, γ, ρ, t, x, z, q_grids[1], q_grids[2], u₀_h, u₁_h, u₂_h, ε₀_h, ε₁_h, ε₂_h; method = "quadrature")

# GHK Method
@time likelihoods_ghk        = likelihood(α₀, α₁, α₂, β, γ, ρ, t, x, z, q_grids[1], q_grids[2], u₀_h, u₁_h, u₂_h, ε₀_h, ε₁_h, ε₂_h; method = "ghk")
@time likelihoods_ghk_pseudo = likelihood(α₀, α₁, α₂, β, γ, ρ, t, x, z, q_grids[1], q_grids[2], u₀_p, u₁_p, u₂_p, ε₀_p, ε₁_p, ε₂_p; method = "ghk")

# Accept-Reject Method
@time likelihoods_accept_reject        = likelihood(α₀, α₁, α₂, β, γ, ρ, t, x, z, q_grids[1], q_grids[2], u₀_h, u₁_h, u₂_h, ε₀_h, ε₁_h, ε₂_h; method = "accept_reject")
@time likelihoods_accept_reject_pseudo = likelihood(α₀, α₁, α₂, β, γ, ρ, t, x, z, q_grids[1], q_grids[2], u₀_p, u₁_p, u₂_p, ε₀_p, ε₁_p, ε₂_p; method = "accept_reject")

# Save results
result = copy(df)

result[!, :likelihood_quadrature]           = likelihoods_quadrature
result[!, :likelihood_ghk]                  = likelihoods_ghk
result[!, :likelihood_ghk_pseudo]           = likelihoods_ghk_pseudo
result[!, :likelihood_accept_reject]        = likelihoods_accept_reject
result[!, :likelihood_accept_reject_pseudo] = likelihoods_accept_reject_pseudo

CSV.write("result.csv", result)

# LBFGS optimization
optimization_results = optimize(θ -> -log_likelihood(θ, t, x, z, q_grids[1], q_grids[2], u_0_h, u_1_h, u_2_h, ε_0_h, ε_1_h, ε_2_h),
                                vcat(α_0, α_1, α_2, β, γ, ρ),
                                LBFGS())
