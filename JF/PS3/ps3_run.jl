using DataFrames, StatFiles, LinearAlgebra, Optim, Distributions, LineSearches, Parameters, Plots
include("ps3_functions.jl")

# read data
df_X = DataFrame(load("JF/PS3/Car_demand_characteristics_spec1.dta"))
df_Z = DataFrame(load("JF/PS3/Car_demand_iv_spec1.dta"))
df_y = DataFrame(load("JF/PS3/Simulated_type_distribution.dta"))

# construct data
market_id = Int64.(df_X[:,"Year"]) # market id (year)
share = Float64.(df_X[:, "share"]) # share
δ₀ = Float64.(df_X[:, "delta_iia"]) # initial δ
p = Float64.(df_X[:, "price"]) # price (nonlinear variable)
X = Float64.(Matrix(df_X[:, 6:end])) # linear variables
Z₁ = X[:, 2:end] # exogenous linear variables (excluding price)
Z₂ = Float64.(Matrix(df_Z[:, 3:end])) # instruments for endogenous variables
Z = hcat(Z₁, Z₂) # full instruments
y = Float64.(vec(Matrix(df_y))) # simulated demographics (income)

data = Data(X, p, Z, y, share, δ₀, market_id) # construct data

# plot comparison of speed of convergence b/w contraction & contraction+Newton
λ₀ = 0.6 # set nonlinear parameter
μ₀ = λ₀ .* p * y' # calculate μ
μ₀_1985 = μ₀[market_id .== 1985, :] # μ₀ in 1985
δ₀_1985 = δ₀[market_id .== 1985] # δ₀ in 1985
share_1985 = share[market_id .== 1985] # share in 1985
δ_err_newton = contraction(share_1985, δ₀_1985, μ₀_1985)[2] # err_vec with contraction and newton
δ_err_contraction = contraction(share_1985, δ₀_1985, μ₀_1985, tol_transit=0)[2] # err_vec with only contraction
Plots.plot(collect(60:length(δ_err_contraction)), δ_err_contraction[60:end], xlabel="iteration", label="err only with contraction")
Plots.plot!(collect(60:length(δ_err_newton)), δ_err_newton[60:end], label="err with contraction and newton")

# grid search of λ
λ_seq = collect(range(0.0, 1.0, length=100)) # sequence of λ for search
@elapsed λ_search = blp_grid_search(data, λ_seq) # grid search
λ_search.minimizer # λ that minimize objective
Plots.plot(λ_seq, λ_search.obj) # plot objective

# BLP
λ_init = λ_search.minimizer # initial value of λ
@elapsed blp_opt = Optim.optimize(λ -> blp(first(λ), data)[1], [λ_init], LBFGS(linesearch=BackTracking())) # find optimal λ
blp_opt.minimizer # minimizer