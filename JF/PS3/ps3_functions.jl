using Statistics, Optim, LinearAlgebra, Parameters

# construct data
struct Data
    X::Matrix{Float64} # linear variables (JTxK)
    p::Vector{Float64} # nonlinear variable (price) (JTx1)
    Z::Matrix{Float64} # instruments (JTxL)
    y::Vector{Float64} # demographic draws (Rx1)
    share::Vector{Float64} # share (JTx1)
    δ₀::Vector{Float64} # initial δ (log(s)-log(s₀)) (JTx1)
    market_id::Vector{Int64} # market id (JTx1)
end

# aggregate demand given δ and μ
function agg_demand(δ::Vector{Float64}, μ::Matrix{Float64})
    # δ is Jx1 vector, μ is JxR matrix
    V = δ .+ μ # utility matrix (JxR)
    numerator = exp.(V) # numerator of choice prob (JxR)
    denominator = 1 .+ sum(exp.(V), dims=1) # denominator of choice prob (Rx1)
    denominator = reshape(denominator, 1, :) # reshape denominator as 1xR matrix
    choice_prob = numerator ./ denominator # each individual's choice prob of product j (JxR)
    share = mean(choice_prob, dims=2) # mean choice prob across individuals (Jx1 matrix)
    share = vec(share) # reshape share as Jx1 vector

    return share # Jx1 vector
end

# Jacobian of demand σ (for Newton method)
function agg_demand_Jacobian(σ::Vector{Float64})
    Dσ = - σ * σ' + diagm(σ) # ∂σ/∂δ (JxJ)
    Df = - Dσ ./ σ # ∂(log(s)-log(σ))/∂δ
    return Df
end

# update δ using BLP contraction
function update_δ_contraction(share::Vector{Float64}, δ::Vector{Float64}, μ::Matrix{Float64})
    σ = agg_demand(δ, μ) # predicted share
    δ_new = δ + log.(share) - log.(σ)
    return δ_new
end

# update δ using Newton method
function update_δ_newton(share::Vector{Float64}, δ::Vector{Float64}, μ::Matrix{Float64})
    σ = agg_demand(δ, μ)
    f = log.(share) - log.(σ) # distance b/w share and predicted share
    Df = agg_demand_Jacobian(σ)
    δ_new = δ - Df \ f # update δ using Newton method
    return δ_new
end

# contraction mapping
function contraction(share::Vector{Float64}, δ::Vector{Float64}, μ::Matrix{Float64}; err = 100.0, tol = 1e-12, tol_transit = 1.0)
    err_vec = 0.0 # initialize norm transition b/w δ and δ_new
    while err > tol # repeat until converge
        if err > tol_transit # use BLP contraction
            δ_new = update_δ_contraction(share, δ, μ)
        else # use Newton when distance b/w δ and δ_new is small
            δ_new = update_δ_newton(share, δ, μ)
        end
        err = norm(δ_new - δ) # distance b/w δ_new and δ
        err_vec = vcat(err_vec, err) # add err to err_vec
        δ = δ_new # update δ
    end

    return δ, err_vec
end

# two-step GMM in the first stage (linear parameter)
function gmm_twostep(δ::Vector{Float64}, X::Matrix{Float64}, Z::Matrix{Float64})
    # first step
    W₁ = inv(Z' * Z) # first weighting matrix
    β₁ =  (X' * Z * W₁ * Z' * X) \ (X' * Z * W₁ * Z' * δ) # β in first step

    # second step
    ξ = δ - X * β₁ # residuals
    Zξ = Z .* ξ # resudual-weighted instruments
    W₂ = inv(Zξ' * Zξ) # optimal weighting matrix (second step)
    β = (X' * Z * W₂ * Z' * X) \ (X' * Z * W₂ * Z' * δ) # linear parameter β in second step
    ρ = δ - X * β # structural residuals (depend on nonlinear parameter λ)

    return ρ, β, W₂
end

function blp(λ::Float64, data::Data; tol = 1e-12, tol_transit = 1.0)
    @unpack X, p, Z, y, share, δ₀, market_id = data # unpack variables

    # make μ
    μ = λ .* p * y' # JTxR matrix

    # solve δ using contraction
    δ = zeros(length(δ₀)) # initialize δ
    for t in unique(market_id) # solve for each market
        sₜ = share[market_id .== t] # share in market t
        δₜ = δ₀[market_id .== t] # δ₀ in market t
        μₜ = μ[market_id .== t, :] # μ in market t
        δ[market_id .== t] = contraction(sₜ, δₜ, μₜ, tol=tol, tol_transit=tol_transit)[1] # insert solved δ
    end

    # solve β using two-step GMM
    ρ, β, W₂ = gmm_twostep(δ, X, Z)

    # objective function (weighted sum of structural residuals)
    obj = ρ' * Z * W₂ * Z' * ρ

    return obj, β
end

function blp_grid_search(data::Data, λ_seq::Vector{Float64}; tol = 1e-12, tol_transit = 1.0)
    obj_vec = zeros(length(λ_seq)) # initialize obj_vec
    for i = 1:length(λ_seq)
        obj_vec[i] = blp(λ_seq[i], data, tol=tol, tol_transit=tol_transit)[1]
    end
    min, i_min = findmin(obj_vec)
    λ_min = λ_seq[i_min]

    return (obj = obj_vec, min = min, minimizer = λ_min)
end