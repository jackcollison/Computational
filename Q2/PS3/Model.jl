# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: November, 2022

# Import packages
using DataFrames, StatFiles, LinearAlgebra, Statistics

# Multinomial logit function
function p(x)
    # Overflow trick
    m = max(0.0, maximum(x))
    n = exp.(x .- m)

    # Return value
    return n ./ (exp.(-m) .+ sum(n, dims=1))
end

# Contraction mapping within a market
function contraction_mapping_market(data::DataFrame, market::Float64, markets::Vector{Union{Missing, Float64}}, Y::Array{Float64}, λₚ::Float64; tol::Float64=1e-12, newton_tol::Float64=1.0, maxiter::Int64=1000, verbose::Bool=false)
    # Find index of current market and initialize
    index = (markets .== market)
    P = data[index, :price]
    S = data[index, :share]
    δ₀ = data[index, :delta_iia]

    # Useful values  
    R = length(Y)
    J = length(S)

    # Pre-compute μ
    μ = λₚ * P * Y'

    # Initialize errors, results, counter
    error = Inf
    errors = []
    i = 0
    δ = nothing

    # Iterate while error is large
    while error > tol && i < maxiter
        # Increment counter
        i += 1

        # Compute choice probabilities
        σ = p(δ₀ .+ μ)
        𝛔 = sum(σ, dims=2) / R

        # Check Newton condition
        if error > newton_tol
            # Update with contraction mapping
            δ = δ₀ + log.(S) - log.(𝛔)
        else
            # Update with Newton step
            Δ = (1 / R) * ((I(J) .* (σ * (1 .- σ)')) - ((1 .- I(J)) .* (σ * σ'))) ./ 𝛔
            δ = δ₀ + inv(Δ) * (log.(S) - log.(𝛔))
        end

        # Compute error and save
        error = maximum(abs.((δ - δ₀)))
        push!(errors, error)

        # Print statement
        if verbose
            println("Iteration i = ", i, " for market ", market, " with error ε = ", error)
        end

        # Set new baseline
        δ₀ = δ
    end

    # Return values
    return δ, errors
end

# Contraction mapping across markets
function contraction_mapping(data::DataFrame, markets::Vector{Union{Missing, Float64}}, Y::Array{Float64}, λₚ::Float64; tol::Float64=1e-12, newton_tol::Float64=1.0, maxiter::Int64=1000, verbose::Bool=false)
    # Initialize values
    δ = []
    
    # Loop over all markets
    for market in unique(markets)
        # Contraction mapping for each market
        δ = vcat(δ, contraction_mapping_market(data, market, markets, Y, λₚ, tol=tol, newton_tol=newton_tol, maxiter=maxiter, verbose=verbose)[1])
    end

    # Return value
    return δ
end

# Function for ρ
function ρ(data::DataFrame, markets::Vector{Union{Missing, Float64}}, Y::Array{Float64}, λₚ::Float64, W::Array{Float64}; tol::Float64=1e-12, newton_tol::Float64=1.0, maxiter::Int64=1000, verbose::Bool=false)
    # Compute δ and β
    δ = contraction_mapping(data, markets, Y, λₚ, tol=tol, newton_tol=newton_tol, maxiter=maxiter, verbose=verbose)
    β = inv((X' * Z) * W * (Z' * X)) * (X' * Z) * W * Z' * δ

    # Return ρ
    return δ - X * β
end

# GMM objective function
function gmm(data::DataFrame, markets::Vector{Union{Missing, Float64}}, Z::Matrix{Union{Missing, Float64}}, Y::Array{Float64}, λₚ::Array{Float64}, W::Array{Float64}; tol::Float64=1e-12, newton_tol::Float64=1.0, maxiter::Int64=1000, verbose::Bool=false)
    # Compute ξ and return
    λₚ = λₚ[1]
    ξ = ρ(data, markets, Y, λₚ, W, tol=tol, newton_tol=newton_tol, maxiter=maxiter, verbose=verbose)
    return (ξ' * Z * W * Z' * ξ)[1,1]
end