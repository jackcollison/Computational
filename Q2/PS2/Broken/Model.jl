# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: November, 2022

# Import packages
using Random, Distributions, DelimitedFiles

# Import utilities
include("Utilities.jl")

# Quadrature likelihood helper
function LQ(w::Array{Float64}, u::Array{Float64}, α₀::Float64, α₁::Float64, α₂::Float64, β::Array{Float64}, γ::Float64, ρ::Float64, T::Float64, X::Array{Float64}, Z::Array{Float64})
    # Initialize values
    F = Normal()
    σ₀ = 1 / (1 - ρ)^2

    # Check cases
    if T == 1.0
        # Return result for T = 1
        return pdf(F, (α₀ + X' * β + Z[1] * γ) / σ₀)
    elseif T == 2.0
        # Return result for T = 2
        f₂(ε) = cdf(F, -α₁ - X' * β - Z[2] * γ - ρ * ε[1]) * pdf(F, ε[1] / σ₀) / σ₀
        return Quadrature(f₂, w, u, a=[-α₀ - X' * β - Z[2]* γ])
    elseif T == 3.0
        # Return result for T = 3
        f₃(ε) = cdf(F, -α₂ - X' * β - Z[3] * γ - ρ * ε[2]) * pdf(F, ε[2] - ρ * ε[1]) * pdf(F, ε[1] / σ₀) / σ₀
        return Quadrature(f₃, w, u, a=[α₀ + X' * β + Z[3] * γ, α₁ + X' * β + Z[3] * γ])
    elseif T == 4.0
        # Return result for T = 4
        f₄(ε) = cdf(F, α₂ + X' * β + Z[3] * γ - ρ * ε[2]) * pdf(F, ε[2] - ρ * ε[1]) * pdf(F, ε[1] / σ₀) / σ₀
        return Quadrature(f₄, w, u, [α₀ + X' * β + Z[3] * γ, α₁ + X' * β + Z[3] * γ])
    else
        # Return error
        error("Please enter a valid number for T.")
    end
end

# Quadrature likelihood
function QuadratureLikelihood(α₀::Float64, α₁::Float64, α₂::Float64, β::Array{Float64}, γ::Float64, ρ::Float64, T::Float64, X::Array{Float64}, Z::Array{Float64}, p₁::Matrix{Float64}, p₂::Matrix{Float64})
    # Initialize results
    results = zeros(size(X, 1))

    # Initialize weights and points
    w₁ = p₁[:,2]
    u₁ = p₁[:,1]
    w₂ = p₂[:,3]
    u₂ = p₂[:,1:2]

    # Iterate over points
    for i in 1:size(X, 1)
        # Points and weights
        w = nothing
        u = nothing

        # Check cases
        if T[i] == 3 || T[i] == 4
            # Update
            w = w₂
            u = u₂
        elseif T[i] == 2
            # Update
            w = w₁
            u = u₁
        end

        # Calculate likelihood
        results[i] = LQ(w, u, α₀, α₁, α₂, β, γ, ρ, T[i], X[i,:], Z[i,:])
    end

    # Return value
    return results
end

# GHK helper
function GHK(α₀::Float64, α₁::Float64, α₂::Float64, β::Array{Float64}, γ::Float64, ρ::Float64, T::Float64, X::Array{Float64}, Z::Array{Float64})
    # Initialize values
    F = Normal()
    σ₀ = 1 / (1 - ρ)^2
    # ϕ₁ = cdf(F, )
end

# GHK likelihood
function GHKLikelihood(α₀::Float64, α₁::Float64, α₂::Float64, β::Array{Float64}, γ::Float64, ρ::Float64, T::Float64, X::Array{Float64}, Z::Array{Float64})
    return nothing
end

# Accept-Reject helper
function AR(α₀::Float64, α₁::Float64, α₂::Float64, β::Array{Float64}, γ::Float64, ρ::Float64, T::Float64, X::Array{Float64}, Z::Array{Float64}, ε₀::Array{Float64, 1}, ε₁::Array{Float64, 1}, ε₂::Array{Float64, 1})
    # Check cases
    if T == 1.0
        # Return result for T = 1
        return mean(α₀ + X' * β + Z[1] * γ .+ ε₀ .< 0)
    elseif T == 2.0
        # Return result for T = 2
        return mean((α₀ + X' * β + Z[1] * γ .- ε₀ .< 0) .* (α₁ + X' * β + Z[2] * γ .+ ε₁ .< 0))
    elseif T == 3.0
        # Return result for T = 3
        return mean((α₀ + X' * β + Z[1] * γ .- ε₀ .< 0) .* (α₁ + X' * β + Z[2] * γ .- ε₁ .> 0) .* (α₂ + X' * β + Z[3] * γ .+ ε₂ .< 0))
    elseif T == 4.0
        # Return result for T = 4
        return mean((α₀ + X' * β + Z[1] * γ .- ε₀ .< 0) .* (α₁ + X' * β + Z[2] * γ .- ε₁ .> 0) .* (α₂ + X' * β + Z[3] * γ .- ε₂ .> 0))
    else
        # Return error
        error("Please enter a valid number for T.")
    end
end

# Accept-Reject likelihood
function AcceptRejectLikelihood(α₀::Float64, α₁::Float64, α₂::Float64, β::Array{Float64}, γ::Float64, ρ::Float64, T::Float64, X::Array{Float64}, Z::Array{Float64}, ε₀::Array{Float64, 1}, ε₁::Array{Float64, 1}, ε₂::Array{Float64, 1})
    # Initialize results
    results = zeros(size(X, 1))

    # Iterate over points
    for i in 1:size(X, 1)
        # Calculate likelihood
        results[i] = AR(α₀, α₁, α₂, β, γ, ρ, T[i], X[i,:], Z[i,:], ε₀, ε₁, ε₂)
    end

    # Return value
    return results
end

# Likelihood switcher
function Likelihood(α₀::Float64, α₁::Float64, α₂::Float64, β::Array{Float64}, γ::Float64, ρ::Float64, T::Float64, X::Array{Float64}, Z::Array{Float64}, N::Int64=100, method::String="quadrature"; p₁::Array{Float64}=nothing, p₂::Array{Float64}=nothing)
    # Bad inputs
    if (method != "quadrature") && (method != "ghk") && (method != "accept-reject")
        # Return error
        error("Please input a valid integration method.")
    end

    # Check if quadrature
    if method == "quadrature"
        # Check inputs
        if p₁ === nothing || p₂ === nothing
            # Return error
            error("Please input Gaussian weights to use quadrature.")
        end

        # Return quadrature likelihood
        return QuadratureLikelihood(α₀, α₁, α₂, β, γ, ρ, T, X, Z, p₁, p₂)
    end
    # Set seed
    Random.seed!(0)

    # Generate random variables
    F = Normal()
    ε₀ = rand(F)
    ε₁ = 0.0
    ε₂ = 0.0

    # Check methods and return
    if method == "ghk"
        # Return GHK likelihood
        return GHKLikelihood(α₀, α₁, α₂, β, γ, ρ, T, X, Z)
        # TODO: Input draws
    elseif method == "accept-reject"
        # Return Accept-Reject likelihood
        return AcceptRejectLikelihood(α₀, α₁, α₂, β, γ, ρ, T, X, Z, ε₀, ε₁, ε₂)
    end
end

# Log-likelihood wrapper
function LogLikelihood(θ::Array{Float64}, T::Float64, X::Array{Float64}, Z::Array{Float64}, N::Int64=100, method::String="quadrature", p₁::Array{Float64}=nothing, p₂::Array{Float64}=nothing)
    # Unpack θ
    K = size(X, 2)
    α₀ = θ[1]
    α₁ = θ[2]
    α₂ = θ[3]
    β = θ[4:(K + 3)]
    γ = θ[K + 4]
    ρ = θ[K + 5]

    # Return value
    return sum(log.(Likelihood(α₀, α₁, α₂, β, γ, ρ, T, X, Z, N, method, p₁=p₁, p₂=p₂)))
end