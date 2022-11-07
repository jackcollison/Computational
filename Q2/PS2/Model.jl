# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: November, 2022

# Import packages
using Distributions

# Import utilities
include("Utilities.jl")

# Quadrature likelihood helper
function LQ(w::Array{Float64}, u::Array{Float64}, α₀::Float64, α₁::Float64, α₂::Float64, β::Float64, γ::Float64, ρ::Float64, T::Float64, X::Array{Float64}, Z::Array{Float64})
    # Initialize values
    F = Normal()
    σ₀ = 1 / (1 - ρ)^2

    # Check cases
    if T == 1.0
        # Return result for T = 1
        return pdf(F, (α₀ + X' * β + Z[1] * γ) / σ₀)
    elseif T == 2.0
        # Return result for T = 2
        f(ε₀) = cdf(F, -α₁ - X' * β - Z[2] * γ - ρ * ε₀) * pdf(F, ε₀ / σ₀) / σ₀
        return Quadrature(f, w, u, a=[-α₀ - X' * β - Z[2]* γ])
    elseif T == 3.0
        # Return result for T = 3
        f(ε₀, ε₁) = cdf(F, -α₂ - X' * β - Z[3] * γ - ρ * ε₁) * pdf(F, ε₁ - ρ * ε₀) * pdf(F, ε₀ / σ₀) / σ₀
        return Quadrature(f, w, u, a=[α₀ + X' * β + Z[3] * γ, α₁ + X' * β + Z[3] * γ])
    elseif T == 4.0
        # Return result for T = 4
        f(ε₀, ε₁) = cdf(F, α₂ + X' * β + Z[4] * γ - ρ * ε₁) * pdf(F, ε₁ - ρ * ε₀) * pdf(F, ε₀ / σ₀) / σ₀
        return Quadrature(f, w, u, a=[α₀ + X' * β + Z[4] * γ, α₁ + X' * β + Z[4] * γ])
    else
        # Return error
        error("Please enter a valid number for T.")
    end
end

# Quadrature likelihood
function QuadratureLikelihood(α₀::Float64, α₁::Float64, α₂::Float64, β::Float64, γ::Float64, ρ::Float64, T::Float64, X::Array{Float64}, Z::Array{Float64})
    # Initialize results
    results = zeros(size(X, 1))

    # Read data
    p₁ = readdlm("/Users/jackcollison/Desktop/Wisconsin/Coursework/Second Year/Computational/Q2/PS2/KPU_d1_l20.csv", ',', Float64)
    p₂ = readdlm("/Users/jackcollison/Desktop/Wisconsin/Coursework/Second Year/Computational/Q2/PS2/KPU_d2_l20.csv", ',', Float64)

    # Initialize weights and points
    w₁ = p₁[:,2]
    u₁ = points[:,1]
    w₂ = p₁[:,3]
    u₂ = points[:,1:2]

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
end

