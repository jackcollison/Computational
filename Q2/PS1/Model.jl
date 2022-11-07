# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: November, 2022

# Load libraries
using LinearAlgebra, Printf

# Helper function for Λ estimation
function Λ(β::Array{Float64}, X::Array{Float64})
    # Return value
    exp(β' * X) / (1 + exp(β' * X))
end

# Log-likelihood function
function LogLikelihood(β::Array{Float64}, X::Matrix{Float64}, Y::Matrix{Float64})
    # Initialize
    N = size(X, 1)
    L = 0

    # Loop over points
    for i = 1:N 
        # Update log-likelihood
        Λᵢ = Λ(β, X[i,:])
        L += log((Λᵢ^Y[i]) * ((1 - Λᵢ)^(1 - Y[i])))
    end

    # Return value
    return L
end

# Score function
function Score(β::Array{Float64}, X::Matrix{Float64}, Y::Matrix{Float64})
    # Initialize
    N = size(X, 1)
    g = zeros(K)

    # Populate score elements
    for i = 1:N 
        # Update score element
        g += (Y[i] - Λ(β, X[i,:])) .* X[i,:]
    end

    # Return value
    return g  
end

# Hessian function
function Hessian(β::Array{Float64}, X::Matrix{Float64}, _...)
    # Initialize Hessian
    H = zeros(size(X, 2), size(X, 2))

    # Populate Hessian elements
    for i in 1:size(X, 1)
        # Update Hessian element
        H .+= -Λ(β, X[i,:]) * (1 - Λ(β, X[i,:])) * X[i,:] * X[i,:]'
    end

    # Return value
    return H
end

# Numerical score function
function NumericalScore(β::Array{Float64}, X::Matrix{Float64}, Y::Matrix{Float64}; ε::Float64=1e-6)
    # Initialize
    K = length(β)
    g = zeros(K)
    h = Matrix{Float64}(I, K, K) .* ε

    # Iterate
    for i = 1:K
        # Numerical score
        g[i] = (LogLikelihood(β .+ h[i,:], X, Y) - LogLikelihood(β .- h[i,:], X, Y)) / (2 * ε)
    end

    # Return value
    return g
end

# Numerical Hessian function
function NumericalHessian(β::Array{Float64, 1}, X::Matrix{Float64}, Y::Matrix{Float64}; ε::Float64=1e-6)
    # Initialize
    K = length(β)
    H = zeros(K, K)
    h = Array{Float64}(I, K, K) .* ε

    # Iterate
    for i = 1:K
        # Numerical Hessian
        H[i, :] = (NumericalScore(β .+ h[:,i], X, Y, ε=ε) .- NumericalScore(β .- h[:,i], X, Y, ε=ε)) ./ (2 * ε)
    end

    # Return value
    return H
end

# Step size function
function StepSize(f, g, β::Array{Float64}, d::Array{Float64}, θ...; α::Float64=1.0, ftol::Float64=1e-4, gtol::Float64=0.9, xtol::Float64=1e-6, lsmaxit::Int64=1000)
    # Initialize values
    β₀ = β
    fval = -f(β, θ...)
    gval = -g(β, θ...)
    ∂ϕ₀ = sum(gval .* d)

    # Check direction
    if α <= 0 || ∂ϕ₀ > 0
        # Raise error
        error("Initialization of step is incorrect.")
    end

    # Set values
    c₁ = ftol
    c₂ = gtol
    ϕ₀ = -f(β, θ...)
    αₗ = 0.0
    αᵣ = Inf
    k = 0

    # Iterate while error is large
    while abs(αᵣ - αₗ) > xtol && k < lsmaxit
        # Update values
        k += 1
        β = β₀ + α * d
        fval = -f(β, θ...)

        # Check Wolfe condition
        if fval >= ϕ₀ + α * c₁ * ∂ϕ₀
            # Update values
            αᵣ = α
            α = (αᵣ + αₗ) / 2.0
        else
            # Update values
            gval = -g(β, θ...)
            ∂ϕα = sum(gval .* d)

            # Armijo backtracking conditions
            if ∂ϕα >= c₂ * ∂ϕ₀
                # Return value
                return α
            else
                # Update values
                αₗ = α
                if isinf(αᵣ)
                    # Increase α
                    α = 2.0 * αₗ
                else
                    # Split α
                    α = (αᵣ + αₗ) / 2.0
                end
            end
        end
    end

    # Return value
    return α
end

# Solve with Newton
function NewtonMethod(f, g, H, β::Array{Float64}, θ...; ε::Float64=1e-12, verbose::Bool=true)
    # Initialize values
    i = 0
    error = Inf

    # Iterate while error is large
    while error >= ε
        # Update counter, value
        i += 1

        # Take Newton step
        gval = g(β, θ...)
        Hval = H(β, θ...)
        d = -inv(Hval) * gval
        α = StepSize(f, g, β, d, θ...)
        β₁ = β + α * d

        # Update error, coefficients
        error = maximum(abs.(β₁ - β))
        β = β₁

        # Print statement
        if verbose
            @printf "Newton's method is at iteration i = %i with error ε = %.4g and step size α = %.4g\n" i error α
        end
    end

    # Print statement
    println("\n************************************************************************")
    @printf "Newton's method converged in %i iterations with error ε = %.4g\n" i error
    println("************************************************************************")

    # Return value
    return β, f(β, θ...)
end

# Gradient of negative log-likelihood
function g!(G::Array{Float64, 1}, β::Array{Float64, 1})
    # Store value
    G .= -Score(β, X, Y)
end

# Hessian of negative log-likelihood
function h!(H::Array{Float64, 2}, β::Array{Float64, 1})
    # Store value
    H .= -Hessian(β, X)
end