# Load libraries
using LinearAlgebra

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
        β₁ = β - inv(Hval) * gval

        # Update error, coefficients
        error = maximum(abs.(β₁ - β))
        β = β₁

        # Print statement
        if verbose
            println("Newton's method is at iteration i = ", i, " with error ε = ", error)
        end
    end

    # Print statement
    println("\n************************************************************************************")
    println("Newton's method converged in i = ", i, " iterations with error ε = ", error)
    println("************************************************************************************")

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