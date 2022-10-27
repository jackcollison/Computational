# Load libraries
using LinearAlgebra

# Helper function for Λ estimation
function Λ(X::Array{Float64}, β::Array{Float64})
    # Return value
    exp(β' * X) / (1 + exp(β' * X))
end

# Log-likelihood function
function LogLikelihood(X::Array{Float64}, Y::Array{Float64, 1}, β::Array{Float64})
    # Return value
    sum([log(Λ(X[i,:], β)^Y[i]*(1 - Λ(X[i,:], β)^(1 - Y[i]))) for i in size(X, 1)])
end

# Score function
function Score(X::Array{Float64}, Y::Array{Float64, 1}, β::Array{Float64})
    # Return value
    sum([(Y[i] - Λ(X[i,:], β)) for i in 1:size(X, 1)] .* X, 1)
end

# Hessian function
function Hessian(X::Array{Float64}, β::Array{Float64})
    # Initialize Hessian
    H = zeros(size(X, 2), size(X, 2))

    # Populate Hessian values
    for i in 1:size(X, 1)
        H .+= Λ(X[i,:], β) * (1 - Λ(X[i,:], β)) * X[i,:]' * X[i,:]
    end

    # Return value
    H
end

# Numerical score function
function NumericalScore(β::Array{Float64, 1}, X::Array{Float64}, Y::Array{Float64}, ε::Float64=1e-6)
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
function NumericalHessian(β::Array{Float64, 1}, X::Array{Float64}, Y::Array{Float64}, ε::Float64=1e-6)
    # Initialize
    K = length(β)
    H = zeros(K, K)
    h = Matrix{Float64}(I, K, K) .* ε

    # Iterate
    for i = 1:K
        # Numerical Hessian
        H[i, :] = (NumericalScore(β .+ h, X, Y, ε) .- NumericalScore(β .- h, X, Y, ε)) ./ (2 * ε)
    end

    # Return value
    return H
end

# Step size function
function StepSize(f, g, β::Array{Float64}, d::Array{Float64}, α::Float64=1.0, ftol::Float64=1e-4, gtol::Float64=0.9, xtol::Float64=1e-6, lsmaxit::Int64=1000, θ...)
    # Initialize values
    fval = f(β, θ...)
    gval = g(β, θ...)
    ∂ϕ₀ = ∂f.T * d

    # Check condition
    if α <= 0 | ∂ϕ₀ > 0
        # Raise error
        error("Initialization step incorrect.")
    end

    # Set values
    c₁ = ftol
    c₂ = gtol
    ϕ₀ = fval
    αₗ = 0.0
    αᵣ = Inf

    # Initialize counter
    k = 0

    # Iterate while error is large
    while abs(αᵣ - αₗ) > xtol & k < lsmaxit
        # Increment counter
        k += 1

        # Update point
        β = x₀ + α * d
        fval = f(β, θ...)

        # Check Frank-Wolfe condition
        if fval >= ϕ₀ + α * c₁ * ∂ϕ₀
            # Update values
            αᵣ = α
            α = (αₗ + αᵣ) / 2.0
        else
            # Update values
            gval = g(β, θ...)
            ∂ϕα = gval.T * d

            # Check Frank-Wolfe condition
            if ∂ϕα >= c₂ * ∂ϕ₀
                # Return values
                return α
            else
                # Update and continue
                αₗ = α
                if isinf(αᵣ)
                    # Update value
                    α = 2 * αₗ
                else:
                    # Update value
                    α = (αₗ + αᵣ) / 2.0
                end
            end
        end
    end

    # Return values
    return α
end

# Solve with Newton
function Newton(f, g, H, β::Array{Float64}, ε::Float64=1e-12, verbose::Bool=false, θ...)
    # Initialize values
    i = 0
    β₀ = β
    fval = f(β₀, θ...)
    gval = g(β₀, θ...)
    Hval = H(β₀, θ...)
    error = Inf

    # Iterate while error is large
    while error >= ε
        # Update counter, value
        i += 1

        # Calculate direction, step size, values
        d = -inv(Hval) * gval
        α = StepSize(f, g, β₀, d, 1.0, θ...)
        β₁ = β₀ + α * d

        # Update error, coefficients
        error = maximum(abs.(β₁ - β₀))
        β₀ = β₁

        # Print statement
        if verbose
            print("Newton's method is at iteration i = ", i, " with error ε = ", error)
        end
    end

    # Print statement
    print("Newton's method converged in i = ", i, " iterations with error ε = ", error)

    # Return value
    return β₁
end