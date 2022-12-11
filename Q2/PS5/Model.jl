# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: December, 2022

##################################################
################# INITIALIZATION #################
##################################################

# Import required packages
using Parameters, Tables, LinearAlgebra, CSV, Printf, DataFrames, Statistics, Random

# Create primitives
@with_kw struct Primitives
    # Parameters
    β::Float64 = 1.0 / 1.05
    α::Float64 = 0.06
    a::Float64 = 40.0
    b::Float64 = 10.0
    q̄::Array{Float64} = collect(0.0:5.0:45.0)
    Q̄::Matrix{Tuple{Float64, Float64}} = [(x, y) for x in q̄, y in q̄]
    n::Int64 = length(q̄)
end

########################################################
################### HELPER FUNCTIONS ###################
########################################################

# Best response function
function Rᶜ(a::Float64, q̄ᵢ::Float64, q₋ᵢ::Float64)
    # Return value
    return max(min(q̄ᵢ, (a - q₋ᵢ) / 2), 0.0)
end

# Best reply function for Cournot
function qᶜ(p::Primitives, q̄ᵢ::Float64, q̄₋ᵢ::Float64)
    # Unpack primitives
    @unpack a, b = p

    # Pre-compute values
    qⁱ = a / 3
    Rᵢ = Rᶜ(a, q̄ᵢ, q̄₋ᵢ)
    R₋ᵢ = Rᶜ(a, q̄₋ᵢ, Rᵢ)

    # Check conditions
    if q̄ᵢ >= qⁱ && q̄₋ᵢ >= qⁱ
        # Return interior solution
        return qⁱ
    elseif Rᵢ <= q̄ᵢ && R₋ᵢ >= q̄₋ᵢ
        # Return partially constrained solution
        return Rᵢ
    else
        # Return constrained solution
        return q̄ᵢ
    end
end

# Nash equlibrium Cournot profits
function Πᶜ(p::Primitives, q̄ᵢ::Float64, q̄₋ᵢ::Float64)
    # Unpack primitives
    @unpack a, b = p

    # Return value
    return (a - (qᶜ(p, q̄ᵢ, q̄₋ᵢ) + qᶜ(p, q̄₋ᵢ, q̄ᵢ))) / b * qᶜ(p, q̄ᵢ, q̄₋ᵢ)
end

# Capacity evolution
function 𝐏(p::Primitives, i::Int64, Δq::Int64, x::Float64, δ::Float64)
    # Upack primitives
    @unpack α, n = p

    # Check cases
    if Δq == 1
        # Return improvement
        return ((1 - δ) * α * x + δ * α * x * (i == 1)) / (1 + α * x) * (i < n)
    elseif Δq == 0
        # Return no improvement
        return (1 - δ + δ * α * x + (δ - δ * α * x) * (i == 1) + (α * x - δ * α * x) * (i == n)) / (1 + α * x)
    elseif Δq == -1
        # Return depreciation
        return (δ / (1 + α * x)) * (i > 1)
    end

    # Return value
    return 0.0
end

########################################################
################# SOLVE VALUE FUNCTIONS ################
########################################################

# Continuation value
function 𝐖(p::Primitives, i::Int64, j::Int64, V::Matrix{Float64}, x::Float64, δ::Float64)
    # Initialization
    @unpack n = p
    W = 0.0

    # Iterate over states
    for Δ in collect(ifelse(j > 1, -1, 0):1:ifelse(j < n, 1, 0))
        # Update value
        W += V[i, j + Δ] * 𝐏(p, j, Δ, x, δ)
    end

    # Return value
    return W
end

# Closed form best response
function 𝐱(p::Primitives, i::Int64, j::Int64, W::Matrix{Float64}, δ::Float64)
    # Unpack primitives
    @unpack β, α = p

    # Calculate values
    Wᵢ = W[i, j]

    # Check cases
    if i > 1 && i < n
        # Update values
        Wᵢ₋₁ = min(W[i - 1, j], Wᵢ)
        Wᵢ₊₁ = max(W[i + 1, j], Wᵢ)

        # Return value
        return max(0.0, (-1.0 + sqrt(α * β * ((1 - δ) * (Wᵢ₊₁ - Wᵢ) + δ * (Wᵢ - Wᵢ₋₁)))) / α)
    elseif i == 1
        # Update values
        Wᵢ₊₁ = max(W[i + 1, j], Wᵢ)

        # Return value
        return max(0.0, (-1.0 + sqrt(α * β * (Wᵢ₊₁ - Wᵢ))) / α)
    elseif i == n
        # Update values
        Wᵢ₋₁ = min(W[i - 1, j], Wᵢ)

        # Return value
        return max(0.0, (-1.0 + sqrt(β * α * δ * (Wᵢ - Wᵢ₋₁))) / α)
    end

    # Return error
    error("Index is out of range.")
end

# Solve model
function SolveModel(p::Primitives, V::Matrix{Float64}, δ::Float64; tol::Float64=1e-4, verbose::Bool=false)
    # Intiailize variables
    @unpack q̄, n, β = p
    ε = Inf
    k = 0
    Π₀ = V

    # Initialize policy and value functions
    W = V
    X = [𝐱(p, i, j, W, δ) for i in 1:n, j in 1:n]
    V = V - X + β * V

    # Print statement
    if verbose
        println("Firm Iteration    Error")
        println("-------------------------------")
    end

    # Iterate while error is large
    while ε > tol
        # Increment
        k += 1

        # Update values
        W = [𝐖(p, i, j, V, X[j, i], δ) for i in 1:n, j in 1:n]
        Xp = [𝐱(p, i, j, W, δ) for i in 1:n, j in 1:n]
        Vp = Π₀ - Xp

        # Iterate over states
        for i in 1:n
            for Δ in collect(ifelse(i > 1, -1, 0):1:ifelse(i < n, 1, 0))
                # Increment value
                Vp[i,:] += β * [W[i + Δ, j] * 𝐏(p, i, Δ, Xp[i, j], δ) for j in 1:n]
            end
        end

        # Update error
        ε = max(maximum(abs.(Xp - X)), maximum(abs.(Vp - V)))

        # Update values
        X = Xp
        V = Vp

        # Print statement
        if verbose
            @printf "i = %-13d ε = %-12.3g\n" k ε
        end
    end

    # Print statement
    if verbose
        println("\n******************************************************\n")
        @printf "Completed in %d iterations with error ε = %.3g\n" k ε
        println("\n******************************************************\n")
    end 

    # Return values
    return X, V
end

########################################################
################## MARKET SIMULATIONS ##################
########################################################

# Simulate markets
function Simulate(p::Primitives, X::Array{Float64}; T::Int64=25, S::Int64=10000)
    # Set seed for reproducibility
    Random.seed!(0)

    # Unpack primitives
    @unpack n = p

    # Transition probabilities
    D = zeros((n, n))
    P = zeros((n, n))

    # Iterate over states
    for i in 1:n, j in 1:n
        # Find transition
        Δ = Int64(j - i)
        P[i, j] = 𝐏(p, i, Δ, X[i, i], δ)
    end

    # Cumulative distribution
    P = cumsum(P, dims=2)

    # Iterate over simulations
    for s in 1:S
        # Initial state
        ω₁, ω₂ = 1, 1

        # Iterate over time periods
        for t in 1:T
            # Find relevant transitions and draws
            P₁, P₂ = P[ω₁,:], P[ω₂,:]
            ε₁, ε₂ = rand(), rand()

            # Update states
            ω₁ = findall(P₁ .>= ε₁)[1]
            ω₂ = findall(P₂ .>= ε₂)[1]
        end

        # Update distribution
        D[ω₁, ω₂] += 1.0 / S
    end

    # Return value
    return D
end