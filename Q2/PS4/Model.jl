# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: Nocember, 2022

##################################################
################# INITIALIZATION #################
##################################################

# Import required packages
using Parameters, Tables, LinearAlgebra, CSV, Printf, DataFrames, Statistics

# Create primitives
@with_kw struct Primitives
    # Parameters
    α::Float64 = 2.0
    β::Float64 = 0.99
    n::Int64 = 36
    pᵣ::Float64 = 4.0
    pₛ::Float64 = 1.0
    ī::Float64 = 8.0

    # State space and transition matrices
    S = CSV.File("/Users/jackcollison/Desktop/Wisconsin/Coursework/Second Year/Computational/Q2/PS4/PS4_state_space.csv") |> Tables.matrix
    Sₛ = S[:,]
    Iₛ = S[:,3]
    Cₛ = S[:,4]
    Pₛ = S[:,5]

    # Transition matrices
    Π₀ = CSV.File("/Users/jackcollison/Desktop/Wisconsin/Coursework/Second Year/Computational/Q2/PS4/PS4_transition_a0.csv") |> Tables.matrix
    Π₁ = CSV.File("/Users/jackcollison/Desktop/Wisconsin/Coursework/Second Year/Computational/Q2/PS4/PS4_transition_a1.csv") |> Tables.matrix
    F₀ = Π₀[:,3:end]
    F₁ = Π₁[:,3:end]

    # Simulated data
    data = DataFrame(CSV.File("/Users/jackcollison/Desktop/Wisconsin/Coursework/Second Year/Computational/Q2/PS4/PS4_simdata.csv"))
    CP = combine(groupby(data, :state_id), :choice .=> mean .=> :P)
    ID = data.state_id
    Y = data.choice
    P₀ = min.(max.(1 .- CP.P, 0.001), 0.999)
    P₁ = min.(max.(CP.P, 0.001), 0.999)
end

########################################################
################### VALUE FUNCTIONS ####################
########################################################

# Helper function
function GetU(p::Primitives, λ::Float64)
    # Unpack primitives
    @unpack Iₛ, Cₛ, Pₛ, α = p

    # Solve intermediate values
    U₀ = α .* Cₛ .* (Iₛ .> 0) .+ λ .* (Iₛ .== 0) .* (Cₛ .> 0)
    U₁ = α .* Cₛ .- Pₛ

    # Return values
    return U₀, U₁
end

# Choice-specific value function
function V(p::Primitives, λ::Float64, V̄::Array{Float64})
    # Unpack primitives
    @unpack F₀, F₁, β = p

    # Solve intermediate values
    U = GetU(p, λ)

    # Concatenate results
    V₀ = U[1] .+ β * F₀ * V̄
    V₁ = U[2] .+ β * F₁ * V̄

    # Return values
    return hcat(V₀, V₁)
end

# Expected value function
function EV(p::Primitives, λ::Float64, V̄::Array{Float64})
    # Return value
    return log.(sum(exp.(V(p, λ, V̄)), dims=2)) .+ MathConstants.eulergamma
end

# Conditional choice mapping
function CCP(p::Primitives, λ::Float64, 𝐏::Array{Float64})
    # Unpack primitives
    @unpack Iₛ, Cₛ, Pₛ, F₀, F₁, α, β = p
    Iₙ = 1.0 * Matrix(I, size(𝐏, 1), size(𝐏, 1))

    # Solve for U₀, U₁
    U = GetU(p, λ)

    # Find errors
    e₀ = MathConstants.eulergamma .- log.(1 .- 𝐏)
    e₁ = MathConstants.eulergamma .- log.(𝐏)

    # Find distribution
    F = F₀ .* (1 .- 𝐏) + F₁ .* 𝐏

    # Expected values
    Eᵤ = (1 .- 𝐏) .* (U[1] + e₀) .+ 𝐏 .* (U[2] .+ e₁)
    V̄ = inv(Iₙ .- β .* F) * Eᵤ

    # Compute value
    Ṽ = V(p, λ, V̄)

    # Return value
    return V̄, 1.0 ./ sum(1.0 .+ exp.(-(Ṽ[:,2] - Ṽ[:,1])), dims=2)
end

########################################################
################# SOLVE VALUE FUNCTIONS ################
########################################################

# Expected value function contraction
function SolveEV(p::Primitives, λ::Float64, V̄::Array{Float64}; tol::Float64=1e-14, verbose::Bool=false)
    # Initialize
    error = Inf
    i = 0

    # Iterate while error is large
    while error > tol
        # Increment counter
        i += 1

        # Calculate expected value and update
        V̄_next = EV(p, λ, V̄)
        error = maximum(abs.(V̄_next .- V̄))
        V̄ = V̄_next

        # Print statement
        if verbose
            println("Expected value at iteration i = ", i, " with error ε = ", error)
        end
    end

    # Return value
    return V̄
end

# Conditional choice probability problem
function SolveCCP(p::Primitives, λ::Float64, 𝐏::Array{Float64}; tol::Float64=1e-14, verbose::Bool=false)
    # Initialize
    V̄ = nothing
    error = Inf
    i = 0

    # Iterate while error is large
    while error > tol
        # Increment counter
        i += 1

        # Calculate conditional choice probability and update
        results = CCP(p, λ, 𝐏)
        V̄ = results[1]
        P_next = results[2]
        error = maximum(abs.(P_next .- 𝐏))
        𝐏 = P_next

        # Print statement
        if verbose
            println("Conditional choice probability at iteration i = ", i, " with error ε = ", error)
        end
    end

    # Return value
    return V̄, 𝐏
end

########################################################
################ LIKELIHOOD FUNCTIONALITY ##############
########################################################

# Nested fixed point algorithm
function NXFP(p::Primitives, λ::Array{Float64}, 𝐏::Array{Float64}; tol::Float64=1e-14, verbose::Bool=false)
    # Unpack primitives
    @unpack Y, ID = p

    # Compute policy function
    𝐏ᵒ = SolveCCP(p, λ[1], 𝐏; tol=tol, verbose=verbose)[2]
    𝐏ˢ = [𝐏ᵒ[s + 1] for s in ID]

    # Return log-likelihood
    return -sum(Y .* log.(𝐏ˢ) .+ (1.0 .- Y) .* log.(1.0 .- 𝐏ˢ))
end