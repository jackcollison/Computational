# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: October, 2022

##################################################
################# INITIALIZATION #################
##################################################

# Import required packages
using Parameters, LinearAlgebra

# Structure for primitives
@with_kw struct Primitives
    # Primitives
    β::Float64 = 0.8
    A::Float64 = 1 / 200.0
    θ::Float64 = 0.64
    S::Array{Float64, 1} = [3.98e-4, 3.58, 6.82, 12.18, 18.79]
    ns::Int64 = length(S)
    F::Array{Float64, 2} = [0.6598 0.2600 0.0416 0.0331 0.0055; 
                            0.1997 0.7201 0.0420 0.0326 0.0056; 
                            0.2000 0.2000 0.5555 0.0344 0.0101;
                            0.2000 0.2000 0.2502 0.3397 0.0101; 
                            0.2000 0.2000 0.2500 0.3400 0.0100] 
    ν::Array{Float64, 1} = [0.37, 0.4631, 0.1102, 0.0504, 0.0063]
    cᵉ::Float64 = 5.0
    γ::Float64 = Base.MathConstants.γ
    ε::Float64 = 1e-4
    εₘ::Float64 = 1e-4
end

# Structure for results
mutable struct Results
    # Hyperparameters
    cᶠ::Float64
    α::Float64

    # Firm decisions
    p::Float64
    N::Array{Float64, 1}
    π::Array{Float64, 1}
    x::Array{Float64, 1}
    W::Array{Float64, 1}

    # Firm distribution
    M::Float64
    μ::Array{Float64, 1}

    # Equilibrium
    Π::Float64
    Lᵈ::Float64
    Lˢ::Float64
end

# Initialize results structure
function Initialize(cᶠ::Float64 = 10.0, α::Float64 = -1.0)
    # Relevant primitives
    @unpack ns = Primitives()

    # Firm decisions
    p = 1.0
    N = zeros(ns)
    π = zeros(ns)
    x = fill(1, ns)
    W = zeros(ns)

    # Firm distribution
    M = 1.0
    μ = ones(ns)

    # Equilibrium
    Π = 0.0
    Lᵈ = 0.0
    Lˢ = 0.0

    # Return results
    return Results(cᶠ, α, p, N, π, x, W, M, μ, Π, Lᵈ, Lˢ)
end

########################################################
##################### FIRM PROBLEM #####################
########################################################

# Bellman for exit decisions
function Bellman(P::Primitives, R::Results)
    # Initialize functions
    @unpack ns, γ = P
    xp = zeros(ns)
    Wp = zeros(ns)

    # Iterate over state space
    for i_s = 1:ns
        # Value from exiting
        Wᵉ = R.π[i_s]

        # Value from staying
        Wˢ = R.π[i_s]
        for i_sp = 1:P.ns
            Wˢ += P.β * P.F[i_s, i_sp] * R.W[i_sp]
        end

        # Check if shock
        if R.α <= 0
            # Firm value and policy without shock
            xp[i_s] = (Wˢ < Wᵉ)
            Wp[i_s] = max(Wˢ, Wᵉ)
        else
            # Log-sum-exp calculation
            m = R.α * max(Wˢ, Wᵉ)
            xp[i_s] = exp(R.α * Wᵉ - m) / (exp.(R.α * Wˢ - m) + exp(R.α * Wᵉ - m))
            Wp[i_s] = (γ / R.α) + (1.0 / R.α) * (m + log(exp(R.α * Wˢ - m) + exp(R.α * Wᵉ - m)))
        end
    end

    # Return values
    return xp, Wp
end

# Solve firm problem
function SolveFirmProblem(R::Results, verbose::Bool=false)
    # Primitives
    P = Primitives()

    # Labor and profit of the firm
    R.N = max.((R.p .* P.S .* P.θ).^(1.0 / (1 - P.θ)), 0)
    R.π   = R.p .* P.S .* (R.N).^P.θ .- R.N .- R.p .* R.cᶠ

    # Initialize counter, error
    i = 0
    error = Inf

    # Iterate while error is large
    while error > P.ε
        # Increment counter
        i += 1

        # Compute value and policy functions
        xp, Wp = Bellman(P, R)
        error = maximum(abs.(R.x .- xp)) + maximum(abs.(R.W .- Wp))
        
        # Update value and policy
        R.x = xp
        R.W = Wp
        
        # Print statement
        if verbose
            println("Firm Iteration = ", i, " with error ε = ", error, "\n")
        end
    end
end

########################################################
################# MARKET CLEARING PRICE ################
########################################################

# Calculate equilibrium price
function UpdatePrice(R::Results, verbose::Bool=false)
    # Unpack tolerance
    @unpack ν, cᵉ, εₘ = Primitives()

    # Initial bounds and midpoint for price search using bisection method.
    pₗ = 0.0
    pₕ = 10.0
    p̄ = (pₕ + pₗ) / 2

    # Loop variables
    i = 0
    error = Inf

    # Iterate while excess is large
    while abs(error) > εₘ
        # Increment counter
        i += 1

        # Update price and solve firm problem
        R.p = p̄
        SolveFirmProblem(R)

        # Update error
        error  = sum(R.W .* ν) / p̄ - cᵉ

        # Check excess
        if error < 0
            # Update lower bound
            pₗ = p̄
        else
            # Update upper bound
            pₕ = p̄
        end

        # Update guess
        p̄ = (pₕ + pₗ) / 2

        # Print statement
        if verbose
            println("Price Iteration = ", i, " with error ε = ", error, "\n")
        end
    end

    # Update price
    R.p = p̄
end

########################################################
################## SOLVE DISTRIBUTION ##################
########################################################

# Update μ distribution
function μUpdate(R::Results, P::Primitives)
    # Initialize
    μp = zeros(P.ns)

    # Iterate over states today
    for i_s = 1:P.ns
        # Iterate over states tomorrow
        for i_sp = 1:P.ns
            # Update μ distribution
            μp[i_sp] += (1 - R.x[i_s]) * P.F[i_s, i_sp] * (R.μ[i_s] + R.M * P.ν[i_s])
        end
    end

    # Return value
    return μp
end

# Solve for steady-state distribution
function Solveμ(R::Results, verbose::Bool=false)
    # Initialize
    P = Primitives()
    error = Inf
    i = 0

    # Iterate while error is large
    while error > 0
        # Increment counter
        i += 1

        # Find error
        μp = μUpdate(R, P) 
        error = maximum(abs.(R.μ .- μp))

        # Update results
        R.μ = μp

        # Print statement
        if verbose
            println("Distribution Iteration = ", i, " with error ε = ", error, "\n")
        end
    end
end

########################################################
##################### MODEL SOLVER #####################
########################################################

# Market clearing conditions
function ClearMarket(R::Results, verbose)
    # Initialize primitives
    P = Primitives()

    # Solve for stationary distribution
    Solveμ(R, verbose)

    # Clear market
    R.Π = sum(R.π .* R.μ) + R.M * sum(R.π .* P.ν)
    R.Lᵈ = sum(R.N .* R.μ) + R.M * sum(R.N .* P.ν)
    R.Lˢ = (1.0 / P.A) - R.Π
end 

# Solves for mass of new entrants using bisection method.
function SolveM(R::Results, verbose::Bool=false)
    # Initial bounds and midpoint for m
    P = Primitives()
    Mₗ = 0.0
    Mₕ = 10.0
    R.M = (Mₕ + Mₗ) / 2

    # Initialize error
    error = Inf
    i = 0

    # Iterate while error is large
    while abs(error) > P.εₘ
        # Increment counter
        i += 1

        # Market clearing condition
        ClearMarket(R, verbose)
        error = R.Lᵈ - R.Lˢ

        # Update bounds
        if error < 0
            Mₗ = R.M
        else
            Mₕ = R.M
        end

        # Update midpoint
        R.M = (Mₕ + Mₗ) / 2

        # Print statement
        if verbose
            println("Mass Iteration = ", i, " with error ε = ", error, "\n")
        end
    end
end

# Functionality to solve model
function SolveModel(R::Results, verbose::Bool=false)
    # Solve model
    UpdatePrice(R, verbose)
    SolveM(R, verbose)
end
