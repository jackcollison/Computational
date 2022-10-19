# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: October, 2022

##################################################
################# INITIALIZATION #################
##################################################

# Import required packages
using Parameters, Random, Distributions

# Structure for results
@with_kw struct Primitives 
    # Value and policy functions
    β::Float64 = 0.99
    α::Float64 = 0.36
    δ::Float64 = 0.025
    λ::Float64 = 0.5
    N::Int64 = 5000
    T::Int64 = 11000
    B::Int64 = 1000
    Kˢˢ::Float64 = 11.55
    tol_v::Float64 = 1e-4
    tol_c::Float64 = 1e-4
    tol_r::Float64 = 0.99
    maxit::Int64 = 10000
end

# Structure for grids
@with_kw struct Grids
    # Individual capital grid
    nk::Int64 = 21
    k::Array{Float64,1} = range(0.0001, stop = 15.0, length = nk)

    # Aggregate capital grid
    nK::Int64 = 11
    K::Array{Float64,1} = range(11.0, stop = 15.0, length = nK)

    # Labor productivity grid
    ε::Array{Float64,1} = [0.3271, 0.0]
    ne::Int64 = size(ε, 1)

    # Aggregate technology shock grid
    Z::Array{Float64,1} = [1.01, 0.99]
    nz::Int64 = size(Z, 1)
end

# Structure for shocks
@with_kw struct Shocks
    # Transition matrix parameters
    dᵍ::Float64 = 8.0
    dᵇ::Float64 = 8.0 
    dᵘᵍ::Float64 = 1.5
    dᵘᵇ::Float64 = 2.5
    uᵍ::Float64 = 0.04
    uᵇ::Float64 = 0.1 

    # Transition probabilities for aggregate states
    Πᵍᵍ::Float64 = (dᵍ - 1.0) / dᵍ
    Πᵍᵇ::Float64 = 1.0 - (dᵍ - 1.0) / dᵍ
    Πᵇᵍ::Float64 = 1.0 - (dᵇ - 1.0) / dᵇ
    Πᵇᵇ::Float64 = (dᵇ - 1.0) / dᵇ

    # Transition probabilities for aggregate states and staying unemployed
    πᵍᵍ⁰⁰::Float64 = (dᵘᵍ - 1.0) / dᵘᵍ
    πᵇᵇ⁰⁰::Float64 = (dᵘᵇ - 1.0) / dᵘᵇ
    πᵍᵇ⁰⁰::Float64 = 1.25 * πᵇᵇ⁰⁰
    πᵇᵍ⁰⁰::Float64 = 0.75 * πᵍᵍ⁰⁰

    # Transition probabilities for aggregate states and becoming employed
    πᵍᵍ¹⁰::Float64 = (uᵍ - uᵍ * πᵍᵍ⁰⁰) / (1.0 - uᵍ)
    πᵇᵇ¹⁰::Float64 = (uᵇ - uᵇ * πᵇᵇ⁰⁰) / (1.0 - uᵇ)
    πᵍᵇ¹⁰::Float64 = (uᵇ - uᵍ * πᵍᵇ⁰⁰) / (1.0 - uᵍ)
    πᵇᵍ¹⁰::Float64 = (uᵍ - uᵇ * πᵇᵍ⁰⁰) / (1.0 - uᵇ)

    # Transition probabilities for aggregate states and becoming unemployed
    πᵍᵍ⁰¹::Float64 = 1.0 - (dᵘᵍ - 1.0) / dᵘᵍ
    πᵇᵇ⁰¹::Float64 = 1.0 - (dᵘᵇ - 1.0) / dᵘᵇ
    πᵍᵇ⁰¹::Float64 = 1.0 - 1.25 * πᵇᵇ⁰⁰
    πᵇᵍ⁰¹::Float64 = 1.0 - 0.75 * πᵍᵍ⁰⁰

    # Transition probabilities for aggregate states and staying employed
    πᵍᵍ¹¹::Float64 = 1.0 - (uᵍ - uᵍ * πᵍᵍ⁰⁰) / (1.0 - uᵍ)
    πᵇᵇ¹¹::Float64 = 1.0 - (uᵇ - uᵇ * πᵇᵇ⁰⁰) / (1.0 - uᵇ)
    πᵍᵇ¹¹::Float64 = 1.0 - (uᵇ - uᵍ * πᵍᵇ⁰⁰) / (1.0 - uᵍ)
    πᵇᵍ¹¹::Float64 = 1.0 - (uᵍ - uᵇ * πᵇᵍ⁰⁰) / (1.0 - uᵇ)

    # Markov transition matrix
    Mᵍᵍ::Array{Float64,2} = reshape([πᵍᵍ¹¹ πᵍᵍ⁰¹ πᵍᵍ¹⁰ πᵍᵍ⁰⁰], (2, 2))
    Mᵍᵇ::Array{Float64,2} = reshape([πᵍᵇ¹¹ πᵍᵇ⁰¹ πᵍᵇ¹⁰ πᵍᵇ⁰⁰], (2, 2))
    Mᵇᵍ::Array{Float64,2} = reshape([πᵇᵍ¹¹ πᵇᵍ⁰¹ πᵇᵍ¹⁰ πᵇᵍ⁰⁰], (2, 2))
    Mᵇᵇ::Array{Float64,2} = reshape([πᵇᵇ¹¹ πᵇᵇ⁰¹ πᵇᵇ¹⁰ πᵇᵇ⁰⁰], (2, 2))
    M::Array{Float64,2} = [Πᵍᵍ * Mᵍᵍ Πᵍᵇ * Mᵍᵇ ; Πᵇᵍ * Mᵇᵍ Πᵇᵇ * Mᵇᵇ]
end

# Structure for results
mutable struct Results
    # Value and policy functions
    policy_func::Array{Float64,4}
    value_func::Array{Float64,4}

    # Regression coefficients
    a₀::Float64
    a₁::Float64
    b₀::Float64
    b₁::Float64

    # Regression R²
    R²::Float64
end

########################################################
################## GENERATE PSEUDOPANEL ################
########################################################

# Update state
function UpdateState(M::Array{Float64, 2}, sᵢₜ::Float64, εᵢₜ::Float64)
    # Update transitions
    Π¹¹ = M[1, 1]
    Π⁰⁰ = M[2, 2]

    # Check idiosyncratic states
    if sᵢₜ == 1 && εᵢₜ < Π¹¹
        return 1
    elseif sᵢₜ  == 1 && εᵢₜ > Π¹¹
        return 2
    elseif sᵢₜ  == 2 && εᵢₜ < Π⁰⁰
        return 2
    elseif sᵢₜ  == 2 && εᵢₜ > Π⁰⁰
        return 1
    end
end

# Draw random shocks
function DrawShocks(S::Shocks, N::Int64, T::Int64)
    # Unpack primitives
    @unpack Πᵍᵍ, Πᵇᵇ, Mᵍᵍ, Mᵍᵇ, Mᵇᵍ, Mᵇᵇ = S

    # Draw random shock
    Random.seed!(12032020)
    F = Uniform(0, 1)

    # Allocate space for shocks and initialize
    sᵢₜ = zeros(N, T)
    Sₜ = zeros(T)
    sᵢₜ[:,1] .= 1
    Sₜ[1] = 1

    # Iterate over time
    for t = 2:T
        # Draw aggregate shock
        εₜ = rand(F)

        # Check cases
        if Sₜ[t - 1] == 1 && εₜ < Πᵍᵍ
            Sₜ[t] = 1
        elseif Sₜ[t - 1] == 1 && εₜ > Πᵍᵍ
            Sₜ[t] = 2
        elseif Sₜ[t - 1] == 2 && εₜ < Πᵇᵇ
            Sₜ[t] = 2
        elseif Sₜ[t - 1] == 2 && εₜ > Πᵇᵇ
            Sₜ[t] = 1
        end

        # Iterate over individuals
        for n = 1:N
            # Draw idiosyncratic shock
            εᵢₜ = rand(F)

            # Check aggregate state today and tomorrow
            if Sₜ[t - 1] == 1 && Sₜ[t] == 1
                sᵢₜ[n, t] = UpdateState(Mᵍᵍ, sᵢₜ[n, t - 1], εᵢₜ)
            elseif Sₜ[t - 1] == 1 && Sₜ[t] == 2
                sᵢₜ[n, t] = UpdateState(Mᵍᵇ, sᵢₜ[n, t - 1], εᵢₜ)
            elseif Sₜ[t - 1] == 2 && Sₜ[t] == 1
                sᵢₜ[n, t] = UpdateState(Mᵇᵍ, sᵢₜ[n, t - 1], εᵢₜ)
            elseif Sₜ[t - 1] == 2 && Sₜ[t] == 2
                sᵢₜ[n, t] = UpdateState(Mᵇᵇ, sᵢₜ[n, t - 1], εᵢₜ)
            end
        end
    end

    # Return states
    return sᵢₜ, Sₜ
end