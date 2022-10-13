# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: October, 2022

##################################################
################# INITIALIZATION #################
##################################################

# Import required packages
using Parameters

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
    k::Array{Float64,1} = collect(range(0.0, stop = 15.0, length = nk))

    # Aggregate capital grid
    nK::Int64 = 11
    K::Array{Float64,1} = collect(range(11.0, stop = 15.0, length = nK))

    # Labor productivity grid
    ε::Array{Float64,1} = [0.3721, 0.0]
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
    Πᵍᵇ::Float64 = 1.0 - (dᵇ - 1.0) / dᵇ
    Πᵇᵍ::Float64 = 1.0 - (dᵍ - 1.0) / dᵍ
    Πᵇᵇ::Float64 = (dᵇ - 1.0) / dᵇ

    # Transition probabilities for aggregate states and staying unemployed
    πᵍᵍ⁰⁰::Float64 = (dᵘᵍ - 1.0) / dᵘᵍ
    πᵇᵇ⁰⁰::Float64 = (dᵘᵇ - 1.0) / dᵘᵇ
    πᵇᵍ⁰⁰::Float64 = 1.25 * πᵇᵇ⁰⁰
    πᵍᵇ⁰⁰::Float64 = 0.75 * πᵍᵍ⁰⁰

    # Transition probabilities for aggregate states and becoming employed
    πᵍᵍ⁰¹::Float64 = (uᵍ - uᵍ * πᵍᵍ⁰⁰) / (1.0 - uᵍ)
    πᵇᵇ⁰¹::Float64 = (uᵇ - uᵇ * πᵇᵇ⁰⁰) / (1.0 - uᵇ)
    πᵇᵍ⁰¹::Float64 = (uᵇ - uᵍ * πᵇᵍ⁰⁰) / (1.0 - uᵍ)
    πᵍᵇ⁰¹::Float64 = (uᵍ - uᵇ * πᵍᵇ⁰⁰) / (1.0 - uᵇ)

    # Transition probabilities for aggregate states and becoming unemployed
    πᵍᵍ¹⁰::Float64 = 1.0 - (dᵘᵍ - 1.0) / dᵘᵍ
    πᵇᵇ¹⁰::Float64 = 1.0 - (dᵘᵇ - 1.0) / dᵘᵇ
    πᵇᵍ¹⁰::Float64 = 1.0 - 1.25 * πᵇᵇ⁰⁰
    πᵍᵇ¹⁰::Float64 = 1.0 - 0.75 * πᵍᵍ⁰⁰

    # Transition probabilities for aggregate states and staying employed
    πᵍᵍ¹¹::Float64 = 1.0 - (uᵍ - uᵍ * πᵍᵍ⁰⁰) / (1.0 - uᵍ)
    πᵇᵇ¹¹::Float64 = 1.0 - (uᵇ - uᵇ * πᵇᵇ⁰⁰) / (1.0 - uᵇ)
    πᵇᵍ¹¹::Float64 = 1.0 - (uᵇ - uᵍ * πᵇᵍ⁰⁰) / (1.0 - uᵍ)
    πᵍᵇ¹¹::Float64 = 1.0 - (uᵍ - uᵇ * πᵍᵇ⁰⁰) / (1.0 - uᵇ)

    # Markov transition matrix
    Mᵍᵍ::Array{Float64,2} = reshape([πᵍᵍ¹¹ πᵍᵍ¹⁰ πᵍᵍ⁰¹ πᵍᵍ⁰⁰], (2, 2))
    Mᵇᵍ::Array{Float64,2} = reshape([πᵍᵇ¹¹ πᵍᵇ¹⁰ πᵍᵇ⁰¹ πᵍᵇ⁰⁰], (2, 2))
    Mᵍᵇ::Array{Float64,2} = reshape([πᵇᵍ¹¹ πᵇᵍ¹⁰ πᵇᵍ⁰¹ πᵇᵍ⁰⁰], (2, 2))
    Mᵇᵇ::Array{Float64,2} = reshape([πᵇᵇ¹¹ πᵇᵇ¹⁰ πᵇᵇ⁰¹ πᵇᵇ⁰⁰], (2, 2))
    M::Array{Float64,2} = reshape([Πᵍᵍ * Mᵍᵍ Πᵍᵇ * Mᵍᵇ Πᵇᵍ * Mᵇᵍ Πᵇᵇ * Mᵇᵇ], (4, 4))
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