# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: September, 2022

##################################################
################# INITIALIZATION #################
##################################################

# Import required packages
using Parameters, LinearAlgebra, Printf

# Include model
include("/Users/jackcollison/Desktop/Wisconsin/Coursework/Second Year/Computational/PS3/Model.jl")

# Structure for results
mutable struct TransitionResults 
    # Value and policy functions
    θ::Array{Float64}
    Z::Array{Float64, 1}
    nz::Int64
    Π::Array{Float64, 2}
    Π₀::Array{Float64}
    e::Array{Float64, 2}
    γ::Float64
    Vₙ::Array{Float64}
    V₀::Array{Float64}
    policy_func::Array{Float64}
    labor_supply::Array{Float64}
    Γ::Array{Float64}
    K::Array{Float64}
    L::Array{Float64}
    w::Array{Float64}
    r::Array{Float64}
    b::Array{Float64}
    T::Int64
end

# Initialize transition
function InitializeTransition(SS¹::Results, SS²::Results, Z::Array{Float64}, γ::Float64, Π::Array{Float64, 2}, Π₀::Array{Float64}, T::Int64, verbose::Bool)
    # Primitives and results from previous problems
    prim = Primitives()
    
    # Vectorize primitives from previous problems
    θ = vcat(SS¹.θ, repeat([SS².θ], T - 1))
    nz = length(Z)
    e = prim.η * Z'

    # Initialize value functions
    Vₙ = SS².value_func
    V₀ = zeros(prim.N, prim.na, nz)

    # Initialize policy functions
    policy_func = zeros(prim.N, prim.na, nz, T)
    labor_supply = zeros(prim.Jᴿ - 1, prim.na, nz, T)

    # Initialize distributions
    Γ = ones(prim.N, prim.na, nz, T) ./ sum(ones(prim.N, prim.na, nz))

    # Vectorize aggregate quantities and prices
    K = collect(range(SS¹.K, SS².K, length = T))
    L = repeat([1.], T)
    w = (1 - prim.α) .* K.^prim.α .* L.^(-prim.α)
    r = prim.α .* K.^(prim.α - 1) .* L.^(1 - prim.α) .- prim.δ
    b = θ .* w .* L ./ reshape(sum(Γ[prim.Jᴿ:prim.N, :, :, :], dims=[1, 2, 3]), T)

    # Fill in known values
    Γ[:, :, :, 1] = SS¹.F
    policy_func[:, :, :, T] = SS².policy_func
    labor_supply[:, :, :, T] = SS².labor_supply

    # Return results
    TransitionResults(θ, Z, nz, Π, Π₀, e, γ, Vₙ, V₀, policy_func, labor_supply, Γ, K, L, w, r, b, T)
end

# Extend results
function ExtendTransition(res::TransitionResults, SS¹::Results, SS²::Results, more::Int64)
    # Primitives and results from previous problems
    prim = Primitives()

    # Extend functions
    res.T += more
    res.θ = vcat(res.θ, repeat([res.θ[res.T - more]], more))
    res.policy_func = zeros(prim.N, prim.na, res.nz, res.T)
    res.labor_supply = zeros(prim.Jᴿ - 1, prim.na, res.nz, res.T)
    res.Γ = ones(prim.N, prim.na, res.nz, res.T) ./ sum(ones(prim.N, prim.na, res.nz))

    # Update aggregate quantities and prices
    res.K = vcat(res.K, collect(range(res.K[more], SS².K, length = more)))
    res.L = vcat(res.L, repeat([1.], more))
    res.w = (1 - prim.α) .* res.K.^α .* res.L.^(-α)
    res.r = prim.α .* res.K.^(prim.α - 1) .* res.L.^(1 - prim.α) .- prim.δ
    res.b = res.θ .* res.w .* res.L ./ reshape(sum(res.Γ[prim.Jᴿ:prim.N, :, :, :], dims=[1, 2, 3]), res.T)

    # Fill in known values
    res.Γ[:, :, :, 1] = SS¹.F
    res.policy_func[:, :, :, T] = SS².policy_func
    res.labor_supply[:, :, :, T] = SS².labor_supply
end

######################################################
################# HOUSEHOLD PROBLEMS #################
######################################################

# Bellman operator for retiree
function RetireeBellmanTransition(res::TransitionResults, next_value::Array{Float64}, t::Int64)
    # Unpack primitives
    @unpack N, Jᴿ, β, σ, A, na = Primitives()

    # Set last value function value
    res.V₀[N, :, 1] = RetireeUtility.((1 + res.r[t]) .* A .+ res.b[t], res.γ, σ)

    # Backward induction over age
    for j = (N - 1):-1:Jᴿ
        # Lower bound for policy
        lowest_index = 1

        # Iterate over asset choices today
        for i_a = 1:na
            # Initialize value, calculate budget
            budget = (1 + res.r[t]) * A[i_a] + res.b[t]
            max_util = -Inf

            # Iterate over asset choies tomorrow
            for i_ap = lowest_index:na
                # Compute value to retiree
                v = RetireeUtility(budget - A[i_ap], res.γ, σ) + β * next_value[j + 1, i_ap, 1]

                # Check decreasing utility or end of grid
                if v < max_util
                    # Update results and break
                    res.V₀[j, i_a, 1] = max_util
                    res.policy_func[j, i_a, 1, t] = A[i_ap - 1]
                    lowest_index = i_ap - 1
                    break
                elseif i_ap == na
                    # Update results
                    res.V₀[j, i_a, 1] = v
                    res.policy_func[j, i_a, 1, t] = A[i_ap]
                end

                # Update maximum utility
                max_util = v
            end
        end
    end

    # Check if there is uncertainty
    if res.nz >= 2
        for i_z = 2:res.nz
            # Fill in values for other productivities
            res.policy_func[:, :, i_z, t] = res.policy_func[:, :, 1, t]
            res.V₀[:, :, i_z] = res.V₀[:, :, 1]
        end
    end
end

# Bellman operator for worker
function WorkerBellmanTransition(res::TransitionResults, next_value::Array{Float64}, t::Int64)
    # Unpack primitives
    @unpack Jᴿ, β, σ, A, na = Primitives()

    # Backwards induction over age
    for j = (Jᴿ - 1):-1:1
        # Iterate over productivity today
        for i_z = 1:res.nz
            # Lower bound for policy
            lowest_index = 1

            # Iterate over asset choices today
            for i_a = 1:na
                # Initialize value
                max_util = -Inf

                # Iterate over asset choices tomorrow
                for i_ap = lowest_index:na
                    # Calculate labor decision, consumption, and value
                    ℓ = LaborDecision(A[i_a], A[i_ap], res.e[j, i_z], res.θ[t], res.γ, res.w[t], res.r[t])
                    c = res.w[t] * (1 - res.θ[t]) * res.e[j,i_z] * ℓ + (1 + res.r[t]) * A[i_a] - A[i_ap]
                    v = WorkerUtility(c, ℓ, res.γ, σ)

                    # Add expected value from tomorrow
                    for i_zp = 1:res.nz
                        v += β * res.Π[i_z, i_zp] * next_value[j + 1, i_ap, i_zp]
                    end

                    # Check decreasing utility or end of grid
                    if v < max_util
                        # Update results and break
                        res.V₀[j, i_a, i_z] = max_util
                        res.policy_func[j, i_a, i_z, t] = A[i_ap - 1]
                        res.labor_supply[j, i_a, i_z, t] = LaborDecision(A[i_a], A[i_ap - 1], res.e[j, i_z], res.θ[t], res.γ, res.w[t], res.r[t])
                        lowest_index = i_ap - 1
                        break
                    elseif i_ap == na
                        # Update results
                        res.V₀[j, i_a, i_z] = v
                        res.policy_func[j, i_a, i_z, t] = A[i_ap]
                        res.labor_supply[j, i_a, i_z, t] = ℓ
                    end

                    # Update maximum value
                    max_util = v
                end
            end
        end
    end
end

# Functionality to solve household problem
function SolveHHTransition(res::TransitionResults, verbose::Bool = false)
    # Print statement
    if verbose
        println("Solving household problem...")
    end

    # Initialize next value
    next_value = res.Vₙ

    # Backwards iteration over time
    for t = (T - 1):-1:1
        # Solve retiree and worker problems
        RetireeBellmanTransition(res, next_value, t)
        WorkerBellmanTransition(res, next_value, t)
        next_value = res.V₀

        # Print statement
        if verbose
            println("Solved for household problem at time = ", t)
        end
    end

    # Print statement
    if verbose
        println("Solved household problem!\n")
    end
end

########################################################
################# DISTRIBUTION PROBLEM #################
########################################################

# Functionality to solve stationary distribution problem
function SolveΓTransition(res::TransitionResults, verbose::Bool = false)
    # Unpack primitives, initialize Γ
    @unpack N, n, A, na = Primitives()

    # Create μ weights
    μ₀ = ones(N)
    for i = 1:(N-1)
        μ₀[i + 1] = μ₀[i] / (1 + n)
    end

    # Normalize μ weights and apply
    μ = reshape(repeat(μ₀ ./ sum(μ₀), res.nz * na), N, na, res.nz)

    # Apply weighting and initial conditions
    res.Γ[:, :, :, 1] = res.Γ[:, :, :, 1] ./ μ
    res.Γ[:, :, :, 2:res.T] .= 0.
    res.Γ[1, 1, :, :] .= transpose(res.Π₀)
    
    # Print statement
    if verbose
        println("Solving distribution problem...")
    end

    # Time period iteration
    for t in 1:(res.T - 1)
        # Age profile iteration
        for j = 1:(N - 1)
            # Iterate over state space
            for i_a = 1:na, i_z = 1:res.nz
                # Skip if no mass
                if res.Γ[j, i_a, i_z, t] > 0
                    # Get index of grid
                    i_ap = argmax(A .== res.policy_func[j, i_a, i_z, t])
                    
                    # Increment probability
                    for i_zp = 1:res.nz
                        res.Γ[j + 1, i_ap, i_zp, t + 1] += res.Π[i_z, i_zp] * res.Γ[j, i_a, i_z, t]
                    end
                end
            end
        end

        # Update distribution
        res.Γ[:, :, :, t] = μ .* res.Γ[:, :, :, t]

        # Print statement
        if verbose
            println("Solved for Γ at time = ", t)
        end
    end

    # Update distribution in final state
    res.Γ[:, :, :, res.T] = μ .* res.Γ[:, :, :, res.T]

    # Print statement
    if verbose
        println("Solved distribution problem!\n")
        println("******************************************************************\n")    
    end
end

# ##############################################################
# ######################## UPDATE PRICES #######################
# ##############################################################

# Update prices
function UpdatePricesTransition(res::TransitionResults, verbose::Bool = false)
    # Upack primitive structure, instantiate value function
    @unpack N, Jᴿ, α, δ = Primitives()

    # Print statement
    if verbose
        println("******************************************************************\n")        
        println("Updating prices...")
    end

    # Update prices and benefits
    for t in 1:T
        res.w[t] = (1 - α) * res.K[t]^α * res.L[t]^(-α)
        res.r[t] = α * res.K[t]^(α - 1) * res.L[t]^(1 - α) - δ
        res.b[t] = res.θ[t] * res.w[t] * res.L[t] / sum(res.Γ[Jᴿ:N, :, :, t])
    end

     # Print statement
     if verbose
        println("Updated prices!\n")
    end
end

# ##############################################################
# ######################## MODEL SOLVER ########################
# ##############################################################

# Functionality to run entire model
function SolveModelTransition(res::TransitionResults, SS¹::Results, SS²::Results, verbose::Bool = false, ρ::Float64 = 0.9, tol::Float64 = 1e-3, more::Int64 = 20)
    # Initialize error and counter
    prev_err = Inf
    err = Inf
    i = 0

    # Print statement
    if verbose
        println("###############################################")
        println("################ SOLVING MODEL ################")
        println("###############################################\n")
    end

    # Loop while not converged
    while err > tol
        # Increment counter
        i += 1

        # Update prices, then solve household and distribution problems
        UpdatePricesTransition(res, verbose)
        SolveHHTransition(res, verbose)
        SolveΓTransition(res, verbose)

        # Compute aggregate capital and labor
        Kⁿᵉʷ = zeros(res.T)
        Lⁿᵉʷ = zeros(res.T)
        for t = 1:T
            Kⁿᵉʷ[t] = sum([sum(sum([res.Γ[j, m, z, t] * A[m] for z = 1:res.nz]) for m = 1:na) for j = 1:N])
            Lⁿᵉʷ[t] = sum([sum(sum([res.Γ[j, m, z, t] * res.e[j, z] * res.labor_supply[j, m, z, t] for z = 1:res.nz]) for m = 1:na) for j = 1:(Jᴿ - 1)])
        end

        # Update error term
        err = maximum(abs.(Kⁿᵉʷ - res.K))

        # Print statement
        if verbose
            println("GE Iteration = ", i, " with error ε = ", err, "\n")
        end

        # Continue if threshold met
        if err <= tol
            continue
        end

        # Update aggregate capital and labor
        res.K = (1 - ρ) * res.K + ρ * Kⁿᵉʷ
        res.L = (1 - ρ) * res.L + ρ * Lⁿᵉʷ

        # Check if periods larger enough
        if abs(err - prev_err) < tol
            # Print statement
            if verbose
                println("******************************************************************\n")        
                println("Increasing number of periods...")
            end

            # Increasing number of periods
            ExtendTransition(res, SS¹, SS², more)

            # Print statement
            if verbose
                println("Increased number of periods!")
                println("******************************************************************\n")
            end
        end

        prev_err = err
    end

    # Print convergence
    println("******************************************************************\n")
    println("GE converged in ", i, " iterations!\n")
    println("******************************************************************\n")
end