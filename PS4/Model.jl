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
    value_func::Array{Float64}
    policy_func::Array{Float64}
    labor_supply::Array{Float64}
    F::Array{Float64}
    K::Array{Float64}
    L::Array{Float64}
    w::Array{Float64}
    r::Array{Float64}
    b::Array{Float64}
    T::Int64
end

# Initialize transition
function InitializeTransition(θ₁::Float64, θ₂::Float64, Z::Array{Float64}, γ::Float64, Π::Array{Float64, 2}, Π₀::Array{Float64}, T::Int64, ρ::Float64, verbose::Bool)
    # Primitives and results from previous problems
    prim = Primitives()
    
    # Solve model with idiosyncratic uncertainty and social security
    SS¹ = Initialize(θ₁, Z, γ, Π, Π₀)
    SolveModel(SS¹, verbose, ρ)

    # Solve model with idiosyncratic uncertainty and no social security
    SS² = Initialize(θ₂, Z, γ, Π, Π₀)
    SolveModel(SS², verbose, ρ)

    # Results from previous problems
    θ = vcat(SS¹.θ, repeat([SS².θ], T - 1))
    Z = SS².Z
    nz = length(Z)
    Π = SS².Π
    Π₀ = SS².Π₀
    e = SS².e
    γ = SS².γ

    # Initialize results
    value_func = zeros(prim.N, prim.na, nz, T)
    policy_func = zeros(prim.N, prim.na, nz, T)
    labor_supply = zeros(prim.Jᴿ - 1, prim.na, nz, T)
    F = ones(prim.N, prim.na, nz, T) ./ sum(ones(prim.N, prim.na, nz))

    # Aggregate equilibrium quantities and prices
    K = collect(range(SS¹.K, SS².K, length = T))
    L = repeat([SS¹.L], T)
    w = (1 - α) .* K.^α .* L.^(-α)
    r = α .* K.^(α - 1) .* L.^(1 - α) .- δ
    b = θ .* w .* L ./ reshape(sum(F[Jᴿ:N, :, :, :], dims=[1, 2, 3]), T)

    # Fill in known values
    F[:, :, :, 1] = SS¹.F
    value_func[:, :, :, T] = SS².value_func
    policy_func[:, :, :, T] = SS².policy_func
    labor_supply[:, :, :, T] = SS².labor_supply

    # Return results
    TransitionResults(θ, Z, nz, Π, Π₀, e, γ, value_func, policy_func, labor_supply, F, K, L, w, r, b, T)
end

# Extend results
function ExtendTransition(res::TransitionResults)
    # Update values
    res.T *= 2
    res.θ = vcat(res.θ, repeat([res.θ[T]], T))
    res.value_func = vcat(res.value_func, repeat())
end

######################################################
################# HOUSEHOLD PROBLEMS #################
######################################################

# Bellman operator for retiree
function RetireeBellmanTransition(res::TransitionResults)
    # Unpack primitives
    @unpack N, Jᴿ, β, σ, A, na = Primitives()

    # Set last value function value
    res.value_func[N, :, 1, :] = transpose(repeat(RetireeUtility.(A, res.γ, σ)', res.T))

    # Backwards induction over time
    for t = (res.T - 1):-1:1
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
                    v = RetireeUtility(budget - A[i_ap], res.γ, σ) + β * res.value_func[j + 1, i_ap, 1, t + 1]

                    # Check decreasing utility or end of grid
                    if v < max_util
                        # Update results and break
                        res.value_func[j, i_a, 1, t] = max_util
                        res.policy_func[j, i_a, 1, t] = A[i_ap - 1]
                        lowest_index = i_ap - 1
                        break
                    elseif i_ap == na
                        # Update results
                        res.value_func[j, i_a, 1, t] = v
                        res.policy_func[j, i_a, 1, t] = A[i_ap]
                    end

                    # Update maximum utility
                    max_util = v
                end
            end
        end
    end

    # Check if there is uncertainty
    if res.nz >= 2
        for i_z = 2:res.nz
            # Fill in values for other productivities
            res.policy_func[:, :, i_z, :] = res.policy_func[:, :, 1, :]
            res.value_func[:, :, i_z, :] = res.value_func[:, :, 1, :]
        end
    end
end

# Compute labor decision
function LaborDecision(a::Float64, ap::Float64, e::Float64, θ::Float64, γ::Float64, w::Float64, r::Float64)
    # Compute labor decision
    interior = (γ * (1 - θ) * e * w - (1 - γ) * ((1 + r) * a - ap)) / ((1 - θ) * e * w)
    min(1, max(0, interior))
end

# Worker utility function
function WorkerUtility(c::Float64, ℓ::Float64, γ::Float64, σ::Float64)
    if (c > 0 && ℓ >= 0 && ℓ <= 1)
        (((c^γ) * ((1 - ℓ)^(1 - γ)))^(1 - σ)) / (1 - σ)
    else
        -Inf
    end
end

# Bellman operator for worker
function WorkerBellmanTransition(res::TransitionResults)
    # Unpack primitives
    @unpack Jᴿ, β, σ, A, na = Primitives()

    # Backwards induction over time
    for t = (res.T - 1):-1:1
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
                            v += β * res.Π[i_z, i_zp] * res.value_func[j + 1, i_ap, i_zp, t + 1]
                        end

                        # Check decreasing utility or end of grid
                        if v < max_util
                            # Update results and break
                            res.value_func[j, i_a, i_z, t] = max_util
                            res.policy_func[j, i_a, i_z, t] = A[i_ap - 1]
                            res.labor_supply[j, i_a, i_z, t] = LaborDecision(A[i_a], A[i_ap - 1], res.e[j, i_z], res.θ[t], res.γ, res.w[t], res.r[t])
                            lowest_index = i_ap - 1
                            break
                        elseif i_ap == na
                            # Update results
                            res.value_func[j, i_a, i_z, t] = v
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
end

# Functionality to solve household problem
function SolveHHTransition(res::TransitionResults, verbose::Bool = false)
    # Print statement
    if verbose
        println("Solving household problem...")
    end

    # Solve retiree and worker problems
    RetireeBellmanTransition(res)
    WorkerBellmanTransition(res)

    # Print statement
    if verbose
        println("Solved household problem!\n")
    end
end

########################################################
################# DISTRIBUTION PROBLEM #################
########################################################

# Functionality to solve stationary distribution problem
function SolveFTransition(res::TransitionResults, verbose::Bool = false)
    # Unpack primitives, initialize F
    @unpack N, n, A, na = Primitives()
    res.F = zeros(N, na, res.nz, res.T)
    
    # Create μ weights
    μ₀ = ones(N)
    for i = 1:(N-1)
        μ₀[i + 1] = μ₀[i] / (1 + n)
    end

    # Normalize μ weights, reshape, add ergodic distribution
    μ = reshape(repeat(μ₀ ./ sum(μ₀), res.nz * na), N, na, res.nz)
    res.F[:, :, :, 1] = res.F[:, :, :, 1] ./ μ
    res.F[1, 1, :, :] = res.Π₀

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
                if res.F[j, i_a, i_z, t] > 0
                    # Get index of grid
                    i_ap = argmax(A .== res.policy_func[j, i_a, i_z, t])
                    
                    # Increment probability
                    for i_zp = 1:res.nz
                        res.F[j + 1, i_ap, i_zp, t + 1] += res.Π[i_z, i_zp] * res.F[j, i_a, i_z, t]
                    end
                end
            end
        end
    end

    # Update distribution
    res.F = μ .* res.F

    # Print statement
    if verbose
        println("Solved distribution problem!\n")
        println("******************************************************************\n")    
    end
end

##############################################################
######################## UPDATE PRICES #######################
##############################################################

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
        res.b[t] = res.θ[t] * res.w[t] * res.L[t] / sum(res.F[Jᴿ:N, :, :, t])
    end

     # Print statement
     if verbose
        println("Updated prices!\n")
    end
end

##############################################################
######################## MODEL SOLVER ########################
##############################################################

# Functionality to run entire model
function SolveModelTransition(res::TransitionResults, verbose::Bool = false, ρ::Float64 = 0.9, tol::Float64 = 1e-3)
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
        SolveFTransition(res, verbose)

        # Compute aggregate capital and labor
        Kⁿᵉʷ = [sum([sum(sum([res.F[j, m, z, t] * A[m] for z = 1:res.nz]) for m = 1:na) for j = 1:N]) for t in 1:res.T]
        Lⁿᵉʷ = [sum([sum(sum([res.F[j, m, z, t] * res.e[j, z] * res.labor_supply[j, m, z, t] for z = 1:res.nz]) for m = 1:na) for j = 1:(Jᴿ - 1)]) for t in 1:res.T]

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
            ExtendTransition(res)

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