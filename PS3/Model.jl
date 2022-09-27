# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: September, 2022

##################################################
################# INITIALIZATION #################
##################################################

# Import required packages
using Distributed
using Parameters, LinearAlgebra, Printf, SharedArrays

# Create primitives
@with_kw struct Primitives
    # Life-cycle global constants
    N::Int64 = 66
    n::Float64 = 0.011
    Jᴿ::Int64 = 46

    # Preference global constants
    σ::Float64 = 2.0
    β::Float64 = 0.97

    # Productivity global constants
    η::Array{Float64, 1} = map(x -> parse(Float64, x), readlines("/Users/jackcollison/Desktop/Wisconsin/Coursework/Second Year/Computational/PS3/ef.txt"))
    Π::Array{Float64, 2} = [0.9261 0.0739 ; 0.0189 0.9811]
    Π₀::Array{Float64} = [0.2037 0.7963]

    # Production global constants
    α::Float64 = 0.36
    δ::Float64 = 0.06

    # Asset global constants
    A::Array{Float64, 1} = collect(range(0.0, length = 5000, stop = 75.0))
    na::Int64 = length(A)
end

# Structure for results
@everywhere mutable struct Results 
    # Value and policy functions
    θ::Float64
    Z::Array{Float64, 1}
    nz::Int64
    e::Array{Float64, 2}
    γ::Float64
    value_func::Array{Float64}
    policy_func::Array{Float64}
    labor_supply::Array{Float64}
    F::Array{Float64}
    K::Float64
    L::Float64
    w::Float64
    r::Float64
    b::Float64
end

# Initialization function
function Initialize(θ::Float64, Z::Array{Float64, 1}, γ::Float64)
    # Initialize results
    prim = Primitives()
    nz = length(Z)
    e = prim.η * Z'
    labor_supply = Array{Float64}(zeros(prim.Jᴿ - 1, prim.na, nz))
    value_func = Array{Float64}(zeros(prim.N, prim.na, nz))
    policy_func = Array{Float64}(zeros(prim.N, prim.na, nz))
    F = Array{Float64}(ones(prim.N, prim.na, nz) / sum(ones(prim.N, prim.na, nz)))

    # Initial guesses
    K = 3.0
    L = 0.3
    w = 1.05
    r = 0.05
    b = 0.2

    # Return structure
    Results(θ, Z, nz, e, γ, value_func, policy_func, labor_supply, F, K, L, w, r, b)
end

######################################################
################# HOUSEHOLD PROBLEMS #################
######################################################

# Retiree utility function
function RetireeUtility(c::Float64, σ::Float64, γ::Float64)
    if c > 0
        c^((1 - σ) * γ) / (1 - σ)
    else
        -Inf
    end
end

# Bellman operator for retiree
function RetireeBellman(res::Results)
    # Unpack primitives
    @unpack N, Jᴿ, σ, β, A, na = Primitives()

    # Set last value function value
    res.value_func[N, :, 1] = RetireeUtility.((1 + res.r) .* A .+ res.b, res.γ, σ)

    # Backward induction
    for j = (N-1):-1:Jᴿ
        # Lower bound for policy
        lowest_index = 1

        # Iterate over asset choices today
        for i_a in 1:na
            # Candidate maximum value, budget
            max_util = -Inf
            budget = (1 + res.r) * A[i_a] + res.b

            # Iterate over asset chocies tomorrow
            for i_ap in lowest_index:na
                # Compute value to retiree
                v = RetireeUtility(budget - A[i_ap], σ, res.γ) + β * res.value_func[j + 1, i_ap, 1]

                # Check if value decreases or end of grid
                if v < max_util
                    # Update value and policy functions
                    res.value_func[j, i_a, 1] = max_util
                    res.policy_func[j, i_a, 1] = A[i_ap - 1]

                    # Update lower bound and break
                    lowest_index = i_ap - 1
                    break
                elseif i_ap == na
                    # Update value and policy functions
                    res.value_func[j, i_a, 1] = v
                    res.policy_func[j, i_a, 1] = A[i_ap]
                end

                # Update candidate maximum value
                max_util = v
            end
        end
    end

    # Fill in missing points
    if res.nz >= 2
        for i_z = 2:res.nz
            res.policy_func[:, :, i_z] = res.policy_func[:, :, 1]
            res.value_func[:, :, i_z] = res.policy_func[:, :, 1]
        end
    end
end

# Compute labor decision
function LaborDecision(a::Float64, ap::Float64, e::Float64, θ::Float64, γ::Float64, w::Float64, r::Float64)
    # Compute labor decision
    min(1, max(0, (γ * (1 - θ) * e * w - (1 - γ) * ((1 + r) * a - ap)) / ((1 - θ) * w * e)))
end

# Worker utility function
function WorkerUtility(c::Float64, ℓ::Float64, σ::Float64, γ::Float64)
    if c > 0
        (c^γ * (1 - ℓ)^(1 - γ))^(1 - σ) / (1 - σ)
    else
        -Inf
    end
end

# Bellman operator for worker
function WorkerBellman(res::Results)
    # Unpack primitives
    @unpack N, n, Jᴿ, σ, β, η, Π, Π₀, α, δ, A, na = Primitives()

    # Backward induction
    for j = (Jᴿ-1):-1:1
        # Iterate over possible states
        for i_z = 1:res.nz
            # Lower bound for policy
            lowest_index = 1

            # Iterate over asset choices today
            for i_a in 1:na
                # Candidate maximum value
                max_util = -Inf

                # Iterate over asset choices tomorrow
                for i_ap in lowest_index:na
                    # Solve for labor decision
                    ℓ = LaborDecision(A[i_a], A[i_ap], res.e[j, i_z], res.θ, res.γ, res.w, res.r)

                    # Solve for consumption
                    c = (res.w * (1 - res.θ) * res.e[j, i_z]) * ℓ + (1 + res.r) * A[i_a] - A[i_ap]

                    # Compute value to worker
                    v = WorkerUtility(c, ℓ, σ, res.γ)

                    # Add on next period expected value
                    for i_zp = 1:res.nz
                        v += β * res.value_func[j + 1, i_ap, i_zp] * Π[i_z, i_zp]
                    end

                    # Check if value decreases or end of grid
                    if v < max_util
                        # Update value and policy functions
                        res.value_func[j, i_a, i_z] = max_util
                        res.policy_func[j, i_a, i_z] = A[i_ap - 1]
                        res.labor_supply[j, i_a, i_z] = LaborDecision(A[i_a], A[i_ap - 1], res.e[j, i_z], res.θ, res.γ, res.w, res.r)

                        # Update lower bound and break
                        lowest_index = i_ap - 1
                        break
                    elseif i_ap == na
                        # Update value and policy functions
                        res.value_func[j, i_a, i_z] = v
                        res.policy_func[j, i_a, i_z] = A[i_ap]
                        res.labor_supply[j, i_a, i_z] = ℓ
                    end

                    # Update candidate maximum value
                    max_util = v
                end
            end
        end
    end
end

# Functionality to solve household problem
function SolveHH(res::Results, verbose::Bool = false)
    # Print statement
    if verbose
        println("Solving household problem...")
    end

    # Solve retiree and worker problems
    RetireeBellman(res)
    WorkerBellman(res)

    # Print statement
    if verbose
        println("Solved household problem!\n")
    end
end

########################################################
################# DISTRIBUTION PROBLEM #################
########################################################

# Functionality to solve stationary distribution problem
function SolveF(res::Results, verbose::Bool = false)
    # Unpack primitives, initialize F
    @unpack N, n, Π, Π₀, A, na = Primitives()
    res.F = Array{Float64}(zeros(N, na, res.nz))
    res.F[1, 1, :] = Π₀

    # Print statement
    if verbose
        println("Solving distribution problem...")
    end

    # Age profile iteration
    for j = 1:(N - 1)
        # Iterate over state space
        for i_a = 1:na
            for i_z = 1:res.nz
                # Skip if no mass
                if results.F[j, i_a, i_z] == 0
                    continue
                end

                # Get index of grid
                i_ap = argmax(A .== res.policy_func[j, i_a, i_z])
                
                # Loop over productivity tomorrow
                for i_zp = 1:res.nz
                    # Increment distribution
                    res.F[j + 1, i_ap, i_zp] += Π[i_z, i_zp] * res.F[j, i_a, i_z]
                end
            end
        end
    end

    # Create μ weights
    μ₀ = ones(N)
    for i = 1:(N-1)
        μ₀[i + 1] = μ₀[i] / (1 + n)
    end

    # Normalize μ weights and reshape
    μ = reshape(repeat(μ₀ ./ sum(μ₀), res.nz * na), N, na, res.nz)

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
function UpdatePrices(res::Results, verbose::Bool = false)
    # Upack primitive structure, instantiate value function
    @unpack N, Jᴿ, α, δ = Primitives()

    # Print statement
    if verbose
        println("******************************************************************\n")        
        println("Updating prices...")
    end

    # Update prices and benefits
    res.w = (1 - α) * res.K^α * res.L^(-α)
    res.r = α * res.K^(α - 1) * res.L^(1 - α) - δ
    res.b = res.θ * res.w * res.L / sum(res.F[Jᴿ:N, :, :])

     # Print statement
     if verbose
        println("Updated prices!\n")
    end
end

##############################################################
######################## MODEL SOLVER ########################
##############################################################

# Functionality to run entire model
function SolveModel(res::Results, verbose::Bool = false, ρ::Float64 = 0.9, tol::Float64 = 1e-3)
    # Initialize error and counter
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
        UpdatePrices(res, verbose)
        SolveHH(res, verbose)
        SolveF(res, verbose)

        # Compute aggregate capital and labor
        Kⁿᵉʷ = sum([sum(sum([res.F[j, m, z] * A[m] for z = 1:res.nz]) for m = 1:na) for j = 1:N])
        Lⁿᵉʷ = sum([sum(sum([res.F[j, m, z] * res.e[j, z] * res.labor_supply[j, m, z] for z = 1:res.nz]) for m = 1:na) for j = 1:(Jᴿ - 1)])

        # Update error term
        err = abs(Kⁿᵉʷ - res.K) + abs(Lⁿᵉʷ - res.L)

        # Print statement
        if verbose
            println("GE Iteration = ", i, " with error ε = ", err, "\n")
        end

        # Continue if threshold met
        if err <= tol
            continue
        end

        # Update aggregate capital and labor
        res.K = ρ * res.K + (1 - ρ) * Kⁿᵉʷ
        res.L = ρ * res.L + (1 - ρ) * Lⁿᵉʷ
    end

    # Print convergence
    println("******************************************************************\n")
    println("GE converged in ", i, " iterations!\n")
    println("******************************************************************\n")
end