# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: September, 2022

##################################################
################# INITIALIZATION #################
##################################################

# Import required packages
using Distributed
@everywhere using Parameters, LinearAlgebra, Printf, SharedArrays

# Create primitives
@everywhere @with_kw struct Primitives
    # Define global constants
    β::Float64 = 0.9932
    α::Float64 = 1.5
    S::Array{Float64, 1} = [1.0, 0.5]
    ns::Int64 = length(S)
    Π::Array{Float64, 2} = [0.97 0.03; 0.5 0.5]
    A::Array{Float64, 1} = collect(range(-2.0, length = 1000, stop = 5.0))
    na::Int64 = length(A)
end

# Structure for results
@everywhere mutable struct Results
    # Value and policy functions
    value_func::SharedArray{Float64,2}
    policy_func::SharedArray{Float64,2}
    μ::SharedArray{Float64, 2}
    q::Float64
end

# Initialization function
@everywhere function Initialize()
    # Initialize results
    prim = Primitives()
    val_func = SharedArray{Float64}(reshape(zeros(2 * prim.na), prim.na, 2))
    pol_func = SharedArray{Float64}(reshape(zeros(2 * prim.na), prim.na, 2))
    μ = SharedArray{Float64}(ones(prim.na) ./ prim.na * [0.9434 0.0566])
    q = (1 + prim.β) / 2

    # Return structure
    Results(val_func, pol_func, μ, q) 
end

#####################################################
################# HOUSEHOLD PROBLEM #################
#####################################################

#Bellman operator for household
@everywhere function HHBellman(res::Results)
    # Upack primitive structure, instantiate value function
    @unpack β, α, S, ns, Π, A, na = Primitives()
    v_next = SharedArray{Float64}(zeros(na, ns))

    # Iterate over state space
    @sync @distributed for ((i_s, s), (i_a, a)) in collect(Iterators.product(enumerate(S), enumerate(A)))
        # Consumption matrix
        C = s + a .- (res.q .* A)
        C = ifelse.(C .> 0, 1, 0) .* C

        # Value matrix and maximand
        V = (C.^(1 - α) .- 1) ./ (1 - α) + β * res.value_func * Π[i_s, :]
        Vmax = findmax(V)

        # Update value and policy functions
        v_next[i_a, i_s] = Vmax[1]
        res.policy_func[i_a, i_s] = A[Vmax[2]]
    end

    # Return value
    v_next
end

# Functionality to solve household problem
@everywhere function SolveHH(res::Results, verbose::Bool = false, tol::Float64 = 1e-5)
    # Unpack primitives, initialize error, counter
    @unpack β, α, S, ns, Π, A, na = Primitives()
    err = Inf
    i = 0

    # Print statement
    if verbose
        println("###############################################")
        println("########## SOLVING HOUSEHOLD PROBLEM ##########")
        println("###############################################\n")
    end

    # Loop until convergence
    while err > tol
        # Increment counter
        i += 1

        # Call Bellman operator and update
        v_next = HHBellman(res)
        err = maximum([abs.(maximum(v_next[:,i] .- res.value_func[:,i])) / abs(maximum(v_next[:,i])) for i in 1:ns])
        res.value_func = v_next

        # Print progress
        if verbose
            @printf "HH Iteration = %-12d Error = %.5g\n" i err
        end
    end

    # Print convergence
    if verbose
        println("\n********************************************************************\n")
        println("Household problem converged in ", i, " iterations!\n")
        println("********************************************************************\n")
    end
end

########################################################
################# DISTRIBUTION PROBLEM #################
########################################################

# Update operator for distribution
@everywhere function μUpdate(res::Results)
    # Upack primitive structure, instantiate value function
    @unpack β, α, S, ns, Π, A, na = Primitives()
    μ_next = SharedArray{Float64}(zeros(na, ns))

    # Iterate over state space
    @sync @distributed for (i_s, (i_a, a)) in collect(Iterators.product(1:ns, enumerate(A)))
        # Increment update
        μ_next[i_a, i_s] += sum((ifelse.(res.policy_func .== a, 1, 0) .* res.μ) * Π[:, i_s])
    end

    # Return μ
    μ_next
end

# Functionality to solve stationary distribution problem
@everywhere function Solveμ(res::Results, verbose::Bool = false, tol::Float64 = 1e-5)
    # Unpack primitives, initialize error, counter
    @unpack β, α, S, ns, Π, A, na = Primitives()
    err = Inf
    i = 0

    # Print statement
    if verbose
        println("###############################################")
        println("######## SOLVING DISTRIBUTION PROBLEM #########")
        println("###############################################\n")
    end

    # Loop until convergence
    while err > tol
        # Increment counter
        i += 1

        # Call update operator
        μ_next = μUpdate(res)
        err = maximum([abs.(maximum(μ_next[:,i] .- res.μ[:,i])) / abs(maximum(μ_next[:,i])) for i in 1:ns])
        res.μ = μ_next

        # Print progress
        if verbose
            @printf "μ Iteration = %-12d Error = %.5g\n" i err
        end
    end

    # Print convergence
    if verbose
        println("\n********************************************************************\n")
        println("μ converged in ", i, " iterations!\n")
        println("********************************************************************\n")
    end
end

##############################################################
################# MARKET CLEARING CONDITIONS #################
##############################################################

# Update price based on market clearing
@everywhere function UpdatePrice(res::Results, verbose::Bool = false, tol::Float64 = 1e-3)
    # Upack primitive structure, instantiate value function
    @unpack β, α, S, ns, Π, A, na = Primitives()
    ed = sum(res.μ .* res.policy_func)

    # Print statement
    if verbose
        println("###############################################")
        println("############### UPDATING PRICES ###############")
        println("###############################################\n")
    end

    # Check positive excess demand
    if ed > tol
        # Update price
        oldq = res.q
        res.q += abs(ed) * (1 - res.q) / 2

        # Print convergence
        if verbose
            println("********************************************************************\n")
            @printf "Excess Demand = %-8.6g Old Price = %-8.6f New Price = %.6f\n\n" ed oldq res.q
            println("********************************************************************\n")
        end

        # Return value
        return(false)
    elseif ed < -tol
        # Update price
        oldq = res.q
        res.q -= abs(ed) * (res.q - β) / 2

        # Print convergence
        if verbose
            println("********************************************************************\n")
            @printf "Excess Demand = %-8.6g Old Price = %-8.6f New Price = %.6f\n\n" ed oldq res.q
            println("********************************************************************\n")
        end

        # Return value
        return(false)
    else
        # Print convergence
        if verbose
            println("********************************************************************\n")
            @printf "Excess Demand = %.6f is within threshold!\n\n" ed
            println("********************************************************************\n")
        end

        # Return within tolerance
        return(true)
    end
end

##############################################################
######################## MODEL SOLVER ########################
##############################################################

# Functionality to run entire model
@everywhere function SolveModel(res::Results, verbose::Bool = false)
    # Initialize convergence
    converged = false
    i = 0

    # Print statement
    if verbose
        println("###############################################")
        println("################ SOLVING MODEL ################")
        println("###############################################\n")
    end

    # Loop while not converged
    while !converged
        # Increment counter
        i += 1

        # Solve household, distribution, and update prices
        SolveHH(res, verbose)
        Solveμ(res, verbose)
        converged = UpdatePrice(res, verbose)

        # Print statement
        if verbose
            println("GE Iteration = ", i, "\n")
        end
    end

    # Print convergence
    println("******************************************************************\n")
    println("GE converged in ", i, " iterations!\n")
    println("******************************************************************\n")
end