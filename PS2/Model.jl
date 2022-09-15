##################################################
################# INITIALIZATION #################
##################################################

# Import required packages
using Parameters, LinearAlgebra, Interpolations, Optim, Printf

# Create primitives
@with_kw struct Primitives
    # Define global constants
    β::Float64 = 0.9932
    α::Float64 = 1.5
    S::Array{Float64, 1} = [1.0, 0.5]
    ns::Int64 = length(S)
    Π::Array{Float64, 2} = [0.97 0.03; 0.5 0.5]
    A::Array{Float64, 1} = collect(range(-2.0, length = 100, stop = 5.0))
    na::Int64 = length(A)
end

# Structure for results
mutable struct Results
    # Value and policy functions
    value_func::Array{Float64,2}
    policy_func::Array{Float64,2}
    μ::Array{Float64, 2}
    q::Float64
end

# Initialization function
function Initialize()
    # Initialize results
    prim = Primitives()
    val_func = reshape(zeros(2 * prim.na), prim.na, 2) 
    pol_func = reshape(zeros(2 * prim.na), prim.na, 2)
    μ = reshape(ones(2 * prim.na), prim.na, 2) / (prim.na * 2)
    q = (1 + prim.β) / 2

    # Return structure
    Results(val_func, pol_func, μ, q) 
end

#####################################################
################# HOUSEHOLD PROBLEM #################
#####################################################

#Bellman operator for household
function HHBellman(res::Results)
    # Upack primitive structure, instantiate value function
    @unpack β, α, S, ns, Π, A, na = Primitives()
    v_next = zeros(na, ns)

    # Iterate over state space
    for (i_s, s) in enumerate(S)
        for (i_a, a) in enumerate(A)
            # Budget constraint, consumption, candidate maximum
            budget =  s + a
            candidate_max = -Inf

            # Iterate over state space
            for (i_ap, ap) in enumerate(A)
                # Consumption
                c = budget - ap * res.q

                # Check for positivity
                if c > 0
                    # Calculate value function
                    val = (c^(1 - α) - 1) / (1 - α) + β * Π[i_s, :]⋅res.value_func[i_ap, :]

                    # Check candidate value
                    if val > candidate_max
                        # Set new maximizers
                        res.policy_func[i_a, i_s] = ap
                        candidate_max = val
                    end
                end
            end

            # Update value function
            v_next[i_a, i_s] = candidate_max
        end
    end

    # Return value
    v_next
end

# Functionality to solve household problem
function SolveHH(res::Results, verbose::Bool = false, tol::Float64 = 1e-5)
    # Initialize error, counter
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
        err = abs.(maximum(v_next .- res.value_func)) / abs(maximum(v_next))
        res.value_func = v_next

        # Print progress
        if verbose
            @printf "HH Iteration = %-12d Error = %.5g\n" i err
        end
    end

    # Print convergence
    if verbose
        println("\n******************************************************************\n")
        println("Household problem converged in ", i, " iterations!\n")
        println("******************************************************************\n")
    end
end

########################################################
################# DISTRIBUTION PROBLEM #################
########################################################

# Update operator for distribution
function μUpdate(res::Results)
    # Upack primitive structure, instantiate value function
    @unpack β, α, S, ns, Π, A, na = Primitives()
    μ_next = zeros(na, ns)

    # Iterate over state space
    for i_s in 1:ns
        for (i_a, a) in enumerate(A)
            # Get policy function index
            Idx = map((ap) -> ap == a, res.policy_func)

            # Increment update
            μ_next[i_a, i_s] += sum((Idx .* res.μ) * Π[:, i_s])
        end
    end

    # Return μ
    μ_next
end

# Functionality to solve stationary distribution problem
function Solveμ(res::Results, verbose::Bool = false, tol::Float64 = 1e-5)
    # Initialize error, counter
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
        err = abs.(maximum(μ_next .- res.μ)) / abs(maximum(μ_next))
        res.μ = μ_next

        # Print progress
        if verbose
            @printf "μ Iteration = %-12d Error = %.5g\n" i err
        end
    end

    # Print convergence
    if verbose
        println("\n******************************************************************\n")
        println("μ converged in ", i, " iterations!\n")
        println("******************************************************************\n")
    end
end

##############################################################
################# MARKET CLEARING CONDITIONS #################
##############################################################

# Update price based on market clearing
function UpdatePrice(res::Results, verbose::Bool = false, tol::Float64 = 1e-2)
    # Upack primitive structure, instantiate value function
    @unpack β, α, S, ns, Π, A, na = Primitives()
    ed = sum(res.μ .* res.policy_func)

    # Check positive excess demand
    if ed > tol
        # Update price
        oldq = res.q
        res.q += abs(ed) * (1 - res.q) / 2

        # Print convergence
        if verbose
            println("\n******************************************************************\n")
            @printf "Excess Demand = %-8.6g Old Price = %-8.6d New Price = %.6d\n\n" ed oldq res.q
            println("******************************************************************\n")
        end

        # Return value
        return(false)
    elseif ed < -tol
        # Update price
        oldq = res.q
        res.q -= abs(ed) * (1 - res.q) / 2

        # Print convergence
        if verbose
            println("\n******************************************************************\n")
            @printf "Excess Demand = %-8.6g Old Price = %-8.6f New Price = %.6f\n\n" ed oldq res.q
            println("******************************************************************\n")
        end

        # Return value
        return(false)
    else
        # Print convergence
        if verbose
            println("\n******************************************************************\n")
            @printf "Excess Demand = %.6f is within threshold!\n\n" ed
            println("******************************************************************\n")
        end

        # Return within tolerance
        return(true)
    end
end

##############################################################
######################## MODEL SOLVER ########################
##############################################################

# Functionality to run entire model
function SolveModel(res::Results, verbose::Bool = false)
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
    println("***********************************************\n")
    println("GE converged in ", i, " iterations!")
    println("***********************************************\n")
end