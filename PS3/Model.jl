##################################################
################# INITIALIZATION #################
##################################################

# Import required packages
using Distributed
@everywhere using Parameters, LinearAlgebra, Printf, SharedArrays

# Create primitives
@everywhere @with_kw struct Primitives
    # Life-cycle global constants
    N::Int64 = 66
    n::Float64 = 0.011
    Jᴿ::Int64 = 46

    # Preference global constants
    σ::Float64 = 2.0
    β::Float64 = 0.97

    # Productivity global constants
    η::Array{Float64} = map(x -> parse(Float64, x), readlines("/Users/jackcollison/Desktop/Wisconsin/Coursework/Second Year/Computational/PS3/ef.txt"))
    Π::Array{Float64, 2} = [0.9261 0.0739 ; 0.0189 0.9811]
    Π₀::Array{Float64} = [0.2037 0.7963]

    # Production global constants
    α::Float64 = 0.36
    δ::Float64 = 0.06

    # Asset global constants
    A::Array{Float64, 1} = collect(range(0.0, length = 1000, stop = 75.0))
    na::Int64 = length(A)
end

# Structure for results
@everywhere mutable struct Results 
    # Value and policy functions
    θ::Float64
    Z::SharedArray{Float64, 1}
    nz::Int64
    e::SharedArray{Float64, 2}
    γ::Float64
    value_func::SharedArray{Float64}
    policy_func::SharedArray{Float64}
    labor_supply::SharedArray{Float64}
    F::SharedArray{Float64}
    K::Float64
    L::Float64
    w::Float64
    r::Float64
    b::Float64
end

# Initialization function
@everywhere function Initialize(θ::Float64, Z::Array{Float64, 1}, γ::Float64)
    # Initialize results
    prim = Primitives()
    nz = length(Z)
    e = prim.η * Z'
    labor_supply = SharedArray{Float64}(zeros(prim.Jᴿ - 1, prim.na, nz))
    value_func = SharedArray{Float64}(zeros(prim.N, prim.na, nz))
    policy_func = SharedArray{Float64}(zeros(prim.N, prim.na, nz))
    F = SharedArray{Float64}(ones(prim.N, prim.na, nz) / sum(ones(prim.N, prim.na, nz)))

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

# Bellman operator for retiree
@everywhere function RetireeBellman(res::Results)
    # Unpack primitives
    @unpack N, n, Jᴿ, σ, β, η, Π, Π₀, α, δ, A, na = Primitives()

    # Set last value function value
    res.value_func[N, :, 1] = (((1 + res.r) .* A .+ res.b).^((1 - σ) * res.γ)) ./ (1 - σ)

    # Backward induction
    for j = (N-1):-1:Jᴿ
        # Iterate over state space
        @sync @distributed for (i_a, a) in collect(enumerate(A))
            # Consumption matrix
            C = ((1 + res.r) * a + res.b) .- A
            C = ifelse.(C .> 0, 1, 0) .* C

            # Value matrix and maximand
            V = C.^((1 - σ) * res.γ) ./ (1 - σ) + β * res.value_func[j + 1, :, 1]
            Vmax = findmax(V)

            # Update value and policy functions
            res.value_func[j, i_a, :] = transpose(repeat([Vmax[1]], res.nz))
            res.policy_func[j, i_a, :] = transpose(repeat([A[Vmax[2]]], res.nz))
        end
    end
end

# Bellman operator for worker
@everywhere function WorkerBellman(res::Results)
    # Unpack primitives
    @unpack N, n, Jᴿ, σ, β, η, Π, Π₀, α, δ, A, na = Primitives()

    # Backward induction
    for j = (Jᴿ-1):-1:1
        # Iterate over state space
        @sync @distributed for ((i_a, a), (i_z, z)) in collect(Iterators.product(enumerate(A), enumerate(res.Z)))
            # Labor decision
            ℓ = min.(1, max.(0, (res.γ * (1 - res.θ) * res.e[j, i_z] * res.w .- (1 - res.γ) .* ((1 + res.r) * a .- A)) ./ ((1 - res.θ) * res.w * res.e[j, i_z])))

            # Consumption matrix
            C = (res.w * (1 - res.θ) * res.e[j, i_z]) .* ℓ .+ (1 + res.r) * a .- A
            C = ifelse.(C .> 0, 1, 0) .* C

            # Value matrix and maximand
            V = (C.^((1 - σ) * res.γ)) ./ (1 - σ) + β * res.value_func[j + 1, :, :] * Π[i_z, :]
            Vmax = findmax(V)

            # Update value and policy functions
            res.value_func[j, i_a, i_z] = Vmax[1]
            res.policy_func[j, i_a, i_z] = A[Vmax[2]]
            res.labor_supply[j, i_a, i_z] = ℓ[Vmax[2]]
        end
    end
end

# Functionality to solve household problem
@everywhere function SolveHH(res::Results, verbose::Bool = false)
    # Unpack primitives
    @unpack N, n, Jᴿ, σ, β, η, Π, Π₀, α, δ, A, na = Primitives()

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
@everywhere function SolveF(res::Results, verbose::Bool = false)
    # Unpack primitives, initialize F
    @unpack N, n, Jᴿ, σ, β, η, Π, Π₀, α, δ, A, na = Primitives()
    res.F = SharedArray{Float64}(zeros(N, na, res.nz))
    res.F[1, 1, :] = Π₀

    # Print statement
    if verbose
        println("Solving distribution problem...")
    end

    # Age profile iteration
    for j = 1:(N - 1)
        # Iterate over state space
        @sync @distributed for (i_z, (i_a, a)) in collect(Iterators.product(1:res.nz, enumerate(A)))
            # Increment update
            res.F[j + 1, i_a, i_z] += sum((ifelse.(res.policy_func[j, :, :] .== a, 1, 0) .* res.F[j, :, :]) * Π[:, i_z])
        end
    end

    # Create μ weights
    μ = ones(N)
    for i in 1:(N-1)
        μ[i + 1] = μ[i] / (1 + n)
    end

    # Normalize μ weights and reshape
    μ = reshape(repeat(μ ./ sum(μ), res.nz * na), N, na, res.nz)

    # Update distribution
    res.F = res.F .* μ

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
@everywhere function UpdatePrices(res::Results, verbose::Bool = false)
    # Upack primitive structure, instantiate value function
    @unpack N, n, Jᴿ, σ, β, η, Π, Π₀, α, δ, A, na = Primitives()

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
@everywhere function SolveModel(res::Results, verbose::Bool = false, ρ::Float64 = 0.9, tol::Float64 = 1e-3)
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
        Kⁿᵉʷ = sum([sum(sum(res.F[j, m, :] .* A[m]) for m in 1:na) for j in 1:N])
        Lⁿᵉʷ = sum([sum(sum(res.F[j, m, :] .* res.e[j,:] .* res.labor_supply[j, m, :]) for m in 1:na) for j in 1:(Jᴿ - 1)])

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