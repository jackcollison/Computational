# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: October, 2022

########################################################
###################### MODEL SOLVER ####################
########################################################

# Include household solver
include("Household.jl")

# Initialization of results
function Initialize()
    # Get primitives
    G = Grids()

    # Value and policy functions
    policy_func = zeros(G.nk, G.ne, G.nK, G.nz)
    value_func = zeros(G.nk, G.ne, G.nK, G.nz)

    # Regression coefficients
    a₀ = 0.095
    a₁ = 0.999
    b₀ = 0.085
    b₁ = 0.999

    # Regression R²
    R² = 0.0

    # Return results
    Results(policy_func, value_func, a₀, a₁, b₀, b₁, R²)
end

# Capital simulation
function CapitalSimulation(R::Results, sᵢₜ::Array{Float64}, Sₜ::Array{Float64})
    # Get primitives
    P = Primitives()
    G = Grids()

    # Initialize capital
    K_path = zeros(P.T)
    k_today = ones(P.N) .* P.Kˢˢ
    k_tomorrow = zeros(P.N)

    # Interpolation function
    p̂ = interpolate(R.policy_func, BSpline(Linear()))

    # Initial condition
    i_K = GetIndex(P.Kˢˢ, G.K)

    # Update initial capital paths
    for i in 1:P.N
        # Index of steady state
        i_k = GetIndex(P.Kˢˢ, G.k)
        k_tomorrow[i] = p̂(i_k, sᵢₜ[i, 1], i_K, Sₜ[1])
    end

    # Aggregate initial condition
    K_path[1] = mean(k_tomorrow)
    k_today = k_tomorrow

    # Iterate over time
    for t = 2:P.T
        # Get aggregate capital index
        i_K = GetIndex(K_path[t - 1], G.K)

        # Iterate over individuals
        for i = 1:P.N
            # Get index and update
            i_k = GetIndex(k_today[i], G.k)
            k_tomorrow[i] = p̂(i_k, sᵢₜ[i, t], i_K, Sₜ[t])
        end

        # Update aggregate capital path and capital today
        K_path[t] = mean(k_tomorrow)
        k_today = k_tomorrow
    end

    # Return value
    return K_path
end

# Regression helper for coefficients
function CoefficientSolver(Y::Array{Float64}, X::Array{Float64})
    β₁ = (sum(X .* Y) - mean(X) * sum(Y)) / (sum(X .* X) - sum(X) * mean(X))
    β₀ = mean(Y) - β₁ * mean(X)
    return β₀, β₁
end

# Regression solver
function AutoRegression(Sₜ::Array{Float64}, K_path::Array{Float64})
    # Get primitives
    P = Primitives()

    # Filter for burn in period
    X = log.(K_path[P.B:(P.T - 1)])
    Y = log.(K_path[(P.B + 1):P.T])
    SBₜ = Sₜ[P.B:(P.T - 1)]

    # Instantiate variables
    β = []
    SSR = 0.0
    SST = 0.0

    # Iterate over aggregate states
    for s = 1:2
        # Separate state, generate variables, and run regression
        Xˢ = X[SBₜ .== s]
        Yˢ = Y[SBₜ .== s]
        β̂₀, β̂₁ = CoefficientSolver(Yˢ, Xˢ)

        # Update variables
        SSR += sum((Yˢ .- (β̂₀ .+ β̂₁ .* Xˢ)).^2)
        SST += sum((Yˢ .- mean(Yˢ)).^2)
        β = vcat(β, [β̂₀, β̂₁])
    end

    # Calculate R²
    R² = 1.0 - SSR / SST

    # Return values
    return β, R²
end

# Model solver
function SolveModel(R::Results, verbose::Bool=false)
    # Get primitives
    P = Primitives()
    S = Shocks()

    # Draw shocks, initialize errors, counter
    sᵢₜ, Sₜ = DrawShocks(S, P.N, P.T)
    error = Inf
    i = 0

    # Iterate while not converged
    while (error > P.tol_c || R.R² < P.tol_r) && i <= P.maxit
        # Unpack current values, increment counter
        @unpack a₀, a₁, b₀, b₁, R² = R
        i += 1

        # Solve household problems, update capital path, and run autoregression
        SolveHousehold(R, verbose)
        K_path = CapitalSimulation(R, sᵢₜ, Sₜ)
        (â₀, â₁, b̂₀, b̂₁), R̂² = AutoRegression(Sₜ, K_path)

        # Update error and values
        error = abs(a₀ - â₀) + abs(a₁ - â₁) + abs(b₀ - b̂₀) + abs(b₁ - b̂₁)
        R.a₀ = P.λ * â₀ + (1 - P.λ) * a₀
        R.a₁ = P.λ * â₁ + (1 - P.λ) * a₁
        R.b₀ = P.λ * b̂₀ + (1 - P.λ) * b₀
        R.b₁ = P.λ * b̂₁ + (1 - P.λ) * b₁
        R.R² = R̂²

        # Print statement
        println("****************************************************************************\n")
        println("Aggregate Iteration = ", i, "\n")
        println("Coefficient Error   = ", error, "\n")
        println("Current R²          = ", R.R², "\n")
        println("****************************************************************************\n")
    end

    # Print statement
    if i == P.maxit
        println("****************************************************************************\n")
        println("Aggregate problem did not converge within maximum iterations...\n")
        println("****************************************************************************")
    else
        println("****************************************************************************\n")
        println("Aggregate problem converged in ", i, " iterations!\n")
        println("****************************************************************************\n")
    end

    # Print final results
    println("****************************************************************************\n")
    println("Results:")
    println("a₀ = ", round(R.a₀, digits = 3), ", a₁ = ", round(R.a₁, digits = 3))
    println("b₀ = ", round(R.b₀, digits = 3), ", b₁ = ", round(R.b₁, digits = 3))
    println("R² = ", round(R.R², digits = 3), "\n")
    println("****************************************************************************")
end