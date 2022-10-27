# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: October, 2022

########################################################
#################### HOUSEHOLD PROBLEM #################
########################################################

# Import required packages
using LinearAlgebra, Interpolations, Optim

# Include psueodpanel
include("Initialization.jl")

# Bellman operator for households
function Bellman(P::Primitives, G::Grids, S::Shocks, R::Results)
    # Unpack primitives and grids
    @unpack β, α, δ = P
    @unpack nk, k, ε, ne, K, nK, nz, Z = G
    @unpack uᵍ, uᵇ, M = S
    @unpack policy_func, value_func, a₀, a₁, b₀, b₁ = R

    # Initialize values
    next_policy_func = zeros(nk, ne, nK, nz)
    next_value_func = zeros(nk, ne, nK, nz)

    # Interpolation functions
    k̂ = interpolate(k, BSpline(Linear()))
    v̂ = interpolate(value_func, BSpline(Linear()))

    # Iterate over productivity state space
    for (i_z, z) in enumerate(Z)
        # Check state space
        if i_z == 1
            Lₜ = ε[1] * (1 - uᵍ)
        elseif i_z == 2
            Lₜ = ε[1] * (1 - uᵇ)
        end

        # Iterate over aggregate capital state space
        for (i_K, Kₜ) in enumerate(K)
            # Check state space
            if i_z == 1
                Kp = a₀ + a₁ * log(Kₜ)
            elseif i_z == 2
                Kp = b₀ + b₁ * log(Kₜ)
            end

            # Calculate wages and interest rates
            w = (1 - α) * z * (Kₜ / Lₜ) ^ α
            r = α * z * (Kₜ / Lₜ) ^ (α - 1)

            # Exponentiate and get index
            Kp = exp(Kp)
            i_Kp = GetIndex(Kp, K)

            # Iterate over ε shocks
            for (i_ε, εᵢ) in enumerate(ε)
                # Get row index
                row = i_ε + ne * (i_z - 1)

                # Iterate over individual capital state space
                for (i_k, kₜ) in enumerate(k)
                    # Calculate budget
                    budget = r * kₜ + w * εᵢ + (1.0 - δ) * kₜ

                    # Continuation value for interpolation
                    vp(i_kp) = M[row, 1] * v̂(i_kp, 1 ,i_Kp, 1) + M[row, 2] * v̂(i_kp, 2, i_Kp, 1) + M[row, 3] * v̂(i_kp, 1, i_Kp, 2) + M[row, 4] * v̂(i_kp, 2, i_Kp, 2)
                    v(i_kp) = log(budget - k̂(i_kp)) +  β * vp(i_kp)
                    obj(i_kp) = -v(i_kp)

                    # Bounds for optimization
                    upper = GetIndex(budget, k)
                    lower = 1.0

                    # Optimize objective and update values
                    opt = optimize(obj, lower, upper)

                    # Update policy and value functions
                    next_policy_func[i_k, i_ε, i_K, i_z] = k̂(opt.minimizer[1])
                    next_value_func[i_k, i_ε, i_K, i_z] = -opt.minimum
                end
            end
        end
    end

    # Return values
    return next_policy_func, next_value_func
end

# Get index for interpolation problem
function GetIndex(val::Float64, grid::Array{Float64,1})
    # Initialize
    n = length(grid)
    index = 0

    # Check conditions
    if val <= grid[1]
        # Lowest index
        index = 1
    elseif val >= grid[n]
        # Highest index
        index = n
    else
        # Generate bounds
        index_upper = findfirst(x-> x > val, grid)
        index_lower = index_upper - 1
        val_upper, val_lower = grid[index_upper], grid[index_lower]

        # Get midpoint
        index = index_lower + (val - val_lower) / (val_upper - val_lower)
    end

    # Return value
    return index
end

# Solve household problem
function SolveHousehold(R::Results, verbose::Bool)
    # Instantiate primitives
    P = Primitives()
    G = Grids()
    S = Shocks()

    # Initialize error, counter
    error = Inf
    i = 0

    # Value function iteration
    while error > P.tol_v
        # Increment counter
        i += 1

        # Solve household Bellman, update error
        next_policy_func, next_value_func = Bellman(P, G, S, R)
        error = maximum(abs.(next_value_func .- R.value_func))

        # Update policy and value functions
        R.value_func = next_value_func
        R.policy_func = next_policy_func

        # Print statement
        if verbose
            println("Household Iteration = ", i, " with error ε = ", error, "\n")
        end
    end

    # Print convergence
    println("****************************************************************************\n")
    println("Household problem converged in ", i, " iterations!\n")
    println("****************************************************************************\n")
end

########################################################
###################### MODEL SOLVER ####################
########################################################

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