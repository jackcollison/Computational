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
    P = Primitives()
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
    K_path = vcat(P.Kˢˢ, zeros(P.T - 1))
    previous_k = repeat(P.Kˢˢ, P.N)
    previous_K = P.Kˢˢ

    # Iterate over time
    for t = 2:P.T
        # Get index of previous aggregate capital, initialize current individual capital
        i_K = GetIndex(previous_K, G.K)
        current_k = zeros(P.N)

        # Iterate over individuals
        for i = 1:P.n
            # Get index of policy functions and update current individual capital
            i_k = GetIndex(previous_k[i], G.k)
            current_k[i] = R.policy_func[i_k, sᵢₜ[i, t], i_K, Sₜ[t]]
        end

        # Update aggregate capital path
        K_path[t] = sum(current_k) / P.N

        # Update previous capital
        previous_k = current_k
        previous_K = K_path[t]
    end

    # Return value
    return K_path
end

# Generate variables
function GenerateVariables(Kˢ::Array{Float64})
    return Kˢ[1:(len(Kˢ) - 1)], Kˢ[2:len(Kˢ)]
end

# Regression helper for coefficients and R²
function CoefficientSolver(Y::Array{Float64}, X::Array{Float64})
    # Agument X matrix
    Y = log.(Y)
    X = hcat(ones(len(Y)), log.(X))

    # Solve for coefficients and return
    return inv(X' * X) * X' * Y
end

# Regression solver
function AutoRegression(R::Results, Sₜ::Array{Float64}, K_path::Array{Float64})
    # Get primitives
    P = Primitives()

    # Filter for burn in period
    KB = K_path[(P.B - 1):P.T]
    SBₜ = Sₜ[(P.B - 1):P.T]

    # Instantiate variables
    β = []
    SSR = 0.0
    SST = 0.0

    # Iterate over aggregate states
    for s = 1:2
        # Separate state, generate variables, and run regression
        Kˢ = KB[SBₜ .== s]
        Xˢ, Yˢ = GenerateVariables(Kˢ)
        βˢ = CoefficientSolver(Xˢ, Yˢ)

        # Update variables
        SSR += sum((Yˢ - Xˢ * βˢ).^2)
        SST += sum((Yˢ .- mean(Yˢ).^2))
        β = vcat(β, βˢ)
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
    G = Grids()
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
        â₀, â₁, b̂₀, b̂₁, R̂² = AutoRegression(R, Sₜ, K_path)

        # Update error and values
        error = abs(a₀ - â₀) + abs(a₁ - â₁) + abs(b₀ - b̂₀) + abs(b₁ - b̂₁)
        R.a₀ = P.λ * â₀ + (1 - λ) * a₀
        R.a₁ = P.λ * â₁ + (1 - λ) * a₁
        R.b₀ = P.λ * b̂₀ + (1 - λ) * b₀
        R.b₁ = P.λ * b̂₁ + (1 - λ) * b₁
        R.R² = R̂²

        # Print statement
        if verbose
            println("****************************************************************************\n")
            println("Aggregate Iteration = ", i, "\n")
            println("Coefficient Error   = ", error, "\n")
            println("Current R²          = ", R.R², "\n")
            println("****************************************************************************\n")
        end
    end

    # Print statement
    if verbose && i == P.maxit
        println("******************************************************************\n")
        println("Aggregate problem did not converge within maximum iterations...\n")
        println("******************************************************************")
    elseif verbose
        println("******************************************************************\n")
        println("Aggregate problem converged in ", i, " iterations!\n")
        println("******************************************************************")
    end
end