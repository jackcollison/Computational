# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: October, 2022

########################################################
#################### HOUSEHOLD PROBLEM #################
########################################################

# Import required packages
using LinearAlgebra, Interpolations, Optim

# Include psueodpanel
include("PseudoPanel.jl")

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