# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: November, 2022

# Import libraries
using Distributions

# Transformations
function ρ(u::Float64, a::Float64, b::Float64)
    # Check cases for bounds
    if isinf(a) && isinf(b)
        # Return unbounded support
        return -log((1 - u) / u), -(u / (1 - u)) * (-1 / u - (1 - u) / (u^2))
    elseif !isinf(a) && isinf(b)
        # Return support bounded below
        return -log(1 - u) + a, 1 / (1 - u)
    elseif isinf(a) && !isinf(b)
        # Return support bounded above
        return log(u) + b, 1 / u
    else
        # Return bounded support
        return (b - a) * u + a, b - a
    end
end

# Quadrature method
function Quadrature(f, w::Array{Float64}, u::Array{Float64}; a::Array{Float64}=nothing, b::Array{Float64}=nothing)
    # Find dimensions
    dims = size(u, 2)

    # Check lower bounds
    if a === nothing
        # Fill lower bounds
        a = repeat([-Inf], dims)
    end

    # Check upper bounds
    if b === nothing
        # Fill lower bounds
        b = repeat([Inf], dims)
    end

    # Initialize storage
    p = zeros((size(u, 1), dims))
    g = zeros((size(u, 1), dims))

    # Loop over dimensions
    for i = 1:dims
        # Find normalized points
        pᵢ, gᵢ = ρ.(u[:,i], a[i], b[i])
        p[:,i] = pᵢ
        g[:,i] = gᵢ
    end

    # Return value
    if dims == 1
        # One dimensional case
        return sum(w .* f.(p) .* g)
    elseif dims == 2
        # Two dimensional case
        return sum(w .* f.(p[:,1], p[:,2]) .* g[:,1] .* g[:,2])
    end
end