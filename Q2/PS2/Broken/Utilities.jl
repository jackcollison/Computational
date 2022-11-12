# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: November, 2022

# Import libraries
using Distributions

# Transformations
function ρᵤ(u::Float64, a::Float64, b::Float64)
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
function Quadrature(f, w::Array{Float64}, u::Array{Float64}, a::Array{Float64}=repeat([-Inf], size(u, 2)), b::Array{Float64}=repeat([Inf], size(u, 2)))
    # Find dimensions
    dims = size(u, 2)

    # Initialize storage
    p = zeros((size(u, 1), dims))
    g = zeros((size(u, 1), dims))

    # Loop over dimensions
    for i = 1:dims
        # Find normalized points
        pᵢ, gᵢ = ρᵤ.(u[:,i], a[i], b[i])
        p[:,i] .= pᵢ
        g[:,i] .= gᵢ
    end

    # Return value
    return sum(w .* f.(p...) .* prod.(eachrow(g)))
end