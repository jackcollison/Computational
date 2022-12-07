# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: November, 2022

# Include libraries and model
using PlotlyJS, LaTeXStrings
include("Model.jl")

# Unpack primitives
p = Primitives()
@unpack Q̄, q̄, n, β = p

# Static problem
V = [Πᶜ(p, Q̄[i, j][1], Q̄[i, j][2]) for i in 1:n, j in 1:n]
Q = [qᶜ(p, Q̄[i, j][1], Q̄[i, j][2]) for i in 1:n, j in 1:n]

# Surface plot for static problem
plot(surface(z=V, x=q̄, y=q̄))
plot(surface(z=Q, x=q̄, y=q̄))

# Solve firms' problems
δ = 0.0
Xᵒ, Vᵒ = SolveModel(p, V, δ; verbose=true)
plot(surface(z=Xᵒ, x=q̄, y=q̄))

# Solve firms' problems
δ = 0.01
Xᵒ, Vᵒ = SolveModel(p, V, δ; verbose=true)
plot(surface(z=Xᵒ, x=q̄, y=q̄))

# Solve firms' problems
δ = 0.1
Xᵒ, Vᵒ = SolveModel(p, V, δ; verbose=true)
plot(surface(z=Xᵒ, x=q̄, y=q̄))

# Solve firms' problems
δ = 0.3
Xᵒ, Vᵒ = SolveModel(p, V, δ; verbose=true)
plot(surface(z=Xᵒ, x=q̄, y=q̄))