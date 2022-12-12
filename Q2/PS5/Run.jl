# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: December, 2022

# Include libraries and model
using PlotlyJS, LaTeXStrings
include("Model.jl")

# Unpack primitives
p = Primitives()
@unpack Q̄, q̄, n, β = p

# Static problem
V = [Πᶜ(p, Q̄[i, j][1], Q̄[i, j][2]) for i in 1:n, j in 1:n]
Q = [qᶜ(p, Q̄[i, j][1], Q̄[i, j][2]) for i in 1:n, j in 1:n]

# Generate plot
GeneratePlot(q̄, q̄, V, "Π(qᵢ, qⱼ)")
GeneratePlot(q̄, q̄, Q, "q(qⱼ)")

# Solve firms' problems
δ = 0.0
Xᵒ, Vᵒ = SolveModel(p, V, δ; verbose=true)
GeneratePlot(q̄, q̄, Xᵒ, "x(q̄ᵢ, q̄ⱼ)")

# Simulate distribution
S = Simulate(p, Xᵒ)
GeneratePlot(q̄, q̄, S, "μ(q̄ᵢ, q̄ⱼ)")

# Solve firms' problems
δ = 0.01
Xᵒ, Vᵒ = SolveModel(p, V, δ; verbose=true)
GeneratePlot(q̄, q̄, Xᵒ, "x(q̄ᵢ, q̄ⱼ)")

# Simulate distribution
S = Simulate(p, Xᵒ)
GeneratePlot(q̄, q̄, S, "μ(q̄ᵢ, q̄ⱼ)")

# Solve firms' problems
δ = 0.1
Xᵒ, Vᵒ = SolveModel(p, V, δ; verbose=true)
GeneratePlot(q̄, q̄, Xᵒ, "x(q̄ᵢ, q̄ⱼ)")

# Simulate distribution
S = Simulate(p, Xᵒ)
GeneratePlot(q̄, q̄, S, "μ(q̄ᵢ, q̄ⱼ)")

# Solve firms' problems
δ = 0.3
Xᵒ, Vᵒ = SolveModel(p, V, δ; verbose=true)
GeneratePlot(q̄, q̄, Xᵒ, "x(q̄ᵢ, q̄ⱼ)")

# Simulate distribution
S = Simulate(p, Xᵒ)
GeneratePlot(q̄, q̄, S, "μ(q̄ᵢ, q̄ⱼ)")