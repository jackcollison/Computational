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

# Set layout
layout = Layout(
    ;scene=attr(
        ;xaxis=attr(;title="q̄ᵢ"), 
        yaxis=attr(;title="q̄ⱼ"),
        zaxis=attr(;title="Π(qᵢ, qⱼ)")
    )
)

# Surface plot for static problem profit
plot(surface(z=V, x=q̄, y=q̄), layout)

# Set layout
layout = Layout(
    ;scene=attr(
        ;xaxis=attr(;title="q̄ᵢ"), 
        yaxis=attr(;title="q̄ⱼ"),
        zaxis=attr(;title="q(qⱼ)")
    )
)

# Surface plot for static problem quantity
plot(surface(z=Q, x=q̄, y=q̄), layout)

# Set layout
layout = Layout(
    ;scene=attr(
        ;xaxis=attr(;title="q̄ᵢ"), 
        yaxis=attr(;title="q̄ⱼ"),
        zaxis=attr(;title="x(q̄ᵢ, q̄ⱼ)"),
        main=attr()
    )
)

# Solve firms' problems
δ = 0.0
Xᵒ, Vᵒ = SolveModel(p, V, δ; verbose=true)
plot(surface(z=Xᵒ, x=q̄, y=q̄), layout)

# Solve firms' problems
δ = 0.01
Xᵒ, Vᵒ = SolveModel(p, V, δ; verbose=true)
plot(surface(z=Xᵒ, x=q̄, y=q̄), layout)

# Solve firms' problems
δ = 0.1
Xᵒ, Vᵒ = SolveModel(p, V, δ; verbose=true)
plot(surface(z=Xᵒ, x=q̄, y=q̄), layout)

# Solve firms' problems
δ = 0.3
Xᵒ, Vᵒ = SolveModel(p, V, δ; verbose=true)
plot(surface(z=Xᵒ, x=q̄, y=q̄), layout)

# Transition probabilities
P = [𝐏(p, i, Int64(i - j), Xᵒ[])]