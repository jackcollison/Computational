# Library for plots
using Plots

# Import model
include("Model.jl")

# Set verbosity
verbose = true

# Initialize and fit
R₀ = Initialize()
@time SolveModel(R₀, verbose)

# Initialize and fit
R₁ = Initialize(10.0, 1.0)
@time SolveModel(R₁, verbose)

# Initialize and fit
R₂ = Initialize(10.0, 2.0)
@time SolveModel(R₂, verbose)

# Initialize and fit
R₃ = Initialize(10.0, 3.0)
@time SolveModel(R₃, verbose)

# Generate plot
plot(R₀.x, label = "Standard")
plot!(R₁.x, label = "TV1 Shocks α = 1")
plot!(R₂.x, label = "TV1 Shocks α = 2")
plot!(R₃.x, label = "TV1 Shocks α = 3")
plot!(title = "Exit Decisions for cᶠ = 10", xlab = "State", ylab = "Exit Probability")

# Initialize and fit
R₀ = Initialize(15.0)
@time SolveModel(R₀, verbose)

# Initialize and fit
R₁ = Initialize(15.0, 1.0)
@time SolveModel(R₁, verbose)

# Initialize and fit
R₂ = Initialize(15.0, 2.0)
@time SolveModel(R₂, verbose)

# Initialize and fit
R₃ = Initialize(15.0, 3.0)
@time SolveModel(R₃, verbose)

# Generate plot
plot(R₀.x, label = "Standard")
plot!(R₁.x, label = "TV1 Shocks α = 1")
plot!(R₂.x, label = "TV1 Shocks α = 2")
plot!(R₃.x, label = "TV1 Shocks α = 3")
plot!(title = "Exit Decisions for cᶠ = 15", xlab = "State", ylab = "Exit Probability")
