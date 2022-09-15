########################################################
################# DIAGNOSTIC FUNCTIONS #################
########################################################

# Required packages
using Plots

# Include model
include("Model.jl")

# Cross sectional wealth distribution functionality
function WealthDistribution(Results::res, a::Float64)
    # Unpack primitives
    @unpack β, α, S, ns, Π, A, na = Primitives()
    
    # Get wealth distribution
    W = S .+ res.pol_func .- a
end

# Gini index functionality
function Gini(Results::res)
    return 0
end

# Lorenz Curve functionality
function LorenzCurve(Results::res)
    return 0
end

# First best welfare and λ functionality
function λ(Results::res)
    # Unpack primitives, first best welfare
    @unpack β, α, S, ns, Π, A, na = Primitives()
    WFB = 0

    # Return lambda
    ((WFB + 1 / ((1 - α) * (1 - β))) / (res.value_func .+ 1 / ((1 - α) * (1 - β))))^(1 / (1 - α)) .- 1
end