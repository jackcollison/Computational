########################################################
################# DIAGNOSTIC FUNCTIONS #################
########################################################

# Required packages
using Plots, Printf

# Include model
include("Model.jl")

# Cross sectional wealth distribution functionality
function WealthDistribution(res::Results)
    return 0
end

# Lorenz Curve functionality
function LorenzCurve(W::Array{Float64, 2})
    return 0
end

# Gini index functionality
function Gini(res::Results)
    return 0
end

# First best welfare calculation
function FBWelfare(res::Results)
    return 0
end

# First best welfare and λ functionality
function λ(res::Results)
    # Unpack primitives, first best welfare
    @unpack β, α, S, ns, Π, A, na = Primitives()
    WFB = FBWelfare(res)

    # Return lambda
    ((WFB + 1 / ((1 - α) * (1 - β))) / (res.value_func .+ 1 / ((1 - α) * (1 - β)))).^(1 / (1 - α)) .- 1
end

# Welfare comparison
function WelfareComparision(res::Results, λ::Array{Float64, 2})
    # Calculate welfares
    WFB = FBWelfare(res)
    WINC = sum([res.value_func[:,s]⋅res.μ[:,s] for s in 1:ns])
    WG = sum([λ[:,s]⋅res.μ[:,s] for s in 1:ns])

    # Formatting
    println("\n******************************************************************\n")
    @printf "Wᶠᵇ = %-8.6g Wⁱⁿᶜ = %-8.6f WG= %.6f\n\n" WFB WINC WG
    println("******************************************************************\n")
end

# Proportion of those who prefer complete markets
function PreferComplete(res::Results, λ::Array{Float64, 2})
    # Calculate and return
    pc = sum([λ[:,s]⋅res.μ[:,s] for s in 1:ns])

    # Formatting
    println("\n******************************************************************\n")
    @printf "P(Complete) = %-8.6g\n\n" pc
    println("******************************************************************\n")
end