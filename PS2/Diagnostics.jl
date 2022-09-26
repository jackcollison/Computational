# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: September, 2022

########################################################
################# DIAGNOSTIC FUNCTIONS #################
########################################################

# Required packages
using Printf

# Include model
include("Model.jl")

# Wealth distribution functionality
function WealthDistribution(res::Results)
    # Unpack primitives and generate distributions
    @unpack β, α, S, ns, Π, A, na = Primitives()
    vcat([A .+ S[1] res.μ[:, 1]], [A .+ S[2] res.μ[:, 2]])
end

# Lorenz Curve functionality
function LorenzCurve(W::Array{Float64, 2})
    # Calulate cumulative distributions
    W = W[sortperm(W[:, 1]), :]
    [cumsum(W[:, 2]) cumsum(W[:, 1] .* W[:, 2] ./ sum(W[:, 1] .* W[:, 2]))]
end

# Gini index functionality
function Gini(L::Array{Float64, 2})
    sum((L[:, 1] .- L[:, 2]) .* (L[:, 1] .- [0.0 ; L[1:(size(L, 1) - 1), 1]])) * 2
end

# First best welfare calculation
function FBWelfare(res::Results)
    # Unpack primitives and compute welfare
    @unpack β, α, S, ns, Π, A, na = Primitives()
    ((([0.9434 0.0566]⋅S)^(1 - α) - 1) / (1 - α)) / (1 - β)
end

# First best welfare and λ functionality
function λ(res::Results)
    # Unpack primitives, first best welfare
    @unpack β, α, S, ns, Π, A, na = Primitives()
    WFB = FBWelfare(res)

    # Return λ
    ((WFB + 1 / ((1 - α) * (1 - β))) ./ (res.value_func .+ 1 / ((1 - α) * (1 - β)))).^(1 / (1 - α)) .- 1
end

# Welfare comparison
function WelfareComparision(res::Results, λ::Array{Float64, 2})
    # Calculate welfares
    WFB = FBWelfare(res)
    WINC = sum([res.value_func[:,s]⋅res.μ[:,s] for s in 1:ns])
    WG = sum([λ[:,s]⋅res.μ[:,s] for s in 1:ns])

    # Formatting
    println("\n***************************************************\n")
    @printf "Wᶠᵇ = %-8.6g Wⁱⁿᶜ = %-8.6f WG= %.6f\n\n" WFB WINC WG
    println("*****************************************************\n")
end

# Proportion of those who prefer complete markets
function PreferComplete(res::Results, λ::Array{Float64, 2})
    # Calculate and return
    pc = sum([(λ[:,s] .> 0)⋅res.μ[:,s] for s in 1:ns])

    # Formatting
    println("*****************************************************\n")
    @printf "P(Complete) = %-8.6g\n\n" pc
    println("*****************************************************\n")
end