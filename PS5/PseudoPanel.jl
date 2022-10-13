# Author: Jack Collison
# Course: ECON899: Recent Advances in Economics
# Date: October, 2022

########################################################
################## GENERATE PSEUDOPANEL ################
########################################################

# Import required packages
using Random, Distributions

# Include initialization
include("Initialization.jl")

# Update state
function UpdateState(M::Array{Float64, 2}, sᵢₜ::Float64, εᵢₜ::Float64)
    # Update transitions
    Π¹¹ = M[1, 1]
    Π⁰⁰ = M[2, 2]

    # Check idiosyncratic states
    if sᵢₜ == 1 && εᵢₜ < Π¹¹
        return 1
    elseif sᵢₜ  == 1 && εᵢₜ > Π¹¹
        return 2
    elseif sᵢₜ  == 2 && εᵢₜ < Π⁰⁰
        return 2
    elseif sᵢₜ  == 2 && εᵢₜ > Π⁰⁰
        return 1
    end
end

# Draw random shocks
function DrawShocks(S::Shocks, N::Int64, T::Int64)
    # Unpack primitives
    @unpack Πᵍᵍ, Πᵇᵇ, Mᵍᵍ, Mᵍᵇ, Mᵇᵍ, Mᵇᵇ = S

    # Draw random shock
    Random.seed!(0)
    F = Uniform(0, 1)

    # Allocate space for shocks and initialize
    sᵢₜ = zeros(N, T)
    Sₜ = zeros(T)
    sᵢₜ[:,1] .= 1
    Sₜ[1] = 1

    # Iterate over time
    for t = 2:T
        # Draw aggregate shock
        εₜ = rand(F)

        # Check cases
        if Sₜ[t - 1] == 1 && εₜ < Πᵍᵍ
            Sₜ[t] = 1
        elseif Sₜ[t - 1] == 1 && εₜ > Πᵍᵍ
            Sₜ[t] = 2
        elseif Sₜ[t - 1] == 2 && εₜ < Πᵇᵇ
            Sₜ[t] = 2
        elseif Sₜ[t - 1] == 2 && εₜ > Πᵇᵇ
            Sₜ[t] = 1
        end

        # Iterate over individuals
        for n = 1:N
            # Draw idiosyncratic shock
            εᵢₜ = rand(F)

            # Check aggregate state today and tomorrow
            if Sₜ[t - 1] == 1 && Sₜ[t] == 1
                sᵢₜ[n, t] = UpdateState(Mᵍᵍ, sᵢₜ[n, t - 1], εᵢₜ)
            elseif Sₜ[t - 1] == 1 && Sₜ[t] == 2
                sᵢₜ[n, t] = UpdateState(Mᵍᵇ, sᵢₜ[n, t - 1], εᵢₜ)
            elseif Sₜ[t - 1] == 2 && Sₜ[t] == 1
                sᵢₜ[n, t] = UpdateState(Mᵇᵍ, sᵢₜ[n, t - 1], εᵢₜ)
            elseif Sₜ[t - 1] == 2 && Sₜ[t] == 2
                sᵢₜ[n, t] = UpdateState(Mᵇᵇ, sᵢₜ[n, t - 1], εᵢₜ)
            end
        end
    end

    # Return states
    return sᵢₜ, Sₜ
end