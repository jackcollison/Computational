# PS2: Numerical Integration
# Toolbox for quadrature integration
using Optim, ProgressMeter, DataFrames, Distributions

# One-dimensional quadrature integration
function integrate_1d(f, upper_bound, KPU_1d)
# Define functions to translate the (0, 1) interval into appropriate interval
    points = -log.(1 .- KPU_1d[:, :Column1]) .+ upper_bound
    jacobian = 1 ./ (1 .- KPU_1d[:, :Column1])
# sum over grid points
    return sum(KPU_1d[:, :Column2] .* f.(points) .* jacobian)
end

# Two-dimensional quadrature integration
function integrate_2d(f, upper_bound_0, upper_bound_1, KPU_2d)

    points_0 = -log.(1 .- KPU_2d[:, :Column1]) .+ upper_bound_0
    jacobian_0 = 1 ./ (1 .- KPU_2d[:, :Column1])

    points_1 = -log.(1 .- KPU_2d[:, :Column2]) .+ upper_bound_1
    jacobian_1 = 1 ./ (1 .- KPU_2d[:, :Column2])

    return sum(KPU_2d[:, :Column3] .* f.(points_0, points_1) .* jacobian_0 .* jacobian_1)

end

# Standard normal distribution functions
# Standard Normal PDF
function ϕ(x)
    1/sqrt(2 * π) * exp((-1/2)*x^2)
end
# Standard Normal CDF
function Φ(x)
    return cdf(Normal(0, 1), x)
end
# Inverse Standard Normal CDF
function Φ_inverse(p)
    return quantile(Normal(0, 1), p)
end

# Quadrature integration

# Using quadrature integration
function likelihood_quadrature(α₀, α₁, α₂,  β, γ, ρ,
    t, x, z, KPU_1d, KPU_2d)

    result = 0.0
    σ₀ = 1/(1 - ρ)

    if t == 1.0
        result = Φ((-α₀ - x'*β - z[1]*γ)/σ₀)
    elseif t == 2.0
        f_2(ε₀) = ϕ(ε₀/σ₀) / σ₀ * Φ(-α₁ - x'*β - z[2]*γ - ρ * ε₀)
        result = integrate_1d(f_2, -α₀ - x'*β - z[1]*γ, KPU_1d)
    elseif t == 3.0
        f_3(ε₀, ε₁) = ϕ(ε₀/σ₀) / σ₀ * ϕ(ε₁ - ρ*ε₀) * Φ(-α₂ - x'*β - z[3]*γ - ρ*ε₁)
        result = integrate_2d(f_3, -α₀ - x'*β - z[1]*γ, -α₁ - x'*β - z[2]*γ, KPU_2d)
    elseif t == 4.0
        f_4(ε₀, ε₁) = ϕ(ε₀/σ₀) / σ₀ * ϕ(ε₁ - ρ*ε₀) * (1 - Φ(-α₂ - x'*β - z[3]*γ - ρ*ε₁))
        result = integrate_2d(f_4, -α₀ - x'*β - z[1]*γ, -α₁ - x'*β - z[2]*γ, KPU_2d)
    else
        error("Invalid value of t.")
    end

    return result
end


# GHK

# using ghk simulation
function likelihood_ghk(α₀::Float64, α₁::Float64, α₂::Float64,  β::Array{Float64, 1}, γ::Float64, ρ::Float64,t::Float64, x::Array{Float64, 1}, z::Array{Float64, 1}, u₀::Array{Float64, 1}, u₁::Array{Float64, 1}, u₂::Array{Float64, 1})
    n_trials = length(u₀)
    σ₀ = 1/(1 - ρ)
    truncation₀ = Φ((-α₀ - x'*β - z[1]*γ) / σ₀) # evaluates truncation point for first shock probability

    if t == 1.0
        return truncation₀
    else # if t = 2.0 or 3.0 or 4.0
        pr₀ = truncation₀ .+ u₀ * (1 .- truncation₀) # scales uniform rv between zero and the truncation point.
        η₀ = Φ_inverse.(pr₀)
        ε₀ = η₀ .* σ₀

        truncation₁ = Φ.(-α₁ .- x'*β .- z[2]*γ .- ρ.*ε₀) # initializes simulation-specific truncation points

        if t == 2.0
            return sum((1 .- truncation₀) .* truncation₁) / n_trials

        else # if t = 3.0 or 4.0
            pr₁ = truncation₁ .+ u₁ .* (1 .- truncation₁) # scales uniform rv between zero and the truncation point.
            η₁ = Φ_inverse.(pr₁) # initializes first shocks
            ε₁ = ρ .* ε₀ .+ η₁

            truncation₂ = Φ.(-α₂ .- x'*β .- z[3]*γ .- ρ.*ε₁)

            if t == 3.0
                return sum((1 .- truncation₀) .* (1 .- truncation₁) .* truncation₂) / n_trials
            else # t = 4.0
                return sum((1 .- truncation₀) .* (1 .- truncation₁) .* (1 .- truncation₂)) / n_trials
            end
        end
    end
end

# Accept-Reject

function likelihood_accept_reject(α₀::Float64, α₁::Float64, α₂::Float64,  β::Array{Float64, 1}, γ::Float64, ρ::Float64,
    t::Float64, x::Array{Float64, 1}, z::Array{Float64, 1}, ε₀::Array{Float64, 1}, ε₁::Array{Float64, 1}, ε₂::Array{Float64, 1})

    # initialize count variable
    count = 0
    σ₀ = 1/(1 - ρ)
    b₀ = -α₀ - x'*β - z[1]*γ
    b₁ = -α₁ - x'*β - z[2]*γ
    b₂ = -α₂ - x'*β - z[3]*γ 
    a₀ = (ε₀ .> b₀)
    a₁ = (ε₁ .> b₁)
    a₂ = a₀ .* a₁

    # based on the value of t counts the number of accepted simulations
    if t == 1.0
        count = sum(Φ.(b₀ / σ₀))
    elseif t == 2.0
        if length((a₀ .== 1)) > 0
            count = sum(a₀ .* Φ.(b₁ .- ρ .* ε₀)) / length((a₀ .== 1))
        end
    elseif t == 3.0
        if length((a₂ .== 1)) > 0
            count = sum(a₂ .* Φ.(b₂ .- ρ .* ε₁)) / length((a₂ .== 1))
        end
    elseif t == 4.0
         if length((a₂ .== 1)) > 0
            count = sum(a₂ .* (1 .- Φ.(b₂ .- ρ .* ε₁))) / length((a₂ .== 1))
         end
    else
        error("Invalid value of t.")
    end

    # returns the frequency of the accepted simulations
    return count # / length(ε₀)
end
