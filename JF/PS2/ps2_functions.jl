using Distributions, Statistics

# CDF of Normal dist
Φ(x::Float64) = cdf(Normal(), x)
# PDF of Normal dist
ϕ(x::Float64) = pdf(Normal(), x)
# Inverse of CDF of Normal dist
Φ_inv(q::Float64) = quantile(Normal(), q)

# set primitive set of parameters from one parameter θ
function parameter_sep(θ::Vector{Float64})
    ρ = θ[1]
    α = θ[2:4]
    γ = θ[5]
    β = θ[6:end]
    return ρ, α, γ, β
end

# calculate mean payoff
function calc_payoff(θ::Vector{Float64}, X::Vector{Float64}, Z::Vector{Float64})
    # get parameter
    ρ, α, γ, β = parameter_sep(θ)

    # calculate payoff from T=t as V_t
    V_0 = α[1] + X' * β + Z[1] * γ
    V_1 = α[2] + X' * β + Z[2] * γ
    V_2 = α[3] + X' * β + Z[2] * γ

    return V_0, V_1, V_2
end

# transform quadrature nodes
function transform_node(Q::Vector{Float64}, upper::Float64)
    return log.(Q) .+ upper
end

# choice probability using Gaussian Quadrature
function choice_prob_quadrature(θ::Vector{Float64}, X::Vector{Float64}, Z::Vector{Float64}, Q1::Matrix{Float64}, Q2::Matrix{Float64})
    # get primitive parameter
    ρ, α, γ, β = parameter_sep(θ)
    σ = 1 / abs(1-ρ) # sd of ϵ₀

    # calculate payoff
    V_0, V_1, V_2 = calc_payoff(θ, X, Z)

    # Pr(T=1)
    prob_T1 = Φ(- V_0 / σ)

    # Pr(T=2)
    ϵ_1d = transform_node(Q1[:,1], V_0) # transformed nodes (1-dimensional)
    jacob_1d = 1 ./ Q1[:,1] # jacobian
    w_1d = Q1[:,2] # weights for 1-dimensional quadrature
    integral_points_T2 = Φ.(- V_1 .- ρ .* ϵ_1d) .* ϕ.(ϵ_1d ./ σ) ./ σ .* jacob_1d .* w_1d # weighted points
    prob_T2 = sum(integral_points_T2) # Pr(T=2)

    # Pr(T=3)
    ϵ₀ = transform_node(Q2[:,1], V_0) # transfomed nodes of ϵ₀ (2-dimensional)
    ϵ₁ = transform_node(Q2[:,2], V_1) # transformed nodes of ϵ₁ (2-dimensional)
    jacob_2d = (1 ./ Q2[:,1]) .* (1 ./ Q2[:,2]) # jacobian
    w_2d = Q2[:,3] # weights for 2-dimensional quadrature
    integral_points_T3 = Φ.(- V_2 .- ρ .* ϵ₁) .* ϕ.(ϵ₁ .- ρ .* ϵ₀) .* ϕ.(ϵ₀ ./ σ) ./ σ .* jacob_2d .* w_2d
    prob_T3 = sum(integral_points_T3) # Pr(T=3)

    # Pr(T=4)
    integral_points_T4 = Φ.(V_2 .- ρ .* ϵ₁) .* ϕ.(ϵ₁ .- ρ .* ϵ₀) .* ϕ.(ϵ₀ ./ σ) ./ σ .* jacob_2d .* w_2d
    prob_T4 = sum(integral_points_T4) # Pr(T=4)

    return [prob_T1, prob_T2, prob_T3, prob_T4] # vector of choice probability
end

# choice probability using GHK method
function choice_prob_ghk(θ::Vector{Float64}, X::Vector{Float64}, Z::Vector{Float64}, U::Matrix{Float64})
    # get primitive parameter
    ρ, α, γ, β = parameter_sep(θ)
    σ = 1 / abs(1-ρ) # sd of ϵ₀

    # get random draw from uniform distribution
    u₀ = U[:,1]
    u₁ = U[:,2]

    # calculate payoff
    V_0, V_1, V_2 = calc_payoff(θ, X, Z)

    # Pr(T=1)
    prob_T1 = Φ(- V_0 / σ)

    # Pr(T=2)
    ϵ₀ = Φ_inv.(u₀ .* Φ(V_0 / σ)) # transform uniform dist to truncated normal
    prob_T2 = mean(Φ(V_0 / σ) .* Φ.(- V_1 .- ρ .* ϵ₀))

    # Pr(T=3)
    η₁ = Φ_inv.(u₁ .* Φ.(V_1 .- ρ .* ϵ₀)) # transform uniform dist to truncated normal given ϵ₀
    ϵ₁ = ρ .* ϵ₀ .+ η₁ # construct ϵ₁
    prob_T3 = mean(Φ(V_0 / σ) .* Φ.(V_1 .- ρ .* ϵ₀) .* Φ.(- V_2 .- ρ .* ϵ₁))

    # Pr(T=4)
    prob_T4 = mean(Φ(V_0 / σ) .* Φ.(V_1 .- ρ .* ϵ₀) .* Φ.(V_2 .- ρ .* ϵ₁))

    return [prob_T1, prob_T2, prob_T3, prob_T4] # vector of choice probability
end

# choice probability using accept/reject
function choice_prob_ar(θ::Vector{Float64}, X::Vector{Float64}, Z::Vector{Float64}, U::Matrix{Float64})
    # get primitive parameter
    ρ, α, γ, β = parameter_sep(θ)
    σ = 1 / abs(1-ρ) # sd of ϵ₀

    # get random draw from standard normal distribution
    η₀ = U[:,1]
    η₁ = U[:,2]
    η₂ = U[:,3]
    ϵ₀ = σ .* η₀
    ϵ₁ = ρ .* ϵ₀ + η₁
    ϵ₂ = ρ .* ϵ₁ + η₂

    # calculate payoff
    V_0, V_1, V_2 = calc_payoff(θ, X, Z)

    # Pr(T=1)
    prob_T1 = Φ(- V_0 / σ)

    # Pr(T=2)
    prob_T2 = mean(ϵ₀ .< V_0 .&& ϵ₁ .< - V_1)

    # Pr(T=3)
    prob_T3 = mean(ϵ₀ .< V_0 .&& ϵ₁ .< V_1 .&& ϵ₂ .< - V_2)

    # Pr(T=4)
    prob_T4 = mean(ϵ₀ .< V_0 .&& ϵ₁ .< V_1 .&& ϵ₂ .< V_2)

    return [prob_T1, prob_T2, prob_T3, prob_T4] # vector of choice probability
end

# log-likelihood
function LL(θ::Vector{Float64}, Y::Matrix{Float64}, X::Matrix{Float64}, Z::Matrix{Float64}, Q1::Matrix{Float64}, Q2::Matrix{Float64}, U::Matrix{Float64}; method::String = "quadrature")
    LL = 0.0 # reset value of LL
    N = size(Y)[1] # number of observations

    if method == "quadrature"
        for i = 1:N
            choice_prob = choice_prob_quadrature(θ, X[i,:], Z[i,:], Q1, Q2)
            likelihood = sum(Y[i,:] .* choice_prob)
            LL += log(likelihood)
        end
    elseif method == "ghk"
        for i = 1:N
            choice_prob = choice_prob_ghk(θ, X[i,:], Z[i,:], U)
            likelihood = sum(Y[i,:] .* choice_prob)
            LL += log(likelihood)
        end
    elseif method == "ar"
        for i = 1:N
            choice_prob = choice_prob_ar(θ, X[i,:], Z[i,:], U)
            likelihood = sum(Y[i,:] .* choice_prob)
            LL += log(likelihood)
        end
    else
        print("Error: no matching method")
    end

    return LL    
end