using LinearAlgebra

# choice probability
function choice_prob(x::Vector{Float64}, β::Vector{Float64})
    exp(x' * β) / (1 + exp(x' * β)) # binary choice logit
end

# log-likelihood
function LL(β::Vector{Float64}, Y::Vector{Float64}, X::Matrix{Float64})
    LL = 0.0 # reset value of LL
    N = length(Y) # number of observations
    for i = 1:N
        y = Y[i] # Y_i
        x = X[i, :] # X_i
        Λ = choice_prob(x, β) # choice probability
        LL += y * log(Λ) + (1-y) * log(1-Λ)
    end
    return LL
end

# score (first-derivative of log-likelihood)
function Score(β::Vector{Float64}, Y::Vector{Float64}, X::Matrix{Float64})
    K = length(β) # number of parameters
    N = length(Y) # number of observations
    S = zeros(K) # reset value of score
    for i = 1:N
        y = Y[i]
        x = X[i, :]
        Λ = choice_prob(x, β)
        S += x .* (y - Λ)
    end
    return S
end

# Hessian (second-derivative of log-likelihood)
function Hessian(β::Vector{Float64}, Y::Vector{Float64}, X::Matrix{Float64})
    K = length(β)
    N = length(Y)
    H = zeros(K, K) # reset value of Hessian
    for i = 1:N
        y = Y[i]
        x = X[i, :]
        Λ = choice_prob(x, β)
        H -= x * x' .* Λ .* (1 - Λ)
    end
    return H
end

function Newton(β_init::Vector{Float64}, Y::Vector{Float64}, X::Matrix{Float64}; err = 100.0, tol = 1e-10)
    β = β_init # initial value of β
    i::Int64 = 0 # reset count

    while err > tol
        g = Score(β, Y, X) # score
        H = Hessian(β, Y, X) # Hessian
        β_new = β - H \ g # new β
        err = maximum(abs.(β_new - β)) # calculate distance b/w β_new and β
        i += 1 # add count
        ρ = 1 - 1 / (i+1) # update parameter
        β = ρ .* β_new + (1-ρ) .* β  # update β
    end

    println("Converged after ", i, " iterations")

    return β
end

# numerical score
function Score_numerical(β::Vector{Float64}, Y::Vector{Float64}, X::Matrix{Float64}; δ = 1e-12)
    K = length(β)
    S = zeros(K) # reset value of score
    for i = 1:K
        β_plus = copy(β)
        β_minus = copy(β)
        β_plus[i] = β[i] + δ # add small value to i-th element
        β_minus[i] = β[i] - δ # subtract small value from i-th element
        LL_plus = LL(β_plus, Y, X) # LL when moving β_i forward slightly
        LL_minus = LL(β_minus, Y, X) # LL when moving β_i backward slightly
        S[i] = (LL_plus - LL_minus) / (2 * δ) # numerical derivative of LL wrt β_i
    end
    return S
end

# numerical Hessian
function Hessian_numerical(β::Vector{Float64}, Y::Vector{Float64}, X::Matrix{Float64}; δ = 1e-12)
    K = length(β)
    H = zeros(K, K) # reset value of Hessian
    for i = 1:K
        β_plus = copy(β)
        β_minus = copy(β)
        β_plus[i] = β[i] + δ # add small value to i-th element
        β_minus[i] = β[i] - δ # subtract small value from i-th element
        S_plus = Score_numerical(β_plus, Y, X; δ)
        S_minus = Score_numerical(β_minus, Y, X; δ)
        H[:,i] = (S_plus - S_minus) ./ (2 * δ) # numerical derivative of Score wrt β_i
    end
    return H
end